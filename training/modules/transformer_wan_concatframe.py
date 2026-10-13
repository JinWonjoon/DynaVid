from typing import Any, Dict, List, Tuple, Optional, Union
from einops import rearrange, repeat
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import WanTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, PixArtAlphaTextProjection
from diffusers.models.transformers.transformer_wan import WanImageEmbedding, WanAttention, WanAttnProcessor, WanTransformerBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.attention import FeedForward

from training.modules.multi_modal_cross_attn_patch import upgrade_wan_blocks_for_mm
from training.modules.sep_video_flow_attn_joint_patch import upgrade_wan_blocks_for_sep_video_flow_attn_joint
from training.modules.sep_video_flow_patch import upgrade_wan_blocks_for_sep_video_flow

import pdb

logger = logging.get_logger(__name__)



class WanTransformer3DModel_ConcatFrame(WanTransformer3DModel):
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        use_multi_modal_cross_attn=False, 
        use_gate=False, 
        mm_mode='base', 
        use_separate_rope_emb=False, 
        use_separate_modality_emb=False, 
        use_separate_projection_attn_joint=False,
        use_separate_projection=False,  # NEW
        sep_LN=True,                    # NEW
        sep_FFN=True,                   # NEW
        # sep_Attn=False,                 # NEW
        *model_args, 
        **kwargs
    ):
        assert not (use_multi_modal_cross_attn and use_separate_projection), "Do not use multi-modal cross-attention and separate projection for frame-concat model simultaneously."
        assert not (use_separate_projection_attn_joint and use_separate_projection), "Do not use use_separate_projection_attn_joint and separate projection simultaneously."
        
        # 1. 원래 WanTransformer3DModel 로드
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        inner_dim = model.config.num_attention_heads * model.config.attention_head_dim
            
        # 2. Frame Concat에 맞게 수정
        model.use_separate_rope_emb = use_separate_rope_emb
        model.use_separate_modality_emb = use_separate_modality_emb
        if use_separate_modality_emb:
            # Create separate modality embeddings for video and flow
            video_mod_emb = nn.Parameter(torch.zeros(1, 1, inner_dim))
            flow_mod_emb  = nn.Parameter(torch.zeros(1, 1, inner_dim))
            nn.init.normal_(video_mod_emb, std=0.02)
            nn.init.normal_(flow_mod_emb,  std=0.02)
            
            model.video_mod_emb = video_mod_emb
            model.flow_mod_emb  = flow_mod_emb
        
        # 3. Divide proj_out layer into two separate layers for video and flow
        original_proj_out = model.proj_out
        model.proj_out_flow = copy.deepcopy(original_proj_out)
        
        # 4. multi-modal cross-attention Upgrade
        if use_multi_modal_cross_attn:
            upgrade_wan_blocks_for_mm(model, inner_dim, eps=model.config.eps, v_out_scale=0.1, use_gate=use_gate, mm_mode=mm_mode)
        
        model.use_separate_projection_attn_joint = use_separate_projection_attn_joint
        model.use_separate_projection = use_separate_projection
        model.sep_LN = sep_LN
        model.sep_FFN = sep_FFN
        # model.sep_Attn = sep_Attn
        
        # attn = model.blocks[0].attn1
        # print("attn1 processor:", type(attn.processor))
        

        if use_separate_projection_attn_joint:
            upgrade_wan_blocks_for_sep_video_flow_attn_joint(
                model
            )
        elif use_separate_projection:
            upgrade_wan_blocks_for_sep_video_flow(
                model,
                sep_LN=sep_LN,
                sep_FFN=sep_FFN,
                # sep_Attn=sep_Attn,
            )
        
        return model
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        flow_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
                
        # # DEBUG
        # print(f"RANK {torch.distributed.get_rank()} || Current timestep: {timestep}")

        # TODO.Concat the flow_hidden_states with hidden_states along the frame dimension
        # embed video frames and flow frames at once.
        hidden_states = torch.cat([hidden_states, flow_hidden_states], dim=2) # B C 2*Frames H W

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        
        if self.use_separate_rope_emb:
            # RoPE embeddings for frame-concat latents
            # ----------------------------------------
            # 1) Compute rope using ONLY video latent
            # ----------------------------------------
            rotary_emb_video = self.rope(flow_hidden_states) # Batch, Channels, Frames, H, W
            freqs_cos_video, freqs_sin_video = rotary_emb_video   # shapes: (1, N_video, 1, dim)
            
            # ----------------------------------------
            # 2) Repeat them for flow latent
            # ----------------------------------------
            freqs_cos_flow = freqs_cos_video.clone()   # identical positional map
            freqs_sin_flow = freqs_sin_video.clone()
            
            # ----------------------------------------
            # 3) Concat along token dimension
            # ----------------------------------------
            # video tokens + flow tokens
            freqs_cos_total = torch.cat([freqs_cos_video, freqs_cos_flow], dim=1)
            freqs_sin_total = torch.cat([freqs_sin_video, freqs_sin_flow], dim=1)

            # final rotary embedding passed to transformer blocks:
            rotary_emb = (freqs_cos_total, freqs_sin_total) # Batch, Channels, 2*Frames, H, W
        else:
            rotary_emb = self.rope(hidden_states) # Batch, Channels, 2*Frames, H, W
            
        
        # Patch embedding
        hidden_states = self.patch_embedding(hidden_states)  # B, C, 2*F, H/2, W/2
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        
        if self.use_separate_modality_emb:
            N_curr = hidden_states.shape[1]
            # Add modality embeddings
            video_tokens = hidden_states[:, :N_curr//2, :]
            flow_tokens  = hidden_states[:, N_curr//2:, :]
                        
            video_tokens = video_tokens + self.video_mod_emb
            flow_tokens  = flow_tokens  + self.flow_mod_emb
            
            hidden_states = torch.cat([video_tokens, flow_tokens], dim=1)
            

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        
        
        # TODO.Divide the frames for video and flow outputs
        
        # Divide the hidden_states back to video and flow
        # Video output projection
        # Use proj_out for video
        # Use proj_out_flow for flow
        # Divide tokens
        N_total = hidden_states.shape[1]
        video_tokens = hidden_states[:, :N_total//2, :]
        flow_tokens  = hidden_states[:, N_total//2:, :]
        # Project separately
        video_hidden_states = self.proj_out(video_tokens)
        flow_hidden_states  = self.proj_out_flow(flow_tokens)
        
        # Reshape back to video and flow frames
        post_patch_num_frames = post_patch_num_frames // 2 # Divide frames for video and flow
        
        video_hidden_states = video_hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        video_hidden_states = video_hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = video_hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        
        flow_hidden_states = flow_hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        flow_hidden_states = flow_hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        flow_output = flow_hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        
        
        
        """
        hidden_states = self.proj_out(hidden_states)
        
        

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output_total = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        
        
        output = output_total[:, :, :num_frames//2, :, :]
        flow_output = output_total[:, :, num_frames//2:, :, :]
        """
        
        

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, flow_output)

        return Transformer2DModelOutput(sample=output), Transformer2DModelOutput(sample=flow_output)
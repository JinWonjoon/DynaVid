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

import pdb

logger = logging.get_logger(__name__)



class WanTransformer3DModel_VideoJAM(WanTransformer3DModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 1. 원래 WanTransformer3DModel 로드
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # 2. VideoJAM에 맞게 수정
        
        # 1) patch_embedding 수정
        original_patch_embedding = model.patch_embedding
        model.patch_embedding = nn.Conv3d(
            in_channels=original_patch_embedding.in_channels * 2,
            out_channels=original_patch_embedding.out_channels,
            kernel_size=(1, 2, 2),  # VideoJAM의 patch size
            stride=(1, 2, 2),
            padding=0,
        )
        # Copy weights for the original channels
        with torch.no_grad():
            model.patch_embedding.weight[:, :original_patch_embedding.in_channels, :, :, :] = original_patch_embedding.weight
            # Initialize the new channels (for flow) to zero
            model.patch_embedding.weight[:, original_patch_embedding.in_channels:, :, :, :].zero_()
            model.patch_embedding.bias = original_patch_embedding.bias
        
        # 2) proj_out 수정
        original_proj_out = model.proj_out
        model.proj_out_flow = nn.Linear(
            in_features=original_proj_out.in_features,
            out_features=original_proj_out.out_features,
            bias=True,
        ) # Rand init.


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

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)
        
        # Joint patch embedding
        hidden_states = torch.cat([hidden_states, flow_hidden_states], dim=1)  # concat along channel dim

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

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
        
        # Video output projection
        video_hidden_states = self.proj_out(hidden_states)

        video_hidden_states = video_hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        video_hidden_states = video_hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = video_hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        
        # Flow output projection
        flow_hidden_states = self.proj_out_flow(hidden_states)

        flow_hidden_states = flow_hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        flow_hidden_states = flow_hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        flow_output = flow_hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, flow_output)

        return Transformer2DModelOutput(sample=output), Transformer2DModelOutput(sample=flow_output)
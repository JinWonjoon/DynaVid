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



class WanTransformer3DModel_Camera(WanTransformer3DModel):
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 1. 원래 WanTransformer3DModel 로드
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # #########################################################
        # # Camera Embedding Ver.1
        # #########################################################
        # # 2. embedding 교체 (여기서 구조 변경 가능)
        # old_embedder = model.condition_embedder
        # inner_dim = model.config.num_attention_heads * model.config.attention_head_dim
        # time_proj_dim = inner_dim * 6

        # model.condition_embedder = WanTimeTextImageCameraEmbedding(
        #     dim=inner_dim,
        #     time_freq_dim=model.config.freq_dim,
        #     time_proj_dim=time_proj_dim,
        #     text_embed_dim=model.config.text_dim,
        #     image_embed_dim=model.config.image_dim,
        #     pos_embed_seq_len=model.config.pos_embed_seq_len,
        # )
        # missing, unexpected = model.condition_embedder.load_state_dict(old_embedder.state_dict(), strict=False)
        # if len(missing) == 0 or len(unexpected) == 0:
        #     print("Copy parameters of Condition Embedder")
        #     print(f"Missing: {missing}")
        #     print(f"Unexpected: {unexpected}")

        # # 3. time_proj 교체 및 초기화
        # old_time_proj = old_embedder.time_proj
        # new_time_proj = torch.nn.Linear(inner_dim * 2, time_proj_dim).to(old_time_proj.weight.device)

        # with torch.no_grad():
        #     new_time_proj.weight[:, :old_time_proj.in_features] = old_time_proj.weight
        #     new_time_proj.weight[:, old_time_proj.in_features:] = 0.0
        #     if old_time_proj.bias is not None:
        #         new_time_proj.bias.copy_(old_time_proj.bias)

        # model.condition_embedder.time_proj = new_time_proj
        
        #########################################################
        # Camera Embedding Ver.2 : Use plucker embedding
        #########################################################
        inner_dim = model.config.num_attention_heads * model.config.attention_head_dim
        model.camera_embedder = CameraEmbedding(in_channels=6, hidden_size=inner_dim)

        return model
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        plucker_embedding: torch.Tensor = None,
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
        
        assert plucker_embedding is not None, "This model should use camera parameters!"

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states) # BCTHW (T=31)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len,
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        
        # Camera embedding
        cam_proj = self.camera_embedder(plucker_embedding)
        cam_proj = cam_proj.flatten(2).transpose(1, 2)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                # Inject cam_proj
                hidden_states = hidden_states + cam_proj
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                # Inject cam_proj
                hidden_states = hidden_states + cam_proj
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


# class WanTimeTextImageCameraEmbedding(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         time_freq_dim: int,
#         time_proj_dim: int,
#         text_embed_dim: int,
#         image_embed_dim: Optional[int] = None,
#         pos_embed_seq_len: Optional[int] = None,
#     ):
#         super().__init__()

#         self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
#         self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
#         self.act_fn = nn.SiLU()
#         self.time_proj = nn.Linear(dim, time_proj_dim)
#         self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

#         self.image_embedder = None
#         if image_embed_dim is not None:
#             self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)
        
#         # Camera projection
#         # 1) Trial 1. Direct use of camera parameters
#         self.cam_embedder = TimestepEmbedding(in_channels=25, time_embed_dim=dim) # fix input channel of camera parameters [Intrinsics(9), Extrinsics(16)]
#         self.cam_proj = nn.Linear(dim, dim * 2)
        
#         # 2) TODO.Trial 2. Plucker embedding from camera parameters
        

#     def forward(
#         self,
#         timestep: torch.Tensor,
#         encoder_hidden_states: torch.Tensor,
#         encoder_hidden_states_image: Optional[torch.Tensor] = None,
#         timestep_seq_len: Optional[int] = None,
#         camparam: Optional[torch.Tensor] = None,
#     ):
#         assert camparam.shape[-1] == 25, "You should use 25 dimension for camera parameters!"
        
#         timestep = self.timesteps_proj(timestep)
#         if timestep_seq_len is not None:
#             timestep = timestep.unflatten(0, (-1, timestep_seq_len))

#         time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
#         if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
#             timestep = timestep.to(time_embedder_dtype)
#         temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        
#         # Inject camera parameter condition to timestep_proj
#         cam_emb = self.cam_embedder(camparam)
#         cam_proj = self.cam_proj(self.act_fn(cam_emb))
        
#         #################################
#         # Re-use time_proj for injecting camera condition to timestep_proj
#         # At first, camparam_proj does not impact timestep, as its corresponding weights are initialized by zero
#         # As training goes, camparam_proj does impact timestep smoothly.
#         #################################
#         timestep_proj = self.time_proj(self.act_fn(temb))
        
#         # timestep_proj = self.time_proj(torch.cat([self.act_fn(temb), self.act_fn(cam_emb)], dim=-1))
#         # timestep_proj = self.time_proj(torch.cat([self.act_fn(temb), self.act_fn(temb)], dim=-1))

#         encoder_hidden_states = self.text_embedder(encoder_hidden_states)
#         if encoder_hidden_states_image is not None:
#             encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)
        

#         return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image




class WanTransformerBlock_Camera(WanTransformerBlock):
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        cam_proj: torch.Tensor,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states
    


"""
Below is too much naive embedder.
No normalization and residual.
"""

# class CameraEmbedding(nn.Module):
#     def __init__(self, in_channels=6, hidden_size=3072):
#         super().__init__()
#         # temporal downsample: 121 -> 61 -> 31
#         self.conv_t1 = nn.Conv3d(in_channels, 256, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
#         self.conv_t2 = nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
#         self.proj = nn.Conv3d(512, hidden_size, kernel_size=(1,2,2), stride=(1,2,2))
        
#         # Zero-init.
#         nn.init.zeros_(self.proj.weight)
#         if self.proj.bias is not None:
#             nn.init.zeros_(self.proj.bias)

#     def forward(self, plucker):
#         """
#         Args:
#             plucker: (B,6,T_in,H_latent,W_latent), e.g. (B,6,121,44,80)
#         Returns:
#             cam_emb: (B,hidden_size,T_out,H_latent,W_latent), e.g. (B,3072,31,22,40)
#         """
#         x = F.silu(self.conv_t1(plucker))  # (B,128,61,H,W)
#         x = F.silu(self.conv_t2(x))        # (B,256,31,H,W)
#         x = self.proj(x)                   # (B,3072,31,H/2,W/2)
#         return x


class CameraEmbedding(nn.Module):
    def __init__(self, in_channels=6, hidden_size=3072):
        super().__init__()
        # temporal downsample: 121 -> 61 -> 31
        self.ln1 = nn.LayerNorm(in_channels)
        self.conv_t1 = nn.Conv3d(in_channels, 256, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))

        self.ln2 = nn.LayerNorm(256)
        self.conv_t2 = nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))

        self.proj = nn.Conv3d(512, hidden_size, kernel_size=(1,2,2), stride=(1,2,2))

        # residual projection (채널 수 안 맞을 때만 사용)
        self.residual_proj1 = nn.Conv3d(in_channels, 256, kernel_size=1, stride=(2,1,1))
        self.residual_proj2 = nn.Conv3d(256, 512, kernel_size=1, stride=(2,1,1))

        # Zero-init for stable residual learning
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, plucker):
        """
        Args:
            plucker: (B,6,T_in,H,W)
        Returns:
            cam_emb: (B,hidden_size,T_out,H/2,W/2)
        """

        # block 1: pre-norm
        residual = self.residual_proj1(plucker)
        x = plucker.permute(0,2,3,4,1)   # (B,T,H,W,C)
        x = self.ln1(x)
        x = F.silu(x)
        x = x.permute(0,4,1,2,3)         # back to (B,C,T,H,W)
        x = self.conv_t1(x)
        x = x + residual

        # block 2: pre-norm
        residual = self.residual_proj2(x)
        x = x.permute(0,2,3,4,1)
        x = self.ln2(x)
        x = F.silu(x)
        x = x.permute(0,4,1,2,3)
        x = self.conv_t2(x)
        x = x + residual

        # projection
        x = self.proj(x)   # (B,hidden_size, T_out, H/2, W/2)
        return x

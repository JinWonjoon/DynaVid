import math
import copy
from typing import Optional, Tuple, Dict, Any, Union

import torch
from torch import nn

from diffusers import WanTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from diffusers.models.transformers.transformer_wan import WanImageEmbedding, WanAttention, WanAttnProcessor, WanTransformerBlock
from diffusers.models.attention import FeedForward, AttentionMixin

from training.modules.multibranchgen_patch import _upgrade_block_for_multibranch

logger = logging.get_logger(__name__)


JOINT_ATTN_NONE = "none"
JOINT_ATTN_CROSS = "cross"
JOINT_ATTN_SELF = "self"

class WanTransformer3DModel_MultiBranchGen(WanTransformer3DModel):
    """
    WanTransformer3DModel를 상속한 Two branch based multi-modal 버전.
    __init__는 건드리지 않고, from_pretrained 에서 block/프로젝션을 업그레이드한다.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        joint_attn_type: str = JOINT_ATTN_CROSS,  # "cross", "self", "none"
        *model_args,
        **kwargs,
    ):
        # args 객체가 들어오는 경우 (args.pretrained_model_name_or_path)
        if not isinstance(pretrained_model_name_or_path, str) and hasattr(
            pretrained_model_name_or_path, "pretrained_model_name_or_path"
        ):
            pretrained_path = pretrained_model_name_or_path.pretrained_model_name_or_path
        else:
            pretrained_path = pretrained_model_name_or_path

        # 1) 원래 WanTransformer3DModel 로드
        model = super().from_pretrained(
            pretrained_path, *model_args, **kwargs
        )

        # 2) config / dim 정보
        inner_dim = model.config.num_attention_heads * model.config.attention_head_dim
        num_heads = model.config.num_attention_heads
        eps = getattr(model.config, "eps", 1e-6)
        added_kv_proj_dim = getattr(model.config, "added_kv_proj_dim", None)

        # 3) 각 block 업그레이드
        for block in model.blocks:
            _upgrade_block_for_multibranch(
                block,
                dim=inner_dim,
                num_heads=num_heads,
                eps=eps,
                added_kv_proj_dim=added_kv_proj_dim,
                joint_attn_type=joint_attn_type,
            )

        # 4) proj_out_flow 추가 (video proj 복사)
        model.proj_out_flow = copy.deepcopy(model.proj_out)

        # 5) joint config 저장
        model.joint_attn_type = joint_attn_type

        return model

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, C, T, H, W) - video
        flow_hidden_states: torch.Tensor,     # (B, C, T, H, W) - flow
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )


        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        assert flow_hidden_states.shape == hidden_states.shape, "video/flow input shape mismatch"

        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # RoPE (video 기준, flow도 같은 위치 가정)
        rotary_emb = self.rope(hidden_states)

        # Patch embedding (공유)
        hidden_states = self.patch_embedding(hidden_states)          # (B, C', T', H', W')
        flow_hidden_states = self.patch_embedding(flow_hidden_states)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)         # (B, N, C')
        flow_hidden_states = flow_hidden_states.flatten(2).transpose(1, 2)

        # timestep & cond embed
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep_flat = timestep.flatten()
        else:
            ts_seq_len = None
            timestep_flat = timestep

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep_flat, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )

        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))  # (B, N, 6, C')
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))  # (B, 6, C')

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states, flow_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    flow_hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
        else:
            for block in self.blocks:
                hidden_states, flow_hidden_states = block(
                    hidden_states,
                    flow_hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )

        # Output norm & projection & unpatchify (video / flow)
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        # video
        hs_v = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hs_v = self.proj_out(hs_v)
        hs_v = hs_v.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hs_v = hs_v.permute(0, 7, 1, 4, 2, 5, 3, 6)
        video_out = hs_v.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        # flow
        hs_f = (self.norm_out(flow_hidden_states.float()) * (1 + scale) + shift).type_as(flow_hidden_states)
        hs_f = self.proj_out_flow(hs_f)
        hs_f = hs_f.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hs_f = hs_f.permute(0, 7, 1, 4, 2, 5, 3, 6)
        flow_out = hs_f.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return video_out, flow_out

        return Transformer2DModelOutput(sample=video_out), Transformer2DModelOutput(sample=flow_out)
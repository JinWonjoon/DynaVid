import copy
import types
import math
from typing import Optional, Dict, Any, Union, Tuple

import torch
from torch import nn

from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

JOINT_ATTN_NONE = "none"
JOINT_ATTN_CROSS = "cross"
JOINT_ATTN_SELF = "self"


def _upgrade_block_for_multibranch(
    block,
    dim: int,
    num_heads: int,
    eps: float,
    added_kv_proj_dim: Optional[int],
    joint_attn_type: str = JOINT_ATTN_CROSS,
):
    """
    기존 WanTransformerBlock 인스턴스를 in-place로 multi-branch 용도로 업그레이드:
      - *_flow 모듈 추가
      - joint cross/self attention 모듈 추가
      - zero-init joint_linear 추가
      - forward_multibranch 메서드 패치
    """
    
    dim_head = dim // num_heads

    # 1) flow용 모듈 복사
    block.norm1_flow = copy.deepcopy(block.norm1)
    block.attn1_flow = copy.deepcopy(block.attn1)

    block.attn2_flow = copy.deepcopy(block.attn2)
    block.norm2_flow = copy.deepcopy(block.norm2)

    block.ffn_flow = copy.deepcopy(block.ffn)
    block.norm3_flow = copy.deepcopy(block.norm3)

    # 2) joint attention 모듈

    if joint_attn_type == JOINT_ATTN_CROSS:
        # joint cross-attn: video_q ← video, kv ← flow / flow_q ← flow, kv ← video
        # block.joint_cross_attn_flow_to_video = WanAttention(
        #     dim=dim,
        #     heads=num_heads,
        #     dim_head=dim_head,
        #     eps=eps,
        #     cross_attention_dim_head=dim_head,
        #     added_kv_proj_dim=added_kv_proj_dim,
        #     processor=WanAttnProcessor(),
        # )
        # block.joint_cross_attn_video_to_flow = WanAttention(
        #     dim=dim,
        #     heads=num_heads,
        #     dim_head=dim_head,
        #     eps=eps,
        #     cross_attention_dim_head=dim_head,
        #     added_kv_proj_dim=added_kv_proj_dim,
        #     processor=WanAttnProcessor(),
        # )
        
        # with torch.no_grad():
        #     block.joint_cross_attn_flow_to_video.load_state_dict(block.attn1.state_dict())
        #     block.joint_cross_attn_video_to_flow.load_state_dict(block.attn1.state_dict())
        
        # Due to OOM, use single joint cross-attn with concatenated tokens
        block.joint_cross_attn_video_flow = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            eps=eps,
            cross_attention_dim_head=dim_head,
            added_kv_proj_dim=added_kv_proj_dim,
            processor=WanAttnProcessor(),
        )

        with torch.no_grad():
            # 기존 self-attn(attn1)으로부터 weight init
            block.joint_cross_attn_video_flow.load_state_dict(block.attn1.state_dict())


    elif joint_attn_type == JOINT_ATTN_SELF:
        # joint self-attn: concat(video, flow) → self-attn → split
        block.joint_self_attn_video_flow = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            eps=eps,
            cross_attention_dim_head=None,
            added_kv_proj_dim=None,
            processor=WanAttnProcessor(),
        )
        
        with torch.no_grad():
            block.joint_self_attn_video_flow.load_state_dict(block.attn1.state_dict())

    else:
        pass

    # 3) zero-init linear (channel-wise mixing)
    block.joint_linear_video_flow = nn.Linear(dim * 2, dim * 2)
    nn.init.zeros_(block.joint_linear_video_flow.weight)
    nn.init.zeros_(block.joint_linear_video_flow.bias)

    # 기본 joint type
    block.joint_attn_type = joint_attn_type
    
    # 4) modality embedding (per block, per modality)
    block.joint_video_mod_emb = nn.Parameter(torch.zeros(1, 1, dim))
    block.joint_flow_mod_emb  = nn.Parameter(torch.zeros(1, 1, dim))
    nn.init.normal_(block.joint_video_mod_emb, std=0.02)
    nn.init.normal_(block.joint_flow_mod_emb,  std=0.02)

    # 6) temb helper + forward_multibranch 패치

    def _prepare_temb(self, temb: torch.Tensor, hidden_len: int):
        if temb.ndim == 4:
            # (B, N, 6, C)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # (B, 6, C)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        def _expand_to_match(x):
            if x.ndim == 3:
                return x
            elif x.ndim == 2:
                return x.unsqueeze(1).expand(-1, hidden_len, -1)
            return x

        shift_msa = _expand_to_match(shift_msa)
        scale_msa = _expand_to_match(scale_msa)
        gate_msa = _expand_to_match(gate_msa)
        c_shift_msa = _expand_to_match(c_shift_msa)
        c_scale_msa = _expand_to_match(c_scale_msa)
        c_gate_msa = _expand_to_match(c_gate_msa)

        return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    block._prepare_temb = types.MethodType(_prepare_temb, block)

    def forward_multibranch(
        self,
        hidden_states_video: torch.Tensor,
        flow_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attn_type = self.joint_attn_type

        B, N_v, C = hidden_states_video.shape
        B_f, N_f, C_f = flow_hidden_states.shape
        assert B == B_f and C == C_f
        assert N_v == N_f

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self._prepare_temb(temb, N_v)

        # pre-norm
        norm_v = (self.norm1(hidden_states_video.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states_video)
        norm_f = (self.norm1_flow(flow_hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
            flow_hidden_states
        )

        # joint attention
        if joint_attn_type == JOINT_ATTN_CROSS:
            # ref_v = self.joint_cross_attn_flow_to_video(norm_v, norm_f, None, rotary_emb)
            # ref_f = self.joint_cross_attn_video_to_flow(norm_f, norm_v, None, rotary_emb)
            
            # Q=video, K/V=flow
            ref_v = self.joint_cross_attn_video_flow(norm_v, norm_f, None, rotary_emb)
            # Q=flow, K/V=video
            ref_f = self.joint_cross_attn_video_flow(norm_f, norm_v, None, rotary_emb)
            
        elif joint_attn_type == JOINT_ATTN_SELF:
            
            # (1) modality embedding 
            #    norm_v, norm_f: (B, N, C)
            #    video_mod_emb, flow_mod_emb: (1, 1, C) → (B, N, C)
            v_with_mod = norm_v + self.joint_video_mod_emb
            f_with_mod = norm_f + self.joint_flow_mod_emb
            
            # (2) joint token
            joint_tokens = torch.cat([v_with_mod, f_with_mod], dim=1)  # (B, 2N, C)
            
            if rotary_emb is not None:
                if isinstance(rotary_emb, tuple):
                    freqs_cos, freqs_sin = rotary_emb   # 각: (1, N_v, 1, rope_dim)
                    freqs_cos_joint = torch.cat([freqs_cos, freqs_cos], dim=1)  # (1, 2N_v, 1, D)
                    freqs_sin_joint = torch.cat([freqs_sin, freqs_sin], dim=1)  # (1, 2N_v, 1, D)
                    rotary_joint = (freqs_cos_joint, freqs_sin_joint)
                else:
                    # fallback for tensor input
                    rotary_joint = torch.cat([rotary_emb, rotary_emb], dim=1)
            else:
                rotary_joint = None
                
            joint_out = self.joint_self_attn_video_flow(joint_tokens, None, None, rotary_joint)
            ref_v = joint_out[:, :N_v, :]
            ref_f = joint_out[:, N_v:, :]
        else:
            ref_v = torch.zeros_like(norm_v)
            ref_f = torch.zeros_like(norm_f)

        joint_cat = torch.cat([ref_v, ref_f], dim=-1)  # (B, N, 2C)
        mixed = self.joint_linear_video_flow(joint_cat)           # zero-init → 초기에 0
        delta_v, delta_f = mixed.chunk(2, dim=-1)

        # self-attn + joint residual
        attn_v = self.attn1(norm_v, None, None, rotary_emb) + delta_v
        hidden_states_video = (hidden_states_video.float() + attn_v * gate_msa).type_as(hidden_states_video)

        attn_f = self.attn1_flow(norm_f, None, None, rotary_emb) + delta_f
        flow_hidden_states = (flow_hidden_states.float() + attn_f * gate_msa).type_as(flow_hidden_states)

        # cross-attn (encoder)
        norm_v2 = self.norm2(hidden_states_video.float()).type_as(hidden_states_video)
        attn_v2 = self.attn2(norm_v2, encoder_hidden_states, None, None)
        hidden_states_video = hidden_states_video + attn_v2

        norm_f2 = self.norm2_flow(flow_hidden_states.float()).type_as(flow_hidden_states)
        attn_f2 = self.attn2_flow(norm_f2, encoder_hidden_states, None, None)
        flow_hidden_states = flow_hidden_states + attn_f2

        # FFN
        norm_v3 = (self.norm3(hidden_states_video.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states_video
        )
        ff_v = self.ffn(norm_v3)
        hidden_states_video = (hidden_states_video.float() + ff_v.float() * c_gate_msa).type_as(hidden_states_video)

        norm_f3 = (self.norm3_flow(flow_hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            flow_hidden_states
        )
        ff_f = self.ffn_flow(norm_f3)
        flow_hidden_states = (flow_hidden_states.float() + ff_f.float() * c_gate_msa).type_as(flow_hidden_states)

        return hidden_states_video, flow_hidden_states

    block.forward = types.MethodType(forward_multibranch, block)
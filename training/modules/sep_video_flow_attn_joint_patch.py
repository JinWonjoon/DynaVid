import copy
import types
from typing import Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

import pdb

def _split_rotary_emb(rotary_emb, n_video_tokens: int):
    """
    rotary_emb: (freqs_cos, freqs_sin)
      - freqs_*: (1, N_total, 1, D)
    """
    if rotary_emb is None:
        return None, None
    freqs_cos, freqs_sin = rotary_emb
    re_video = (freqs_cos[:, :n_video_tokens, ...], freqs_sin[:, :n_video_tokens, ...])
    re_flow  = (freqs_cos[:, n_video_tokens:, ...], freqs_sin[:, n_video_tokens:, ...])
    return re_video, re_flow


# def _apply_rope_to_qk(
#     q: torch.Tensor,  # (B*H, N_total or 2N_total, D)
#     k: torch.Tensor,  # same shape as q
#     rotary_emb,       # (freqs_cos, freqs_sin), each (1, N, 1, D)
# ):
#     """
#     WanAttnProcessor의 apply_rotary_emb 로직을 그대로 q,k에 적용하는 버전.
#     q, k의 마지막 dim과 rotary_emb[-1] dim이 Wan과 동일하다는 전제.
#     """

#     if rotary_emb is None:
#         return q, k

#     freqs_cos, freqs_sin = rotary_emb  # (1, N, 1, D_rope)

#     def apply_rotary(hidden_states: torch.Tensor):
#         # hidden_states: (B*H, N, D)
#         # Wan 코드와 동일한 방식:
#         # 1) 마지막 dim을 2로 unflatten -> (..., d, 2)
#         # 2) cos,sin을 even/odd index로 사용
#         x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)  # (..., d, 2) -> x1,x2: (..., d)

#         # freqs_cos, freqs_sin: (1, N, 1, D_rope)
#         # Wan 코드: cos = freqs_cos[..., 0::2], sin = freqs_sin[..., 1::2]
#         cos = freqs_cos[..., 0::2]  # (1, N, 1, D_rope/2)
#         sin = freqs_sin[..., 1::2]  # (1, N, 1, D_rope/2)

#         # x1: (B*H, N, d)
#         # cos: (1, N, 1, d) 꼴이므로 브로드캐스트 위해 squeeze/unsqueeze
#         cos = cos.squeeze(-2)  # (1, N, d)
#         sin = sin.squeeze(-2)  # (1, N, d)

#         # x1.ndim == 3 (B*H, N, d) 인 경우를 기준으로 broadcast
#         # 혹시 배치 차원이 더 있으면 while 루프로 맞춰줌
#         _cos = cos
#         _sin = sin
#         while _cos.ndim < x1.ndim:
#             _cos = _cos.unsqueeze(0)
#             _sin = _sin.unsqueeze(0)

#         _cos = _cos.expand_as(x1)
#         _sin = _sin.expand_as(x1)

#         out = torch.empty_like(hidden_states)
#         out[..., 0::2] = x1 * _cos - x2 * _sin
#         out[..., 1::2] = x1 * _sin + x2 * _cos
#         return out.type_as(hidden_states)

#     q = apply_rotary(q)
#     k = apply_rotary(k)
#     return q, k


def _joint_self_attn_two_branch_with_norm_rope(
    attn_video,                     # block.attn1 (WanAttention)
    attn_flow,                      # block.attn1_flow (WanAttention)
    hidden_video: torch.Tensor,     # (B, N, C)
    hidden_flow: torch.Tensor,      # (B, N, C)
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (freqs_cos, freqs_sin)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1) video / flow 각각 WanAttention 스타일로 projection + norm_q/norm_k
    2) head 차원까지 나눈 뒤, 시퀀스 방향으로 concat
    3) RoPE 적용 (WanAttnProcessor와 동일한 방식)
    4) SDPA 실행
    5) 다시 시퀀스 쪼개서 각자 to_out 통과
    """

    B, N_v, C = hidden_video.shape
    B_f, N_f, C_f = hidden_flow.shape
    assert B == B_f and C == C_f, "video/flow shape mismatch"
    assert N_v == N_f, "현재 설계는 N_video == N_flow (frame-wise concat)을 가정"

    L_total = N_v + N_f

    # 공통 head 설정 (두 attn은 복사본이므로 동일해야 함)
    H = attn_video.heads
    Dh = attn_video.inner_dim // H

    # ------------------------------
    # 1) q, k, v projection (각자 모듈)
    # ------------------------------
    q_v = attn_video.to_q(hidden_video)   # (B, N, H*Dh)
    k_v = attn_video.to_k(hidden_video)
    v_v = attn_video.to_v(hidden_video)

    q_f = attn_flow.to_q(hidden_flow)
    k_f = attn_flow.to_k(hidden_flow)
    v_f = attn_flow.to_v(hidden_flow)

    # ------------------------------
    # 2) norm_q / norm_k (각자 모듈)
    # ------------------------------
    q_v = attn_video.norm_q(q_v)
    k_v = attn_video.norm_k(k_v)

    q_f = attn_flow.norm_q(q_f)
    k_f = attn_flow.norm_k(k_f)

    # ------------------------------
    # 3) head 차원으로 reshape + 시퀀스 concat
    #    WanAttnProcessor는 (B, L, H, Dh) → 여기서는 SDPA 편의를 위해 (B, H, L, Dh)
    # ------------------------------
    q_v = q_v.view(B, N_v, H, Dh).permute(0, 2, 1, 3)  # (B, H, N_v, Dh)
    k_v = k_v.view(B, N_v, H, Dh).permute(0, 2, 1, 3)
    v_v = v_v.view(B, N_v, H, Dh).permute(0, 2, 1, 3)

    q_f = q_f.view(B, N_f, H, Dh).permute(0, 2, 1, 3)  # (B, H, N_f, Dh)
    k_f = k_f.view(B, N_f, H, Dh).permute(0, 2, 1, 3)
    v_f = v_f.view(B, N_f, H, Dh).permute(0, 2, 1, 3)

    # 시퀀스 방향 concat: [video tokens, flow tokens]
    q_cat = torch.cat([q_v, q_f], dim=2)   # (B, H, L_total, Dh)
    k_cat = torch.cat([k_v, k_f], dim=2)
    v_cat = torch.cat([v_v, v_f], dim=2)

    # ------------------------------
    # 4) RoPE 적용 (WanAttnProcessor와 동일한 방식)
    #    rotary_emb: (freqs_cos, freqs_sin), each (1, L_total, 1, rope_dim)
    # ------------------------------
    if rotary_emb is not None:
        freqs_cos, freqs_sin = rotary_emb  # (1, L_total, 1, D_rope)

        def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
            # x: (B, H, L, Dh) → (B, L, H, Dh)로 변환 후 Wan 스타일 적용
            x = x.permute(0, 2, 1, 3)  # (B, L, H, Dh)

            # 마지막 dim을 (-1,2)로 쪼개서 even/odd 채널 묶기
            x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)  # 각 (B, L, H, Dh/2)

            cos = freqs_cos[..., 0::2]  # (1, L, 1, Dh/2)
            sin = freqs_sin[..., 1::2]  # (1, L, 1, Dh/2)

            cos = cos.to(x1.device).to(x1.dtype)
            sin = sin.to(x1.device).to(x1.dtype)

            out = torch.empty_like(x)
            out[..., 0::2] = x1 * cos - x2 * sin
            out[..., 1::2] = x1 * sin + x2 * cos

            return out.permute(0, 2, 1, 3).type_as(x)  # 다시 (B, H, L, Dh)로

        q_cat = apply_rotary_emb(q_cat, freqs_cos, freqs_sin)
        k_cat = apply_rotary_emb(k_cat, freqs_cos, freqs_sin)

    # ------------------------------
    # 5) SDPA (PyTorch 2.0)
    #    scale은 F.scaled_dot_product_attention 안에서 자동 처리
    # ------------------------------
    attn_out = F.scaled_dot_product_attention(
        q_cat,
        k_cat,
        v_cat,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )  # (B, H, L_total, Dh)

    # ------------------------------
    # 6) head merge + 시퀀스 split + to_out (각자 모듈)
    # ------------------------------
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L_total, H * Dh)  # (B, L_total, C)

    out_v = attn_out[:, :N_v, :]
    out_f = attn_out[:, N_v:, :]

    # 각 branch의 to_out 사용
    out_v = attn_video.to_out[0](out_v)
    out_v = attn_video.to_out[1](out_v)

    out_f = attn_flow.to_out[0](out_f)
    out_f = attn_flow.to_out[1](out_f)

    return out_v, out_f   # (B, N, C), (B, N, C)



def upgrade_wan_blocks_for_sep_video_flow_attn_joint(model):
    """
    기존 WanTransformerBlock을 in-place로 패치:
      - 모든 모듈을 video / flow로 분리 (가중치는 video 기준으로 deepcopy)
        - norm1_flow, attn1_flow, norm2_flow, attn2_flow, norm3_flow, ffn_flow
      - Self-Attn은 다음과 같이 동작:
        1) video, flow 각각에 대해 별도 proj(to_q/to_k/to_v) 수행
        2) q/k/v를 frame-wise concat 후 한 번의 multi-head attention 수행
        3) out을 다시 video / flow로 split
        4) split된 out에 대해 각각 별도의 to_out / to_out_flow 적용
      - Cross-Attn, FFN, LayerNorm 등은 브랜치별로 완전히 분리된 모듈 사용
      - block.forward를 새 구현으로 교체
    """

    for block in model.blocks:
        # 이미 패치된 블록이면 건너뛰기 (원하면 조건 추가 가능)
        if hasattr(block, "_forward_orig_joint_sep"):
            continue

        # 원래 forward 백업
        block._forward_orig_joint_sep = block.forward

        # ============================================
        # 1) flow용 모듈 복사 (모든 모듈 분리)
        # ============================================
        # LayerNorm
        block.norm1_flow = copy.deepcopy(block.norm1)
        block.norm2_flow = copy.deepcopy(block.norm2)
        block.norm3_flow = copy.deepcopy(block.norm3)

        # Self-Attn
        block.attn1_flow = copy.deepcopy(block.attn1)

        # Cross-Attn
        block.attn2_flow = copy.deepcopy(block.attn2)

        # FFN
        block.ffn_flow = copy.deepcopy(block.ffn)

        # 편의를 위해 attn 설정 정보 캐시
        attn = block.attn1
        num_heads = attn.heads
        dim_head = attn.inner_dim // attn.heads
        scale = dim_head ** -0.5

        block._joint_num_heads = num_heads
        block._joint_dim_head = dim_head
        block._joint_scale = scale

        # ============================================
        # 2) temb helper (Wan 스타일 그대로)
        # ============================================
        def _prepare_temb(self, temb: torch.Tensor, hidden_len: int):
            if temb.ndim == 4:
                # (B, N, 6, C)
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                    self.scale_shift_table.unsqueeze(0) + temb.float()
                ).chunk(6, dim=2)
                shift_msa = shift_msa.squeeze(2)
                scale_msa = scale_msa.squeeze(2)
                gate_msa  = gate_msa.squeeze(2)
                c_shift_msa = c_shift_msa.squeeze(2)
                c_scale_msa = c_scale_msa.squeeze(2)
                c_gate_msa  = c_gate_msa.squeeze(2)
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
            gate_msa  = _expand_to_match(gate_msa)
            c_shift_msa = _expand_to_match(c_shift_msa)
            c_scale_msa = _expand_to_match(c_scale_msa)
            c_gate_msa  = _expand_to_match(c_gate_msa)

            return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

        block._prepare_temb_joint = types.MethodType(_prepare_temb, block)

        # ============================================
        # 3) joint self-attention + 분리된 브랜치 forward
        # ============================================
        def _forward_joint_sep(
            self,
            hidden_states: torch.Tensor,      # (B, N_total, C)  -> frame-concat (video + flow)
            encoder_hidden_states: torch.Tensor,
            temb: torch.Tensor,
            rotary_emb,                       # (freqs_cos, freqs_sin) on concat tokens
        ) -> torch.Tensor:

            B, N_total, C = hidden_states.shape
            assert N_total % 2 == 0, f"Expected even #tokens (video+flow), got {N_total}"
            N = N_total // 2

            video = hidden_states[:, :N, :]  # (B, N, C)
            flow  = hidden_states[:, N:, :]  # (B, N, C)

            # temb에서 shift/scale/gate 추출 (길이는 video/flow에 동일하게 사용)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
                self._prepare_temb_joint(temb, N)

            # ------------------------------------------------
            # 1) LayerNorm + shift/scale (video / flow 분리)
            # ------------------------------------------------
            v_norm = (self.norm1(video.float()) * (1 + scale_msa) + shift_msa).type_as(video)
            f_norm = (self.norm1_flow(flow.float()) * (1 + scale_msa) + shift_msa).type_as(flow)

            
            # ------------------------------------------------
            # 2) joint self-attn:
            #    - video: attn1
            #    - flow : attn1_flow
            #    - q/k/v proj는 서로 다른 모듈 사용
            #    - q/k/v concat 후 한 번의 attention, 다시 split
            # ------------------------------------------------

            v_attn, f_attn = _joint_self_attn_two_branch_with_norm_rope(
                self.attn1,          # video용 WanAttention
                self.attn1_flow,     # flow용 WanAttention
                v_norm,
                f_norm,
                rotary_emb,          # (freqs_cos, freqs_sin) with length 2N
            )
            

            """
            attn_v = self.attn1
            attn_f = self.attn1_flow
            
            H = self._joint_num_heads
            Dh = self._joint_dim_head
            scale = self._joint_scale

            # (a) proj Q/K/V
            #    to_q/to_k/to_v: (B, N, C) -> (B, N, H*Dh)
            q_v = attn_v.to_q(v_norm)
            k_v = attn_v.to_k(v_norm)
            v_v = attn_v.to_v(v_norm)

            q_f = attn_f.to_q(f_norm)
            k_f = attn_f.to_k(f_norm)
            v_f = attn_f.to_v(f_norm)

            # (b) reshape to (B, H, N, Dh)
            def _reshape_qkv(x):
                B_, N_, C_ = x.shape
                x = x.view(B_, N_, H, Dh).permute(0, 2, 1, 3)
                return x  # (B, H, N, Dh)

            q_v = _reshape_qkv(q_v)
            k_v = _reshape_qkv(k_v)
            v_v = _reshape_qkv(v_v)

            q_f = _reshape_qkv(q_f)
            k_f = _reshape_qkv(k_f)
            v_f = _reshape_qkv(v_f)
            
            
            
            
            # (c) concat along token dimension: video + flow
            q_cat = torch.cat([q_v, q_f], dim=2)  # (B, H, 2N, Dh)
            k_cat = torch.cat([k_v, k_f], dim=2)
            v_cat = torch.cat([v_v, v_f], dim=2)

            # (d) RoPE 적용 (frame-concat 전체 토큰에 대해)
            if rotary_emb is not None:
                # rotary_emb: (freqs_cos, freqs_sin) for 2N tokens
                q_cat, k_cat = _apply_rope_to_qk(q_cat, k_cat, rotary_emb)

            # (e) scaled dot-product attention
            #     attn_scores: (B, H, 2N, 2N)
            # attn_scores = torch.matmul(q_cat, k_cat.transpose(-1, -2)) * scale
            # attn_probs = torch.softmax(attn_scores, dim=-1)
            # out_cat = torch.matmul(attn_probs, v_cat)  # (B, H, 2N, Dh)
            
            out_cat = F.scaled_dot_product_attention(
                q_cat,  # (B, H, 2N, Dh)
                k_cat,
                v_cat,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            

            # (f) split back to video / flow
            out_v = out_cat[:, :, :N, :]   # (B, H, N, Dh)
            out_f = out_cat[:, :, N:, :]

            # (g) merge heads -> (B, N, C)
            def _merge_heads(x):
                B_, H_, N_, Dh_ = x.shape
                x = x.permute(0, 2, 1, 3).contiguous().view(B_, N_, H_ * Dh_)
                return x

            out_v = _merge_heads(out_v)   # (B, N, C)
            out_f = _merge_heads(out_f)

            # (h) to_out (video, flow 각각 다른 모듈 사용)
            out_v = attn_v.to_out[1](attn_v.to_out[0](out_v))  # (B, N, C)
            out_f = attn_f.to_out[1](attn_f.to_out[0](out_f))
            
            """
            

            # (i) residual + gate
            video = (video.float() + v_attn.float() * gate_msa.float()).type_as(video)
            flow  = (flow.float()  + f_attn.float() * gate_msa.float()).type_as(flow)

            # ------------------------------------------------
            # 3) Cross-attn (text/image) - 완전 분리된 브랜치
            # ------------------------------------------------
            # video branch
            v2 = self.norm2(video.float()).type_as(video)
            attn_v2 = self.attn2(v2, encoder_hidden_states, None, None)
            video = video + attn_v2

            # flow branch
            f2 = self.norm2_flow(flow.float()).type_as(flow)
            attn_f2 = self.attn2_flow(f2, encoder_hidden_states, None, None)
            flow = flow + attn_f2

            # ------------------------------------------------
            # 4) FFN (LayerNorm + FFN 모두 분리)
            # ------------------------------------------------
            v3 = (self.norm3(video.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(video)
            f3 = (self.norm3_flow(flow.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(flow)

            v_ff = self.ffn(v3)
            f_ff = self.ffn_flow(f3)

            video = (video.float() + v_ff.float() * c_gate_msa.float()).type_as(video)
            flow  = (flow.float()  + f_ff.float() * c_gate_msa.float()).type_as(flow)

            # 다시 concat해서 반환 (frame-concat transformer와 호환)
            return torch.cat([video, flow], dim=1)

        # 메서드로 바인딩
        block.forward = types.MethodType(_forward_joint_sep, block)
        
        



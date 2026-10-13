import types
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

def _split_rotary_emb(rotary_emb, n_video):
    # rotary_emb: (freqs_cos, freqs_sin), each (1, N_total, 1, rope_dim)
    if rotary_emb is None:
        return None, None
    freqs_cos, freqs_sin = rotary_emb
    re_video = (freqs_cos[:, :n_video], freqs_sin[:, :n_video])
    re_flow  = (freqs_cos[:, n_video:], freqs_sin[:, n_video:])
    return re_video, re_flow

def _compute_mm_gate(block, temb):
    """
    Returns:
      gate_f2v, gate_v2f with shape broadcastable to (B,N,C)
      - if gated: (B,1,C)
      - else: float 1.0
    """
    if not getattr(block, "mm_use_gate", False):
        return 1.0, 1.0

    # temb: (B,6,C) typical for wan2.1/2.2
    if temb.ndim == 3:
        temb_gate = temb[:, 2, :]  # (B,C) gate channel
    else:
        # fallback; keep it simple
        temb_gate = temb

    # global timestep switch (B,1,1)
    temb_switch = torch.sigmoid(temb_gate.mean(dim=-1, keepdim=True)).unsqueeze(-1)  # (B,1,1)

    base_gate_f2v = block.mm_gate_f2v_table  # (1,1,C)
    base_gate_v2f = block.mm_gate_v2f_table  # (1,1,C)

    gate_f2v = base_gate_f2v * temb_switch   # (B,1,C) via broadcast
    gate_v2f = base_gate_v2f * temb_switch

    return gate_f2v, gate_v2f

def _mm_mix_and_split(block, v2, f2):
    """
    v2, f2: (B,N,C)
    concat on channel -> (B,N,2C)
    proj (zero-init) -> (B,N,2C)
    split -> dv, df each (B,N,C)
    """
    B, N, C = v2.shape
    z = torch.cat([v2, f2], dim=-1)  # (B,N,2C)
    z = block.mm_mix_proj(z)         # (B,N,2C)
    dv, df = z[..., :C], z[..., C:]
    return dv, df

#-------------------------------------------------------------------------
"""
MMCA #1
- SA
- CA (video<->flow)
- CA (text)
- FFN
"""
def _wan_block_forward_mm(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
    """
    hidden_states: (B, 2N, C)  where first N=video tokens, second N=flow tokens
    encoder_hidden_states: (B, S, D_text)
    temb: (B, 6, C)  (wan2.1/2.2)  [ti2v 케이스는 별도 핸들링 필요]
    rotary_emb: (freqs_cos, freqs_sin) for total tokens (2N)
    """
    B, N2, C = hidden_states.shape
    assert N2 % 2 == 0, f"Expected even tokens, got {N2}"
    N = N2 // 2

    # ---- temb parsing (원본 WanTransformerBlock 그대로) ----
    if temb.ndim == 4:
        # wan2.2 ti2v 케이스는 hidden_states shape이 달라질 수 있어 여기서는 생략/가정
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
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    # ---- split tokens ----
    video = hidden_states[:, :N, :]
    flow  = hidden_states[:, N:, :]

    # ---- rotary split (video/flow 각각 따로 SA 돌리므로 slice) ----
    rotary_video, rotary_flow = _split_rotary_emb(rotary_emb, N)

    # =========================================================
    # 1) Self-attention (shared weight, separate execution)
    # =========================================================
    # video SA
    v_norm = (self.norm1(video.float()) * (1 + scale_msa) + shift_msa).type_as(video)
    v_attn = self.attn1(v_norm, None, None, rotary_video)
    video  = (video.float() + v_attn * gate_msa).type_as(video)

    # flow SA
    f_norm = (self.norm1(flow.float()) * (1 + scale_msa) + shift_msa).type_as(flow)
    f_attn = self.attn1(f_norm, None, None, rotary_flow)
    flow   = (flow.float() + f_attn * gate_msa).type_as(flow)

    # =========================================================
    # 2) Multi-modal cross-attention (shared weight)
    #    - no rotary for now (WanAttention은 q/k에 동일 rotary를 적용하는 설계라
    #      cross-modality q/k 각각 다른 rotary를 넣기 애매해서 안전하게 None 권장)
    # =========================================================
    gate_f2v, gate_v2f = _compute_mm_gate(self, temb)

    # video <- flow
    v2 = self.mm_attn(video, flow, None, None)   # Q=video, KV=flow
    video = (video.float() + gate_f2v * v2.float()).type_as(video)

    # flow <- video
    f2 = self.mm_attn(flow, video, None, None)   # Q=flow, KV=video
    flow = (flow.float() + gate_v2f * f2.float()).type_as(flow)
    # recombine
    hidden_states = torch.cat([video, flow], dim=1)

    # =========================================================
    # 3) Text cross-attention (원본 그대로; video/flow 직접 mixing 없음)
    # =========================================================
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    # =========================================================
    # 4) FFN (원본 그대로)
    # =========================================================
    norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

    return hidden_states


#-------------------------------------------------------------------------
"""
MMCA #2
- MMCA (video<->flow)
- SA
- CA (text)
- FFN
"""
def _wan_block_forward_mm_pre_sa(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
    B, N2, C = hidden_states.shape
    assert N2 % 2 == 0, f"Expected even tokens, got {N2}"
    N = N2 // 2

    # ---- temb parsing (원본 WanTransformerBlock 그대로) ----
    if temb.ndim == 4:
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
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    # ---- split tokens ----
    video = hidden_states[:, :N, :]
    flow  = hidden_states[:, N:, :]

    rotary_video, rotary_flow = _split_rotary_emb(rotary_emb, N)

    # =========================================================
    # 0) Prepare normalized inputs for SA (as in Wan)
    # =========================================================
    v_norm = (self.norm1(video.float()) * (1 + scale_msa) + shift_msa).type_as(video)
    f_norm = (self.norm1(flow.float())  * (1 + scale_msa) + shift_msa).type_as(flow)

    # =========================================================
    # 1) Multi-modal cross-attention BEFORE self-attn
    #    - use v_norm/f_norm to compute CA
    #    - mix with zero-init projection and inject into v_norm/f_norm
    # =========================================================
    gate_f2v, gate_v2f = _compute_mm_gate(self, temb)

    # CA (no rotary)
    v2 = self.mm_attn(v_norm, f_norm, None, None)   # Q=v_norm, KV=f_norm
    f2 = self.mm_attn(f_norm, v_norm, None, None)   # Q=f_norm, KV=v_norm

    # apply gates BEFORE mixing
    v2 = (v2.float() * (gate_f2v if torch.is_tensor(gate_f2v) else gate_f2v)).type_as(v2)
    f2 = (f2.float() * (gate_v2f if torch.is_tensor(gate_v2f) else gate_v2f)).type_as(f2)

    dv, df = _mm_mix_and_split(self, v2, f2)

    # inject into SA inputs
    v_norm = (v_norm.float() + dv.float()).type_as(v_norm)
    f_norm = (f_norm.float() + df.float()).type_as(f_norm)

    # =========================================================
    # 2) Self-attention (shared weight, separate execution)
    # =========================================================
    v_attn = self.attn1(v_norm, None, None, rotary_video)
    video  = (video.float() + v_attn.float() * gate_msa.float()).type_as(video)

    f_attn = self.attn1(f_norm, None, None, rotary_flow)
    flow   = (flow.float() + f_attn.float() * gate_msa.float()).type_as(flow)

    hidden_states = torch.cat([video, flow], dim=1)

    # =========================================================
    # 3) Text cross-attention (원본 그대로)
    # =========================================================
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    # =========================================================
    # 4) FFN (원본 그대로)
    # =========================================================
    norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa.float()).type_as(hidden_states)

    return hidden_states


#-------------------------------------------------------------------------
"""
MMCA #3 
- SA
- MMCA (video<->flow)
- SA result + CA result
- CA (text)
- FFN
"""
def _wan_block_forward_mm_post_sa(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
    B, N2, C = hidden_states.shape
    assert N2 % 2 == 0, f"Expected even tokens, got {N2}"
    N = N2 // 2

    # ---- temb parsing ----
    if temb.ndim == 4:
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
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    video = hidden_states[:, :N, :]
    flow  = hidden_states[:, N:, :]
    rotary_video, rotary_flow = _split_rotary_emb(rotary_emb, N)

    # =========================================================
    # 1) Self-attention (as in your base)
    # =========================================================
    v_norm = (self.norm1(video.float()) * (1 + scale_msa) + shift_msa).type_as(video)
    v_attn = self.attn1(v_norm, None, None, rotary_video)
    video  = (video.float() + v_attn.float() * gate_msa.float()).type_as(video)

    f_norm = (self.norm1(flow.float()) * (1 + scale_msa) + shift_msa).type_as(flow)
    f_attn = self.attn1(f_norm, None, None, rotary_flow)
    flow   = (flow.float() + f_attn.float() * gate_msa.float()).type_as(flow)

    # =========================================================
    # 2) Multi-modal cross-attn AFTER self-attn
    # =========================================================
    gate_f2v, gate_v2f = _compute_mm_gate(self, temb)

    v2 = self.mm_attn(video, flow, None, None)
    f2 = self.mm_attn(flow, video, None, None)

    v2 = (v2.float() * (gate_f2v if torch.is_tensor(gate_f2v) else gate_f2v)).type_as(v2)
    f2 = (f2.float() * (gate_v2f if torch.is_tensor(gate_v2f) else gate_v2f)).type_as(f2)

    dv, df = _mm_mix_and_split(self, v2, f2)

    video = (video.float() + dv.float()).type_as(video)
    flow  = (flow.float()  + df.float()).type_as(flow)

    hidden_states = torch.cat([video, flow], dim=1)

    # =========================================================
    # 3) Text cross-attn
    # =========================================================
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    # =========================================================
    # 4) FFN
    # =========================================================
    norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa.float()).type_as(hidden_states)

    return hidden_states


#-------------------------------------------------------------------------
"""
MMCA #4 (Following UniCon, https://arxiv.org/pdf/2410.11439)
- SA & MMCA (video<->flow)
- SA result + MMCA result
- CA (text)
- FFN
"""
def _wan_block_forward_mm_parallel(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
    B, N2, C = hidden_states.shape
    assert N2 % 2 == 0, f"Expected even tokens, got {N2}"
    N = N2 // 2

    # ---- temb parsing ----
    if temb.ndim == 4:
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
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    video = hidden_states[:, :N, :]
    flow  = hidden_states[:, N:, :]
    rotary_video, rotary_flow = _split_rotary_emb(rotary_emb, N)

    # =========================================================
    # 0) Norm inputs
    # =========================================================
    v_norm = (self.norm1(video.float()) * (1 + scale_msa) + shift_msa).type_as(video)
    f_norm = (self.norm1(flow.float())  * (1 + scale_msa) + shift_msa).type_as(flow)

    # =========================================================
    # 1) Multi-modal cross-attn computed on (v_norm, f_norm)
    #    but injected AFTER self-attn
    # =========================================================
    gate_f2v, gate_v2f = _compute_mm_gate(self, temb)

    v2 = self.mm_attn(v_norm, f_norm, None, None)
    f2 = self.mm_attn(f_norm, v_norm, None, None)

    v2 = (v2.float() * (gate_f2v if torch.is_tensor(gate_f2v) else gate_f2v)).type_as(v2)
    f2 = (f2.float() * (gate_v2f if torch.is_tensor(gate_v2f) else gate_v2f)).type_as(f2)

    dv, df = _mm_mix_and_split(self, v2, f2)

    # =========================================================
    # 2) Self-attn (uses the original v_norm/f_norm)
    # =========================================================
    v_attn = self.attn1(v_norm, None, None, rotary_video)
    video  = (video.float() + v_attn.float() * gate_msa.float()).type_as(video)

    f_attn = self.attn1(f_norm, None, None, rotary_flow)
    flow   = (flow.float() + f_attn.float() * gate_msa.float()).type_as(flow)

    # inject dv/df after SA
    video = (video.float() + dv.float()).type_as(video)
    flow  = (flow.float()  + df.float()).type_as(flow)

    hidden_states = torch.cat([video, flow], dim=1)

    # =========================================================
    # 3) Text cross-attn
    # =========================================================
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    # =========================================================
    # 4) FFN
    # =========================================================
    norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa.float()).type_as(hidden_states)

    return hidden_states



#-------------------------------------------------------------------------
def upgrade_wan_blocks_for_mm(
    model,
    inner_dim,
    eps,
    v_out_scale: float = 0.1,
    use_gate: bool = False,
    mm_mode: str = "base",  # "base" | "pre_sa" | "post_sa" | "parallel"
):
    """
    mm_mode:
      - "base": your current _wan_block_forward_mm (mm after SA, direct add)
      - "pre_sa": CA before SA, inject into v_norm/f_norm via zero-init proj
      - "post_sa": CA after SA, inject into video/flow via zero-init proj
      - "parallel": CA on v_norm/f_norm, inject after SA via zero-init proj
    """
    for block in model.blocks:
        # ============================================================
        # 1) Create multi-modal cross-attention module (shared weight)
        # ============================================================
        mm_attn = type(block.attn2)(
            dim=inner_dim,
            heads=block.attn2.heads,
            dim_head=inner_dim // block.attn2.heads,
            eps=eps,
            added_kv_proj_dim=None,
            cross_attention_dim_head=inner_dim // block.attn2.heads,
            processor=type(block.attn2.processor)(),
        )

        # ============================================================
        # 2) Initialize mm_attn from self-attn (attn1)
        # ============================================================
        with torch.no_grad():
            mm_attn.to_q.weight.copy_(block.attn1.to_q.weight)
            if block.attn1.to_q.bias is not None:
                mm_attn.to_q.bias.copy_(block.attn1.to_q.bias)

            mm_attn.to_k.weight.copy_(block.attn1.to_k.weight)
            if block.attn1.to_k.bias is not None:
                mm_attn.to_k.bias.copy_(block.attn1.to_k.bias)

            mm_attn.to_v.weight.copy_(block.attn1.to_v.weight * v_out_scale)
            if block.attn1.to_v.bias is not None:
                mm_attn.to_v.bias.copy_(block.attn1.to_v.bias * v_out_scale)

            mm_attn.to_out[0].weight.copy_(block.attn1.to_out[0].weight * v_out_scale)
            mm_attn.to_out[0].bias.copy_(block.attn1.to_out[0].bias * v_out_scale)

        block.mm_attn = mm_attn

        # ============================================================
        # 2.5) Add zero-init mixing projection for (v2,f2) -> (dv,df)
        # ============================================================
        block.mm_mix_proj = nn.Linear(2 * inner_dim, 2 * inner_dim, bias=True)
        with torch.no_grad():
            block.mm_mix_proj.weight.zero_()
            block.mm_mix_proj.bias.zero_()

        # ============================================================
        # 3) Timestep-conditional gates (Wan-style)
        # ============================================================
        block.mm_use_gate = use_gate
        dim = inner_dim
        if use_gate:
            block.mm_gate_f2v_table = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
            block.mm_gate_v2f_table = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)

        # ============================================================
        # 4) Replace forward with selected mm variant
        # ============================================================
        if mm_mode == "base":
            block.forward = types.MethodType(_wan_block_forward_mm, block)
        elif mm_mode == "pre_sa":
            block.forward = types.MethodType(_wan_block_forward_mm_pre_sa, block)
        elif mm_mode == "post_sa":
            block.forward = types.MethodType(_wan_block_forward_mm_post_sa, block)
        elif mm_mode == "parallel":
            block.forward = types.MethodType(_wan_block_forward_mm_parallel, block)
        else:
            raise ValueError(f"Unknown mm_mode={mm_mode}. Choose from base|pre_sa|post_sa|parallel.")
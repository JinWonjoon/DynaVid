import copy
import types
import torch

def _split_rotary_emb(rotary_emb, n_video_tokens: int):
    if rotary_emb is None:
        return None, None
    freqs_cos, freqs_sin = rotary_emb
    return (freqs_cos[:, :n_video_tokens, ...], freqs_sin[:, :n_video_tokens, ...]), \
           (freqs_cos[:, n_video_tokens:, ...], freqs_sin[:, n_video_tokens:, ...])

def upgrade_wan_blocks_for_sep_video_flow(
    model,
    sep_LN: bool = True,
    sep_FFN: bool = True,
    # sep_Attn: bool = False,
):
    """
    Keep pretrained modules as VIDEO path:
      - use existing block.norm1/norm2/norm3/ffn/attn1 for video
    Create FLOW copies:
      - block.norm{1,2,3}_flow, block.ffn_flow, block.attn1_flow
    """

    for block in model.blocks:
        # Keep original forward
        if not hasattr(block, "_forward_orig"):
            block._forward_orig = block.forward

        # Create only flow copies (video stays original pretrained)
        if sep_LN and not hasattr(block, "norm1_flow"):
            block.norm1_flow = copy.deepcopy(block.norm1)
            block.norm2_flow = copy.deepcopy(block.norm2)
            block.norm3_flow = copy.deepcopy(block.norm3)

        if sep_FFN and not hasattr(block, "ffn_flow"):
            block.ffn_flow = copy.deepcopy(block.ffn)

        # if sep_Attn and not hasattr(block, "attn1_flow"):
        #     block.attn1_flow = copy.deepcopy(block.attn1)

        block._sep_LN = bool(sep_LN)
        block._sep_FFN = bool(sep_FFN)
        # block._sep_Attn = bool(sep_Attn)

        def _forward_sep(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
            B, N_total, C = hidden_states.shape
            assert N_total % 2 == 0, f"Expected even tokens, got {N_total}"
            N = N_total // 2

            video = hidden_states[:, :N, :]
            flow  = hidden_states[:, N:, :]

            rotary_video, rotary_flow = _split_rotary_emb(rotary_emb, N)

            # ---- temb parsing (match your Wan-style chunk(6) setup)
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

            # =========================================================
            # 1) norm1 (+ shift/scale)  VIDEO uses pretrained norm1
            # =========================================================
            v_norm = (self.norm1(video.float()) * (1 + scale_msa) + shift_msa).type_as(video)

            if getattr(self, "_sep_LN", False):
                f_norm = (self.norm1_flow(flow.float()) * (1 + scale_msa) + shift_msa).type_as(flow)
            else:
                f_norm = (self.norm1(flow.float()) * (1 + scale_msa) + shift_msa).type_as(flow)
            
            # =========================================================
            # 2) self-attn (optionally separate flow attn weights)
            # =========================================================
            x_norm = torch.cat([v_norm, f_norm], dim=1)
            attn_out = self.attn1(x_norm, None, None, rotary_emb)
            v_attn = attn_out[:, :N, :]
            f_attn = attn_out[:, N:, :]
            
            video = (video.float() + v_attn.float() * gate_msa.float()).type_as(video)
            flow  = (flow.float()  + f_attn.float() * gate_msa.float()).type_as(flow)
            
            """
            # TODO. Would be implemented later if needed
            v_attn = self.attn1(v_norm, None, None, rotary_video)

            if getattr(self, "_sep_Attn", False):
                f_attn = self.attn1_flow(f_norm, None, None, rotary_flow)
            else:
                f_attn = self.attn1(f_norm, None, None, rotary_flow)

            video = (video.float() + v_attn.float() * gate_msa.float()).type_as(video)
            flow  = (flow.float()  + f_attn.float() * gate_msa.float()).type_as(flow)
            """

            hidden_states = torch.cat([video, flow], dim=1)

            # =========================================================
            # 3) text cross-attn (norm2 optionally separate for flow)
            # =========================================================
            if getattr(self, "_sep_LN", False):
                # apply pretrained norm2 to video tokens, flow copy to flow tokens
                v2 = self.norm2(video.float()).type_as(video)
                f2 = self.norm2_flow(flow.float()).type_as(flow)
                norm_hidden_states = torch.cat([v2, f2], dim=1)
            else:
                norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)

            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
            hidden_states = hidden_states + attn_output

            # =========================================================
            # 4) FFN (norm3 optionally separate for flow; ffn optionally separate)
            # =========================================================
            video = hidden_states[:, :N, :]
            flow  = hidden_states[:, N:, :]

            if getattr(self, "_sep_LN", False):
                v3 = (self.norm3(video.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(video)
                f3 = (self.norm3_flow(flow.float())  * (1 + c_scale_msa) + c_shift_msa).type_as(flow)
            else:
                v3 = (self.norm3(video.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(video)
                f3 = (self.norm3(flow.float())  * (1 + c_scale_msa) + c_shift_msa).type_as(flow)

            v_ff = self.ffn(v3)

            if getattr(self, "_sep_FFN", False):
                f_ff = self.ffn_flow(f3)
            else:
                f_ff = self.ffn(f3)

            video = (video.float() + v_ff.float() * c_gate_msa.float()).type_as(video)
            flow  = (flow.float()  + f_ff.float() * c_gate_msa.float()).type_as(flow)

            return torch.cat([video, flow], dim=1)

        block.forward = types.MethodType(_forward_sep, block)
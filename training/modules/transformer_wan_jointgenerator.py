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

from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
# from deepspeed.checkpoint import checkpoint, configure

import pdb

logger = logging.get_logger(__name__)


def zero_module(module):
    for p in module.parameters():
        torch.nn.init.zeros_(p)
    return module


class WanTransformer3DModel_JointGenerator(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    _supports_gradient_checkpointing = True
    
    def __init__(self, rgb_model, flow_model, AddAttn=True, FlowCondRGBGen=False):
        super().__init__()
        
        self._config = rgb_model.config
        self.gradient_checkpointing = rgb_model.gradient_checkpointing
        self.AddAttn = AddAttn
        self.FlowCondRGBGen = FlowCondRGBGen
        self.enable_joint_block = True

        self.video_generator = rgb_model
        self.motion_generator = flow_model
        
        joint_blocks = nn.ModuleList()
        for block in self.video_generator.blocks:
            joint_block = JointBlock(block)
            joint_blocks.append(joint_block)
        
        self.joint_blocks = joint_blocks
    
    @property
    def config(self):
        return self._config
    
    def set_joint_block(self, enable):
        self.enable_joint_block = enable
    
    def _gradient_checkpointing_func(self, func, *inputs):
        """
        DeepSpeed ZeRO-3 호환 gradient checkpoint 함수.
        """
        def forward_fn(*args):
            return func(*args)
        return ds_checkpoint(forward_fn, *inputs)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_flow: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_flow: torch.LongTensor,
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
                
        assert hidden_states.shape == hidden_states_flow.shape, "Video and Flow latents should have the same shape!"

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.video_generator.rope(hidden_states)
        rotary_emb_flow = self.motion_generator.rope(hidden_states_flow)

        # 1. Patch embedding
        hidden_states = self.video_generator.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        
        hidden_states_flow = self.motion_generator.patch_embedding(hidden_states_flow)
        hidden_states_flow = hidden_states_flow.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
            timestep_flow = timestep_flow.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        # 2. Condition embedding
        temb, timestep_proj, _, _ = self.video_generator.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        temb_flow, timestep_flow_proj, encoder_hidden_states, encoder_hidden_states_image= self.motion_generator.condition_embedder(
            timestep_flow, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        
        
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
            timestep_flow_proj = timestep_flow_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
            timestep_flow_proj = timestep_flow_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for video_block, motion_block, joint_block in zip(self.video_generator.blocks, self.motion_generator.blocks, self.joint_blocks):
                # Joint block
                if self.enable_joint_block == True:
                    joint_video_output, joint_motion_output = self._gradient_checkpointing_func(
                        joint_block,
                        hidden_states,
                        hidden_states_flow,
                    )
                else:
                    joint_video_output, joint_motion_output = torch.zeros_like(hidden_states), torch.zeros_like(hidden_states_flow)
                
                if self.FlowCondRGBGen:
                    joint_motion_output = torch.zeros_like(joint_motion_output)
                
                if not self.AddAttn:
                    # 3-4. Add joint block outputs to the RGB and condition branches
                    hidden_states = hidden_states + joint_video_output
                    hidden_states_flow = hidden_states_flow + joint_motion_output
                
                # Video block
                if self.AddAttn:
                    hidden_states = self._gradient_checkpointing_func(
                        video_block, hidden_states, encoder_hidden_states, joint_video_output, timestep_proj, rotary_emb
                    )
                else:
                    hidden_states = self._gradient_checkpointing_func(
                        video_block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                    )
                
                # Motion block
                if self.AddAttn:
                    hidden_states_flow = self._gradient_checkpointing_func(
                        motion_block, hidden_states_flow, encoder_hidden_states, joint_motion_output, timestep_flow_proj, rotary_emb_flow
                    )
                else:
                    hidden_states_flow = self._gradient_checkpointing_func(
                        motion_block, hidden_states_flow, encoder_hidden_states, timestep_flow_proj, rotary_emb_flow
                    )
                
                
        else:
            for video_block, motion_block, joint_block in zip(self.video_generator.blocks, self.motion_generator.blocks, self.joint_blocks):
                # Joint block
                if self.enable_joint_block == True:
                    joint_video_output, joint_motion_output = joint_block(hidden_states, hidden_states_flow)
                else:
                    joint_video_output, joint_motion_output = torch.zeros_like(hidden_states), torch.zeros_like(hidden_states_flow)
                
                if self.FlowCondRGBGen:
                    joint_motion_output = torch.zeros_like(joint_motion_output)
                
                if not self.AddAttn:
                    # 3-4. Add joint block outputs to the RGB and condition branches
                    hidden_states = hidden_states + joint_video_output
                    hidden_states_flow = hidden_states_flow + joint_motion_output
                
                # Video block
                if self.AddAttn:
                    hidden_states = video_block(hidden_states, encoder_hidden_states, joint_video_output, timestep_proj, rotary_emb)
                else:
                    hidden_states = video_block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                
                # Motion block
                if self.AddAttn:
                    hidden_states_flow = motion_block(hidden_states_flow, encoder_hidden_states, joint_motion_output, timestep_flow_proj, rotary_emb_flow)
                else:
                    hidden_states_flow = motion_block(hidden_states_flow, encoder_hidden_states, timestep_flow_proj, rotary_emb_flow)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.video_generator.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
            shift_flow, scale_flow = (self.motion_generator.scale_shift_table.unsqueeze(0).to(temb_flow.device) + temb_flow.unsqueeze(2)).chunk(2, dim=2)
            shift_flow = shift_flow.squeeze(2)
            scale_flow = scale_flow.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.video_generator.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
            shift_flow, scale_flow = (self.motion_generator.scale_shift_table.to(temb_flow.device) + temb_flow.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        shift_flow = shift.to(hidden_states.device)
        scale_flow = scale.to(hidden_states.device)

        hidden_states = (self.video_generator.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.video_generator.proj_out(hidden_states)
        hidden_states_flow = (self.motion_generator.norm_out(hidden_states_flow.float()) * (1 + scale_flow) + shift_flow).type_as(hidden_states_flow)
        hidden_states_flow = self.motion_generator.proj_out(hidden_states_flow)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        
        hidden_states_flow = hidden_states_flow.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states_flow = hidden_states_flow.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output_flow = hidden_states_flow.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, output_flow)

        return Transformer2DModelOutput(sample=output), Transformer2DModelOutput(sample=output_flow)
    
    @classmethod
    def from_pretrained(cls, model_path, subfolder, **additional_kwargs):
        
        AddAttn = additional_kwargs.pop('AddAttn', False)
        PretrainedFlowGen = additional_kwargs.pop('PretrainedFlowGen', False)
        FlowCondRGBGen = additional_kwargs.pop('FlowCondRGBGen', False)
        resume_from_pretrained = additional_kwargs.pop('resume_from_pretrained', None)
        motion_generator_ckpt = additional_kwargs.pop('motion_generator_ckpt', None)

        assert (resume_from_pretrained is not None) or (motion_generator_ckpt is not None), "We should use either pre-trained flow generation model or pre-trained joint generation model!"
        
        if AddAttn:
            video_generator = WanTransformer3DModel_AddAttn.from_pretrained(model_path, subfolder=subfolder)
            motion_generator = WanTransformer3DModel_AddAttn.from_pretrained(model_path, subfolder=subfolder)
        else:
            video_generator = WanTransformer3DModel.from_pretrained(model_path, subfolder=subfolder)
            motion_generator = WanTransformer3DModel.from_pretrained(model_path, subfolder=subfolder)
        
        # Load ckpt for motion generator
        if motion_generator_ckpt is not None:
            sd = torch.load(motion_generator_ckpt, map_location='cpu')['module']
            missing, unexpected = motion_generator.load_state_dict(sd)
            assert len(missing) == 0 or len(unexpected) == 0, "Choose wrong checkpoint!"
            
            del sd
        
        # Init. Joint Generator
        model = cls(video_generator, motion_generator, AddAttn, FlowCondRGBGen)
        
        # Load ckpt for joint generator
        if resume_from_pretrained is not None:
            sd = torch.load(resume_from_pretrained, map_location='cpu')['module']
            missing, unexpected = model.load_state_dict(sd)
            assert len(missing) == 0 or len(unexpected) == 0, "Choose wrong checkpoint!"
            
            del sd
        
        del video_generator
        del motion_generator
        torch.cuda.empty_cache()
        
        return model
        


class WanTransformer3DModel_AddAttn(WanTransformer3DModel):
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 1. Load WanTransformer3DModel
        model = WanTransformer3DModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # 2. Change transformer blocks
        num_attention_heads = model.config.num_attention_heads
        attention_head_dim = model.config.attention_head_dim
        ffn_dim = model.config.ffn_dim
        attention_head_dim = model.config.attention_head_dim
        qk_norm = model.config.qk_norm
        cross_attn_norm = model.config.cross_attn_norm
        eps = model.config.eps
        added_kv_proj_dim = model.config.added_kv_proj_dim
        num_layers = model.config.num_layers
        
        inner_dim = num_attention_heads * attention_head_dim        
        
        blocks_AddAttn = nn.ModuleList(
            [
                WanTransformerBlock_AddAttn(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # Copy weight from original blocks
        for block_orig, block_AddAttn in zip(model.blocks, blocks_AddAttn):
            sd = block_orig.state_dict()
            missing, unexpected = block_AddAttn.load_state_dict(sd)
            assert len(missing) == 0 and len(unexpected) == 0
        
        model.blocks = blocks_AddAttn
        
        return model


@maybe_allow_in_graph
class JointBlock(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        
        out_dim = original_block.attn1.to_out[0].out_features
        
        dropout = 0.0
        activation_fn = 'gelu-approximate'
        final_dropout=0.0
        ff_inner_dim = out_dim*2
        ff_bias=True
        
        self.ff = FeedForward(out_dim*2,dropout=dropout,activation_fn=activation_fn,final_dropout=final_dropout,inner_dim=ff_inner_dim,bias=ff_bias,)
        
        self.ff.net[2] = zero_module(self.ff.net[2])

        self.joint_scale = 1.0
       
    def forward(
        self, 
        hidden_states: torch.Tensor,
        cond_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        
        x = torch.cat([hidden_states, cond_hidden_states], dim=2)
        x = self.ff(x)
        joint_output_rgb, joint_output_cond = x.chunk(2, dim=2)

        return joint_output_rgb * self.joint_scale, joint_output_cond * self.joint_scale
    

@maybe_allow_in_graph
class WanTransformerBlock_AddAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        joint_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
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
        
        # Add joint_hidden_states from joint block
        norm_hidden_states = norm_hidden_states + joint_hidden_states
        
        # attention
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
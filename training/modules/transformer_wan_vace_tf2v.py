from typing import Any, Dict, List, Tuple, Optional, Union
from einops import rearrange, repeat
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import WanTransformer3DModel, WanVACETransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

import pdb

logger = logging.get_logger(__name__)


class WanVACETransformer3DModel_TF2V(WanVACETransformer3DModel):
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, subfolder, noise_sigma=0.0, layer6=False):
        # 1. Among WanVACE, Load only WanTransformer3DModel
        wan_ti2v_model = WanTransformer3DModel.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        
        
        config = copy.deepcopy(dict(wan_ti2v_model.config))
        config['_class_name'] = 'WanVACETransformer3DModel'
        if layer6:
            config['vace_layers'] = [0, 5, 10, 15, 20, 25]
        else:
            config['vace_layers'] = [0, 3, 6, 9, 12 ,15, 18, 21, 24, 27]
        
        config['vace_in_channels'] = 48  # 1536 = 6 (Plucker embedding) x 16 x 16 (Due to Unshuffle 16) / 4 (Two Conv2d)
        wan_ti2v_vace_model = cls(**config)
        
        # Save noise level in the class
        wan_ti2v_vace_model.noise_sigma = noise_sigma
        
        # Copy parameters from checkpoint!
        missing, unexpected = wan_ti2v_vace_model.load_state_dict(wan_ti2v_model.state_dict(), strict=False)
        print(f"VACE, Missing key: {missing}")
        
        
        del wan_ti2v_model
        
        return wan_ti2v_vace_model
    
    def add_gaussian_noise(self, x, noise_level=None):
        """
        x: control_hidden_states tensor, shape (B, F, C, H, W)
        noise_level: float, optional; overrides self.noise_sigma
        
        Adds Gaussian noise to x according to given noise_level (σ).
        Returns noisy tensor.
        """
        
        if noise_level is None:
            noise_level = self.noise_sigma
        
        if noise_level <= 0:
            return x
        
        # Generate noise
        noise = torch.randn_like(x)
        
        # Inject noise
        return x + noise * noise_level
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        control_hidden_states: torch.Tensor = None,
        control_hidden_states_scale: torch.Tensor = None,
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
        
        # Add Gaussian noise to control_hidden_states
        if control_hidden_states is not None:
            control_hidden_states = self.add_gaussian_noise(control_hidden_states)
            
        
        if control_hidden_states.shape != hidden_states.shape:
            control_hidden_states = rearrange(control_hidden_states, 'b f c h w -> b c f h w')

        if control_hidden_states_scale is None:
            control_hidden_states_scale = control_hidden_states.new_ones(len(self.config.vace_layers))
        control_hidden_states_scale = torch.unbind(control_hidden_states_scale)
        if len(control_hidden_states_scale) != len(self.config.vace_layers):
            raise ValueError(
                f"Length of `control_hidden_states_scale` {len(control_hidden_states_scale)} should be "
                f"equal to {len(self.config.vace_layers)}."
            )

        # 1. Rotary position embedding
        rotary_emb = self.rope(hidden_states)

        # 2. Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        
        

        control_hidden_states = self.vace_patch_embedding(control_hidden_states)
        control_hidden_states = control_hidden_states.flatten(2).transpose(1, 2)
        control_hidden_states_padding = control_hidden_states.new_zeros(
            batch_size, hidden_states.size(1) - control_hidden_states.size(1), control_hidden_states.size(2)
        )
        control_hidden_states = torch.cat([control_hidden_states, control_hidden_states_padding], dim=1)

        # 3. Time embedding
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # 4. Image embedding
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 5. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # Prepare VACE hints
            control_hidden_states_list = []
            for i, block in enumerate(self.vace_blocks):
                conditioning_states, control_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, control_hidden_states, timestep_proj, rotary_emb
                )
                control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[i]))
            control_hidden_states_list = control_hidden_states_list[::-1]

            for i, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                if i in self.config.vace_layers:
                    control_hint, scale = control_hidden_states_list.pop()
                    hidden_states = hidden_states + control_hint * scale
        else:
            # Prepare VACE hints
            control_hidden_states_list = []
            for i, block in enumerate(self.vace_blocks):
                conditioning_states, control_hidden_states = block(
                    hidden_states, encoder_hidden_states, control_hidden_states, timestep_proj, rotary_emb
                )
                control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[i]))
            control_hidden_states_list = control_hidden_states_list[::-1]

            for i, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                if i in self.config.vace_layers:
                    control_hint, scale = control_hidden_states_list.pop()
                    hidden_states = hidden_states + control_hint * scale

        # 6. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

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
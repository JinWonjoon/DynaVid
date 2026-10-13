import torch
from einops import rearrange, repeat
from training.utils import rgb_hsv_flow_inverse, rgb_to_flow_ours_inverse

import pdb

def decode_flow_latents(vae, latents: torch.Tensor, scale_factor=29.5) -> torch.Tensor:
    """
    Args.
        latents: BFCHW
    Return.
        video: video decoded from latents, (BF)CHW
    """
    
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    ) # BCFHW
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    ) # BCFHW
    latents = rearrange(latents, 'b f c h w -> b c f h w') / latents_std + latents_mean
    flow_RGB_normalized = vae.decode(latents, return_dict=False)[0]
    flow_RGB_normalized = rearrange(flow_RGB_normalized, 'b c f h w -> (b f) c h w')
    
    flow = rgb_to_flow_ours_inverse(flow_RGB_normalized, scale = scale_factor) # (BF)CHW
    # flow = rgb_hsv_flow_inverse(flow_RGB_normalized, sigma=sigma) # (BF)CHW
    
    return flow
from typing import Any, Dict, List, Tuple
from einops import rearrange, repeat
import numpy as np

from accelerate.logging import get_logger
import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanPipeline,
    WanTransformer3DModel,
    WanVACETransformer3DModel,
    UniPCMultistepScheduler,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import T5TokenizerFast, UMT5EncoderModel
from typing_extensions import override

from training.schemas import Components
from training.trainer import Trainer
from training.utils import unwrap_model
from training.pipeline.wan_vace_tf2v_pipeline import WanVACEPipeline_TF2V
from training.modules.transformer_wan_vace_tf2v import WanVACETransformer3DModel_TF2V
from training.constants import LOG_LEVEL, LOG_NAME

from ..utils import register

logger = get_logger(LOG_NAME, LOG_LEVEL)

import pdb

class WanVACETF2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = WanVACEPipeline_TF2V # Should be changed

        components.tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        noise_sigma = self.args.noise_sigma if hasattr(self.args, "noise_sigma") else 0.0
        components.transformer = WanVACETransformer3DModel_TF2V.from_pretrained(model_path, subfolder="transformer", noise_sigma=noise_sigma)

        components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")

        components.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, shift=self.args.shift_training, subfolder="scheduler")

        return components

    @override
    def initialize_pipeline(self) -> WanVACEPipeline_TF2V:
        scheduler = UniPCMultistepScheduler.from_config(self.components.scheduler.config, flow_shift=self.args.shift_training)
        pipe = WanVACEPipeline_TF2V(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=scheduler,
        )
        return pipe
    
    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [], "encoded_flow": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            encoded_flow = sample["encoded_flow"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)
            ret["encoded_flow"].append(encoded_flow)


        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])
        ret["encoded_flow"] = torch.stack(ret["encoded_flow"])

        return ret


    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        
        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype
        
        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )
        
        
        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(False)
                    for name, module in component.named_modules():
                        if "vace" in name.lower():  # 이름에 'vace' 포함 여부 확인 (대소문자 무시)
                            for param in module.parameters():
                                param.requires_grad_(True)
                                
                    # component.requires_grad_(True)
                else:
                    component.requires_grad_(False)
                    
        #################################################
        # FreezeExceptCamEmb
        # Only train CameraEmbedding module while freezing other parameters
        #################################################
        if self.args.FreezeExceptCamEmb:
            raise NotImplementedError("Do not use FreezeExceptCamEmb!")
        
        
        if self.args.training_type == "lora":
            raise NotImplementedError("Do not use Lora!")
            
        ignore_list = self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    
    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))


    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Mean, STD!!")
        
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        # latent = latent_dist.sample() * vae.config.scaling_factor
        
        latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(local_rank, dtype))
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(local_rank, dtype)

        encoded_video = (latent - latents_mean) * latents_std

        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    
    
    @override
    def compute_loss(self, batch) -> torch.Tensor:
        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        latent_flow = batch["encoded_flow"]
        
        batch_size, num_channels, num_frames, height, width = latent.shape
        
        
        
        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Get noisy input
        noise = torch.randn_like(latent)
        
        # Sample a random timestep for each sample
        timestep_id = torch.randint(0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device='cpu').long()
        timesteps = self.components.scheduler.timesteps[timestep_id].to(device=self.accelerator.device)
        
        
        # Add noise according to flow matching.
        # sigmas = self.get_sigmas(self.components.scheduler, timesteps, n_dim=latent.ndim, dtype=self.state.weight_dtype)
        # latent_noisy = (1.0 - sigmas) * latent + sigmas * noise
        latent_noisy = self.components.scheduler.scale_noise(
            sample=latent,
            timestep=timesteps,
            noise=noise,
        )
        
        # target: flow matching velocity (eps - x0)
        target = noise - latent
        
        latent_noisy = latent_noisy.to(dtype=self.state.weight_dtype)
        target = target.to(dtype=self.state.weight_dtype)
        latent_flow = latent_flow.to(dtype=self.state.weight_dtype)
        
        
        model_output = self.components.transformer(
            hidden_states=latent_noisy,
            timestep=timesteps,
            encoder_hidden_states=prompt_embedding,
            control_hidden_states=latent_flow,
            return_dict=False,
        )[0]
        
        loss = torch.mean(((model_output.float() - target.float()) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()
        
        return loss


    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: WanVACEPipeline_TF2V
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        # prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]
        prompt_embedding, latent_flow = eval_data["prompt_embedding"], eval_data["flow_latent"]

        with torch.cuda.amp.autocast(enabled=True, dtype=self.state.weight_dtype):
            video_latent_generate = pipe(
                num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                prompt=None,
                prompt_embeds=prompt_embedding,
                generator=self.state.generator,
                num_inference_steps=self.args.num_inference_steps,
                output_type='latent',
                latent_flow=latent_flow,
                guidance_end=0.1,
            ).frames # BCFHW
            video_generate = self.decode_latents(rearrange(video_latent_generate, 'b c f h w -> b f c h w').to(self.accelerator.device)) # B,F,C,H,W
        video = ((video_generate + 1.) / 2. * 255.)[0].float().clip(0., 255.).to(torch.uint8) # CFHW
        video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
        
        return [("synthesized_video", video)]



register("wan-tf2v-vace", "lora", WanVACETF2VLoraTrainer)


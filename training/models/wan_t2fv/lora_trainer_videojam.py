from typing import Any, Dict, List, Tuple
from einops import rearrange, repeat

import torch
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanPipeline,
    WanTransformer3DModel,
    UniPCMultistepScheduler,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import T5TokenizerFast, UMT5EncoderModel
from typing_extensions import override

from training.constants import LOG_LEVEL, LOG_NAME
from training.schemas import Components
from training.trainer import Trainer
from training.utils import unwrap_model
# from training.pipeline.wan_t2f_pipeline import WanTextToFlowPipeline
from training.modules.transformer_wan_videojam import WanTransformer3DModel_VideoJAM
from training.pipeline.wan_t2fv_videojam_pipeline import WanPipeline_T2FV_VideoJAM

from ..utils import register

import pdb

logger = get_logger(LOG_NAME, LOG_LEVEL)

class WanVideoJAMT2FVLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        # TODO. Change.
        components.pipeline_cls = WanPipeline_T2FV_VideoJAM 

        components.tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = WanTransformer3DModel_VideoJAM.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")

        components.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, shift=5.0, subfolder="scheduler")

        return components

    @override
    def initialize_pipeline(self) -> WanPipeline_T2FV_VideoJAM: # WanTextToFlowPipeline:
        scheduler = UniPCMultistepScheduler.from_config(self.components.scheduler.config, flow_shift=5.0)
        
        #WanTextToFlowPipeline(
            
        pipe = WanPipeline_T2FV_VideoJAM( 
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=scheduler,
        )
        return pipe

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
    def compute_loss(self, batch, use_motion_reg=False) -> torch.Tensor:
        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        latent_flow = batch["encoded_flow"]
        
        batch_size, num_channels, num_frames, height, width = latent_flow.shape
        
        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent_flow.dtype)

        # Get noisy input
        noise = torch.randn_like(latent)
        noise_flow = torch.randn_like(latent_flow)
        
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
        latent_flow_noisy = self.components.scheduler.scale_noise(
            sample=latent_flow,
            timestep=timesteps,
            noise=noise_flow,
        )
        
        # target: flow matching velocity (eps - x0)
        target = noise - latent
        target_flow = noise_flow - latent_flow
        
        latent_noisy = latent_noisy.to(dtype=self.state.weight_dtype)
        target = target.to(dtype=self.state.weight_dtype)
        latent_flow_noisy = latent_flow_noisy.to(dtype=self.state.weight_dtype)
        target_flow = target_flow.to(dtype=self.state.weight_dtype)
        
        model_output = self.components.transformer(
            hidden_states=latent_noisy,
            flow_hidden_states = latent_flow_noisy,
            timestep=timesteps,
            encoder_hidden_states=prompt_embedding,
            return_dict=False,
        )
        
        model_output_RGB, model_output_flow = model_output[0], model_output[1]
        
        target_total = torch.cat([target, target_flow], dim=0)
        model_output_total = torch.cat([model_output_RGB, model_output_flow], dim=0)
        
        # Backup.
        # loss_RGB = torch.mean(((model_output_RGB[batch_size//2:].float() - target[batch_size//2:].float()) ** 2).reshape(batch_size//2, -1), dim=1)  # B,2
        # loss_RGB = torch.mean(((model_output_RGB.float() - target.float()) ** 2).reshape(batch_size//2, -1), dim=1)  # B,2
        # loss_flow = torch.mean(((model_output_flow.float() - target_flow.float()) ** 2).reshape(batch_size, -1), dim=1)  # B,2
        # loss = torch.cat([loss_RGB, loss_flow], dim=0)
        # loss = loss.mean()
        if use_motion_reg:
            # Compute loss only for motion regularization part
            loss = torch.mean(((model_output_flow.float() - target_flow.float()) ** 2).reshape(batch_size, -1), dim=1)
            loss = loss.mean()
        else:
            loss = torch.mean(((model_output_total.float() - target_total.float()) ** 2).reshape(batch_size * 2, -1), dim=1)
            loss = loss.mean()
        
        return loss
    
    


    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: WanPipeline_T2FV_VideoJAM
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        # prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]
        prompt_embedding, neg_prompt_embedding = eval_data["prompt_embedding"], eval_data["neg_prompt_embedding"]

        with torch.cuda.amp.autocast(enabled=True, dtype=self.state.weight_dtype):
            video_flow_latent_generate = pipe(
                num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                prompt=None,
                prompt_embeds=prompt_embedding,
                negative_prompt_embeds=neg_prompt_embedding,
                generator=self.state.generator,
                num_inference_steps=self.args.num_inference_steps,
                output_type='latent',
            ).frames # BCFHW
            
            video_latent_generate, flow_latent_generate = video_flow_latent_generate.chunk(2, dim = 0)
            video_generate = self.decode_latents(rearrange(video_latent_generate, 'b c f h w -> b f c h w').to(self.accelerator.device))
            flow_generate = self.decode_flow_latents(rearrange(flow_latent_generate, 'b c f h w -> b f c h w').to(self.accelerator.device)) # B F C H W -> BF,C,H,W
            
            video = ((video_generate + 1.) / 2. * 255.)[0].float().clip(0., 255.).to(torch.uint8)
            video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
                     
        return [("synthesized_video", video), ("synthesized_flow", flow_generate)]


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
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)
        
        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            self.__prepare_saving_loading_hooks(transformer_lora_config)
            
        ignore_list = self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    # mangling
    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))
    

register("wan-t2fv-videojam", "lora", WanVideoJAMT2FVLoraTrainer)

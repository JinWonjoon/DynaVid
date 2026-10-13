from typing import Any, Callable, Dict, List, Optional, Union
import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_wan_vace import WanVACEPipeline, EXAMPLE_DOC_STRING
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.utils import is_torch_xla_available, replace_example_docstring, logging
from diffusers.image_processor import PipelineImageInput

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanVACETransformer3DModel ,
)
from diffusers.video_processor import VideoProcessor

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

import pdb


class WanVACEPipeline_TF2V(WanVACEPipeline):

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        video: Optional[List[PipelineImageInput]] = None,
        mask: Optional[List[PipelineImageInput]] = None,
        reference_images: Optional[List[PipelineImageInput]] = None,
        conditioning_scale: Union[float, List[float], torch.Tensor] = 1.0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_cam: float = 1.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        latent_flow: Optional[torch.Tensor] = None,
        guidance_end: float = 1.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            video (`List[PIL.Image.Image]`, *optional*):
                The input video or videos to be used as a starting point for the generation. The video should be a list
                of PIL images, a numpy array, or a torch tensor. Currently, the pipeline only supports generating one
                video at a time.
            mask (`List[PIL.Image.Image]`, *optional*):
                The input mask defines which video regions to condition on and which to generate. Black areas in the
                mask indicate conditioning regions, while white areas indicate regions for generation. The mask should
                be a list of PIL images, a numpy array, or a torch tensor. Currently supports generating a single video
                at a time.
            reference_images (`List[PIL.Image.Image]`, *optional*):
                A list of one or more reference images as extra conditioning for the generation. For example, if you
                are trying to inpaint a video to change the character, you can pass reference images of the new
                character here. Refer to the Diffusers [examples](https://github.com/huggingface/diffusers/pull/11582)
                and original [user
                guide](https://github.com/ali-vilab/VACE/blob/0897c6d055d7d9ea9e191dce763006664d9780f8/UserGuide.md)
                for a full list of supported tasks and use cases.
            conditioning_scale (`float`, `List[float]`, `torch.Tensor`, defaults to `1.0`):
                The conditioning scale to be applied when adding the control conditioning latent stream to the
                denoising latent stream in each control layer of the model. If a float is provided, it will be applied
                uniformly to all layers. If a list or tensor is provided, it should have the same length as the number
                of control layers in the model (`len(transformer.config.vace_layers)`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        # Monkey patch.. before fixing in the original Diffusers
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # Simplification of implementation for now
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError("Passing a list of prompts is not yet supported. This may be supported in the future.")
        if num_videos_per_prompt != 1:
            raise ValueError(
                "Generating multiple videos per prompt is not yet supported. This may be supported in the future."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            video,
            mask,
            reference_images,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)


        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        if guidance_scale_cam > 1.0:
            assert guidance_scale > 1.0, "Apply CFG to both text and cam"
            do_classifier_free_guidance_cam = True
        else:
            do_classifier_free_guidance_cam = False
            

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype

        if isinstance(conditioning_scale, (int, float)):
            conditioning_scale = [conditioning_scale] * len(self.transformer.config.vace_layers)
        if isinstance(conditioning_scale, list):
            if len(conditioning_scale) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {len(conditioning_scale)} does not match number of layers {len(self.transformer.config.vace_layers)}."
                )
            conditioning_scale = torch.tensor(conditioning_scale)
        if isinstance(conditioning_scale, torch.Tensor):
            if conditioning_scale.size(0) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {conditioning_scale.size(0)} does not match number of layers {len(self.transformer.config.vace_layers)}."
                )
            conditioning_scale = conditioning_scale.to(device=device, dtype=transformer_dtype)

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        video, mask, reference_images = self.preprocess_conditions(
            video,
            mask,
            reference_images,
            batch_size,
            height,
            width,
            num_frames,
            torch.float32,
            device,
        )
        num_reference_images = len(reference_images[0])

        ########################################
        # Instead of using video/mask/ref_image, we use plucker embedding as condition.
        ########################################
        # conditioning_latents = self.prepare_video_latents(video, mask, reference_images, generator, device)
        # mask = self.prepare_masks(mask, reference_images, generator)
        # conditioning_latents = torch.cat([conditioning_latents, mask], dim=1)
        # conditioning_latents = conditioning_latents.to(transformer_dtype)

        conditioning_latents = latent_flow.to(transformer_dtype)
        

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames + num_reference_images * self.vae_scale_factor_temporal,
            torch.float32,
            device,
            generator,
            latents,
        )
        
        if conditioning_latents.shape[2] != latents.shape[2]:
            logger.warning(
                "The number of frames in the conditioning latents does not match the number of frames to be generated. Generation quality may be affected."
            )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # if self.config.boundary_ratio is not None:
        #     boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        # else:
        #     boundary_timestep = None
        boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale
                
                if (t / self.scheduler.num_train_timesteps) < (1-guidance_end):
                    current_conditioning_scale = torch.zeros_like(conditioning_scale)
                else:
                    current_conditioning_scale = conditioning_scale

                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        control_hidden_states=conditioning_latents,
                        control_hidden_states_scale=current_conditioning_scale,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    
                if do_classifier_free_guidance_cam:
                    negative_conditioning_latents = torch.zeros_like(conditioning_latents)
                    
                    with current_model.cache_context("uncond"):
                        noise_uncond_cam = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            control_hidden_states=negative_conditioning_latents,
                            control_hidden_states_scale=current_conditioning_scale,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        
                        # Following CMG
                        # noise_uncond_all = current_model(
                        #     hidden_states=latent_model_input,
                        #     timestep=timestep,
                        #     encoder_hidden_states=negative_prompt_embeds,
                        #     control_hidden_states=negative_conditioning_latents,
                        #     control_hidden_states_scale=current_conditioning_scale,
                        #     attention_kwargs=attention_kwargs,
                        #     return_dict=False,
                        # )[0]
                        
                        # Following AC3D
                        noise_uncond_text = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            control_hidden_states=conditioning_latents,
                            control_hidden_states_scale=current_conditioning_scale,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        
                    # Following CMG (https://arxiv.org/pdf/2410.10802)
                    # noise_pred = noise_uncond_all + current_guidance_scale * (noise_uncond_cam - noise_uncond_all) + guidance_scale_cam * (noise_pred - noise_uncond_cam)
                    # Following AC3D (https://arxiv.org/pdf/2411.18673)
                    noise_pred = noise_pred + (guidance_scale - 1) * (noise_pred - noise_uncond_text) + (guidance_scale_cam - 1) * (noise_pred - noise_uncond_cam)


                elif self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            control_hidden_states=conditioning_latents,
                            control_hidden_states_scale=current_conditioning_scale,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents[:, :, num_reference_images:]
            latents = latents.to(vae_dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
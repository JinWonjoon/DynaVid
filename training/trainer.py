import os
import hashlib
import json
import logging
import math
import datetime
import numpy as np
import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
from einops import rearrange, repeat

import diffusers
import torch
import transformers
import wandb
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
    broadcast_object_list,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from safetensors.torch import save_file, load_file

from training.schemas import Args, Components, State
from training.constants import LOG_LEVEL, LOG_NAME
from training.datasets import ImgFlowVideoDataset
from training.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
    flow_to_color,
    rgb_hsv_flow_inverse,
    rgb_to_flow_ours_inverse,
    custom_meshgrid,
)
from training.datasets.utils import (
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)

from tqdm import tqdm
import pdb

logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}


class Trainer:
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: List[str] = None
    
    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        
        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None
        if self.accelerator.state.deepspeed_plugin is not None:
            # Check the Sync gradient accumulation steps between Accelerator and DeepSpeed Engine
            assert self.accelerator.gradient_accumulation_steps == self.accelerator.state.deepspeed_plugin.gradient_accumulation_steps, f"Your config files should have the same gradient_accumulation_steps. Now, bash file has {self.accelerator.gradient_accumulation_steps} and DeepSpeed yaml file has {self.accelerator.state.deepspeed_plugin.gradient_accumulation_steps}"
        
        self.do_mixed_batch_training = self.args.do_mixed_batch_training
        self.do_motion_regularization = self.args.do_motion_regularization
        
        if self.do_mixed_batch_training or self.do_motion_regularization:
            self.large_scale_dataset: Dataset = None
            self.large_scale_data_loader: DataLoader = None
    
    
    def _init_distributed(self):
        project_dir  = Path(self.args.output_dir)
        logging_dir  = project_dir / "tmp_logs"          # 나중에 덮어쓸 예정
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to
        
        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )
        
        # 2️⃣  timestamp(run‑id) 생성 & broadcast
        run_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if accelerator.is_main_process else ""
        [run_id] = broadcast_object_list([run_id])  # 모든 rank 동일 run_id 획득

        # 3️⃣  최종 output_dir / logging_dir 확정
        final_out_dir  = project_dir / f"{self.args.run_name}-{run_id}"
        final_log_dir  = final_out_dir / "logs"

        if accelerator.is_main_process:
            final_log_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()  # 디렉터리 생성 동기화

        # 4️⃣  각종 경로 갱신
        self.args.output_dir                            = final_out_dir
        accelerator.project_configuration.project_dir   = final_out_dir
        accelerator.project_configuration.logging_dir   = final_log_dir

        accelerator.init_trackers(
            project_name=self.args.model_name,
            config=vars(self.args),
            init_kwargs={
                "wandb": {
                    "dir": final_log_dir,  # 로그 저장 경로
                    "name": self.args.run_name,
                }
            }
        )
        
        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)


    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)
    
    
    def check_setting(self) -> None:
        # Check for unload_list
        if self.UNLOAD_LIST is None:
            logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")
    
    
    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        self.prepare_trackers() # prepare for the first validation before training.
        self.prepare_resume_from_pretrained()
        if self.args.do_validation:
            self.prepare_for_validation()
        if self.do_mixed_batch_training:
            self.mixed_batch_train()
        elif self.do_motion_regularization:
            self.motion_reg_train()
        else:
            self.train()
    
    
    def prepare_models(self) -> None:
        logger.info("Initializing models")

        """
        For wan2.2, do not use slicing and tiling!!!!!!!
        """
        # if self.components.vae is not None:
        #     if self.args.enable_slicing:
        #         self.components.vae.enable_slicing()
        #     if self.args.enable_tiling:
        #         self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config
    
    
    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")
        
        self.dataset = ImgFlowVideoDataset(
            **(self.args.model_dump()),
            device=self.accelerator.device,
            max_num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            trainer=self,
        )

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )
        
        if self.do_mixed_batch_training or self.do_motion_regularization:
            logger.info("Initializing Large-scale data and dataloader in Synthetic data training")
            
            data_root_orig = self.args.data_root
            self.args.data_root = self.args.large_scale_data_root
            
            self.large_scale_dataset = ImgFlowVideoDataset(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
                do_forced_filtering=True,
            )
            
            self.large_scale_data_loader = torch.utils.data.DataLoader(
                self.large_scale_dataset,
                collate_fn=self.collate_fn,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                shuffle=True,
            )
            
            self.args.data_root = data_root_orig
            
            
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
                if self.args.training_type == "sft" and attr_name == "transformer" and not self.args.FreezeExceptCamEmb:
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)
                    
        #################################################
        # FreezeExceptCamEmb
        # Only train CameraEmbedding module while freezing other parameters
        #################################################
        if self.args.FreezeExceptCamEmb:
            self.components.transformer.camera_embedder.requires_grad_(True)
        
        
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
    
    
    
    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        if self.args.training_type == "sft" or self.args.training_type == "lora":
            cast_training_params([self.components.transformer], dtype=torch.float32)
            # For LoRA, we only want to train the LoRA weights
            # For SFT, we want to train all the parameters
            trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.parameters()))
            trainable_parameters_name = [p[0] for p in filter(lambda p: p[1].requires_grad, self.components.transformer.named_parameters())]
        else:
            raise NotImplementedError("Choose training_type among 'sft', 'lora', 'controlnet'")
            
        print("-"*200)
        print(f"Training type: {self.args.training_type}")
        print(f"Trainable parameters: {trainable_parameters_name}")
        print("-"*200)
        
        trainable_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [trainable_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)
        
        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )
        
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes
        
        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    
    
    def prepare_for_training(self) -> None:
        if self.do_mixed_batch_training or self.do_motion_regularization:
            self.components.transformer, self.optimizer, self.data_loader, self.large_scale_data_loader, self.lr_scheduler = self.accelerator.prepare(
                self.components.transformer, self.optimizer, self.data_loader, self.large_scale_data_loader, self.lr_scheduler
            )
            num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
            if self.do_motion_regularization:
                num_update_steps_per_epoch = num_update_steps_per_epoch * 2
        else:
            self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
                self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler
            )
            num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch
        
        
    # def prepare_for_validation(self):
    #     validation_prompts = load_prompts(self.args.validation_dir / self.args.validation_prompts)

    #     if self.args.validation_images is not None:
    #         validation_images = load_images(self.args.validation_dir / self.args.validation_images)
    #     else:
    #         validation_images = [None] * len(validation_prompts)

    #     if self.args.validation_videos is not None:
    #         validation_videos = load_videos(self.args.validation_dir / self.args.validation_videos)
    #     else:
    #         validation_videos = [None] * len(validation_prompts)

    #     self.state.validation_prompts = validation_prompts
    #     self.state.validation_images = validation_images
    #     self.state.validation_videos = validation_videos

    #     self.validate(0)
    
    
    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        self.accelerator.init_trackers(tracker_name, config=self.args.model_dump())
    
    
    def prepare_for_validation(self):

        data_root = self.args.validation_dir
        metadata_path = data_root / "metadata.jsonl"
        
        assert metadata_path.is_file(), "For this dataset type, you need metadata.jsonl or metadata_revised.jsonl or metadata_reformat.jsonl or metadata_flow.jsonl in the root path"
        
        # Load metadata
        # metadata = {
        #     "video_path": ...,
        #     "hash_code": ...,
        #     "prompt": ...,
        # }
        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append( json.loads(line) )
        
        # Filter metadata based on the optical flow statistics
        metadata = self.filter_metadata(metadata, use_cycle_filtering=self.args.use_cycle_filtering, thres=self.args.filtering_threshold, thres_low=self.args.filtering_threshold_low)
        num_data = len(metadata)
        metadata = random.sample(metadata, self.args.max_scene)
        
        # Just use this as official Wan2.2 setting
        sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=self.state.weight_dtype):
                orig_device = self.components.text_encoder.device
                self.components.text_encoder.to(self.accelerator.device, dtype=self.state.weight_dtype)
                
                sample_neg_prompt_embedding = get_t5_prompt_embeds(
                    self.components.tokenizer,
                    self.components.text_encoder,
                    sample_neg_prompt,
                    device=self.accelerator.device,
                    dtype=self.state.weight_dtype,
                )
                self.components.text_encoder.to(orig_device, dtype=self.state.weight_dtype)
            
        
        prompt_embedding_dir_path = "prompt_embeddings"
        # flow_embedding_dir_path = "flow_latent"
        # if self.args.scale_factor is not None:
        #     assert isinstance(self.args.scale_factor, float), "Use scale factor of float!"
        #     flow_embedding_dir_path = f"flow_latent_{self.args.scale_factor}_WAFT"
        flow_embedding_dir_path = f"flow_latent_WAFT"
        
        prompts = [x["prompt"] for x in metadata]
        prompt_embeddings = [data_root / prompt_embedding_dir_path / (x["hash_code"] + '.safetensors') for x in metadata]
        videos = [data_root / "video_latent"  / (x["hash_code"] + '.safetensors') for x in metadata]
        images = [data_root / "first_frames" / (x["hash_code"] + '.png') for x in metadata]
        flows = [data_root / flow_embedding_dir_path / (x["hash_code"] + '.safetensors') for x in metadata]
        if self.args.use_camparam:
            camparams = [data_root / 'cam_param' / (x["hash_code"] + '.npz') for x in metadata]
            validation_camparams = []
        
        # load prompt embedding
        validation_prompts = []
        validation_prompt_embeddings = []
        validation_video_latents = []
        validation_images = []
        validation_flow_latents = []
        for idx, (prompt, prompt_embedding, video_latent, image, flow_latent) in enumerate(zip(prompts, prompt_embeddings, videos, images, flows)):
            validation_prompts.append(prompt)
            validation_prompt_embeddings.append(self.pad_text_embedding(load_file(prompt_embedding)["prompt_embedding"]).unsqueeze(0))
            validation_video_latents.append(rearrange(load_file(video_latent)["video_latent"], 'f c h w -> c f h w').unsqueeze(0))
            validation_flow_latents.append(rearrange(load_file(flow_latent)["flow_latent"], 'f c h w -> c f h w').unsqueeze(0))
            validation_images.append(image)
            if self.args.use_camparam:
                validation_camparams.append(self.load_camparam_for_validation(camparams[idx]).unsqueeze(0))
            
        
        # Mixed training
        if self.do_mixed_batch_training or self.do_motion_regularization:
            data_root = self.args.large_scale_data_root
            metadata_path = data_root / "metadata.jsonl"
            assert metadata_path.is_file(), "For this dataset type, you need metadata.jsonl or metadata_revised.jsonl or metadata_reformat.jsonl or metadata_flow.jsonl in the root path"
        
            metadata = []
            with open(metadata_path, "r") as f:
                for line in f:
                    metadata.append( json.loads(line) )

            # Filter data
            use_cycle_filtering = True
            filtering_threshold = 90 #TODO. Should be changed!
            filtering_threshold_low = 60
            metadata = self.filter_metadata(metadata, use_cycle_filtering=use_cycle_filtering, thres=filtering_threshold, thres_low=filtering_threshold_low)
            metadata = random.sample(metadata, self.args.max_scene)
            
            
            prompt_embedding_dir_path = "prompt_embeddings"
            # flow_embedding_dir_path = "flow_latent"
            # if self.args.scale_factor is not None:
            #     assert isinstance(self.args.scale_factor, float), "Use scale factor of float!"
            #     flow_embedding_dir_path = f"flow_latent_{self.args.scale_factor}_WAFT"
            flow_embedding_dir_path = f"flow_latent_WAFT"
            
            prompts = [x["prompt"] for x in metadata]
            prompt_embeddings = [data_root / prompt_embedding_dir_path / (x["hash_code"] + '.safetensors') for x in metadata]
            videos = [data_root / "video_latent"  / (x["hash_code"] + '.safetensors') for x in metadata]
            images = [data_root / "first_frames" / (x["hash_code"] + '.png') for x in metadata]
            flows = [data_root / flow_embedding_dir_path / (x["hash_code"] + '.safetensors') for x in metadata]
            if self.args.use_camparam:
                camparams = [data_root / 'cam_param' / (x["hash_code"] + '.npz') for x in metadata]
            
            for idx, (prompt, prompt_embedding, video_latent, image, flow_latent) in enumerate(zip(prompts, prompt_embeddings, videos, images, flows)):
                validation_prompts.append(prompt)
                validation_prompt_embeddings.append(self.pad_text_embedding(load_file(prompt_embedding)["prompt_embedding"]).unsqueeze(0))
                validation_video_latents.append(rearrange(load_file(video_latent)["video_latent"], 'f c h w -> c f h w').unsqueeze(0))
                validation_flow_latents.append(rearrange(load_file(flow_latent)["flow_latent"], 'f c h w -> c f h w').unsqueeze(0))
                validation_images.append(image)
                if self.args.use_camparam:
                    validation_camparams.append(self.load_camparam_for_validation(camparams[idx]).unsqueeze(0))
        
        if self.args.additional_validation_root_path is not None:
            assert not self.do_mixed_batch_training and not self.do_motion_regularization, "Do not use mixed training for the case using additional_validation"
            
            data_root = self.args.additional_validation_root_path
            metadata_path = data_root / "metadata.jsonl"
            assert metadata_path.is_file(), "For this dataset type, you need metadata.jsonl or metadata_revised.jsonl or metadata_reformat.jsonl or metadata_flow.jsonl in the root path"
        
            metadata = []
            with open(metadata_path, "r") as f:
                for line in f:
                    metadata.append( json.loads(line) )

            # Filter data (Do not use.)
            use_cycle_filtering = False
            filtering_threshold = 100
            filtering_threshold_low = 0
            metadata = self.filter_metadata(metadata, use_cycle_filtering=use_cycle_filtering, thres=filtering_threshold, thres_low=filtering_threshold_low)
            metadata = random.sample(metadata, self.args.max_scene) # TODO. remove *2 later.
            
            
            prompt_embedding_dir_path = "prompt_embeddings"
            flow_embedding_dir_path = f"flow_latent_WAFT"
            
            prompts = [x["prompt"] for x in metadata]
            prompt_embeddings = [data_root / prompt_embedding_dir_path / (x["hash_code"] + '.safetensors') for x in metadata]
            videos = [data_root / "video_latent"  / (x["hash_code"] + '.safetensors') for x in metadata]
            images = [data_root / "first_frames" / (x["hash_code"] + '.png') for x in metadata]
            flows = [data_root / flow_embedding_dir_path / (x["hash_code"] + '.safetensors') for x in metadata]
            if self.args.use_camparam:
                camparams = [data_root / 'cam_param' / (x["hash_code"] + '.npz') for x in metadata]
            
            for idx, (prompt, prompt_embedding, video_latent, image, flow_latent) in enumerate(zip(prompts, prompt_embeddings, videos, images, flows)):
                validation_prompts.append(prompt)
                validation_prompt_embeddings.append(self.pad_text_embedding(load_file(prompt_embedding)["prompt_embedding"]).unsqueeze(0))
                validation_video_latents.append(rearrange(load_file(video_latent)["video_latent"], 'f c h w -> c f h w').unsqueeze(0))
                validation_flow_latents.append(rearrange(load_file(flow_latent)["flow_latent"], 'f c h w -> c f h w').unsqueeze(0))
                validation_images.append(image)
                if self.args.use_camparam:
                    validation_camparams.append(self.load_camparam_for_validation(camparams[idx]))
        
        
        validation_videos = [None] * len(validation_prompts)
        validation_neg_prompt_embeddings = [sample_neg_prompt_embedding] * len(validation_prompts)

        self.state.validation_prompts = validation_prompts
        self.state.validation_prompt_embeddings = validation_prompt_embeddings
        self.state.validation_neg_prompt_embeddings = validation_neg_prompt_embeddings
        self.state.validation_images = validation_images
        self.state.validation_videos = validation_videos
        self.state.validation_video_latents = validation_video_latents
        self.state.validation_flow_latents = validation_flow_latents
        if self.args.use_camparam:
            self.state.validation_camparams = validation_camparams
    
    
    def prepare_resume_from_pretrained(self) -> None:
        if self.args.resume_from_pretrained:
            assert self.args.training_type == 'sft', "We do not support resume_from_pretrained for LoRA for now."
            # For SFT
            # TF2V 모델 학습 완료되면 사용하기.
            sd = torch.load(self.args.resume_from_pretrained, map_location='cpu')['module']
            missing, unexpected = self.components.transformer.module.load_state_dict(sd)
            
            assert len(missing) == 0 or len(unexpected) == 0, "Choose wrong checkpoint!"
            
    
    def apply_null_condition(self, text_embedding, plucker_embedding=None):
        """
        For better CFG.
        Apply null condition only for Camera embedding for now.
        """
        
        B = text_embedding.shape[0]
        device = text_embedding.device
        dtype = text_embedding.dtype
        
        assert self.args.uncond_prob_cam >= 0.0 and self.args.uncond_prob_cam <= 1.0, "Use 0 <= uncond_prob_cam <= 1"
        if self.args.uncond_prob_cam > 0.0:
            assert plucker_embedding is not None, "You should use plucker embedding for null cond training"
            
            drop_mask = (torch.rand(B, device=device) < self.args.uncond_prob_cam).float()
            drop_mask = drop_mask.view(B, *([1] * (plucker_embedding.ndim - 1)))
            null_plucker_embedding = torch.zeros_like(plucker_embedding)
            
            plucker_embedding = plucker_embedding * (1 - drop_mask) + null_plucker_embedding * drop_mask
            
        return text_embedding.to(device, dtype), plucker_embedding.to(device, dtype)

        
    
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
    
    
    def train(self) -> None:
        # try:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        

        if resume_from_checkpoint_path is not None:
            try:
                # Ordinary load state
                # 1) load state dict first, such as mp_rank_00_model_states.pt
                # 2) Then, load other weights, such as lora..
                self.accelerator.load_state(resume_from_checkpoint_path)
            except:
                # When error occurs, we sidestep this issue for the case using single or fewer gpus compared to the training time.
                # Just load other weights, such as lora.. (refer to hook func.)
                if "cogvideox-i2vf" in self.args.model_name:
                    print(f"Loading only LoRA weights from {resume_from_checkpoint_path}. For debugging, this is a workaround for the issue with deepspeed.")
                    self.load_only_lora(resume_from_checkpoint_path)
                else:
                    raise NotImplementedError("Only support cogvideox-i2vf for now. For other case, use original load_state func of accelerator.") 
            
            # Adjust learning rates according to the global step
            for _ in range(global_step):
                self.lr_scheduler.scheduler.step()
           
        # Do validation before start
        if self.args.validation_before_training:
            self.validate(global_step)
        # self.__maybe_save_checkpoint(global_step)
        
        
        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        last_validated_step = -1
        if global_step != 0: # For resuming case
            last_validated_step = global_step


        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")
            
            if self.args.training_type == "sft" or self.args.training_type == "lora":
                self.components.transformer.train()
                models_to_accumulate = [self.components.transformer]
            elif self.args.training_type == "controlnet":
                self.components.controlnet.train()
                models_to_accumulate = [self.components.controlnet]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                
                # import torch.distributed as dist
                # if dist.get_rank() == 0:
                #     pdb.set_trace()
                # dist.barrier()
                # if accelerator.is_main_process:
                #     print(f"step-{step} | conv1-{self.components.transformer.camera_embedder.conv_t1.weight.mean()} | conv2-{self.components.transformer.camera_embedder.conv_t2.weight.mean()} | proj-{self.components.transformer.camera_embedder.proj.weight.mean()}")
                

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            if self.components.transformer.__class__.__name__ == "JointSFTCogVideoXTransformer3DModel":
                                param_to_clip = self.components.transformer.cond_branch.parameters()
                            else:
                                param_to_clip = self.components.transformer.parameters()
                            grad_norm = accelerator.clip_grad_norm_(
                                param_to_clip, self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                        # Get into the sync_gradient block?
                        # print(f"[Step {step}] >>> optimizer.step() called (sync_gradients=True)")
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                    else:
                        pass


                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)
                

                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)

                # Maybe run validation
                should_run_validation = (
                    self.args.do_validation and 
                    global_step % self.args.validation_steps == 0 and 
                    global_step != 0 and 
                    global_step != last_validated_step  # prevent duplicate validation
                )
                
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)
                    should_run_validation = False
                    last_validated_step = global_step

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()
        # except Exception as e:
        #     logger.info(f"Error message: {e}")    
    
    
    def mixed_batch_train(self) -> None:
        logger.info("Starting mixed batch training")
        
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset), # Changed
            "train epochs": self.args.train_epochs, # variable will be changed
            "train steps": self.args.train_steps,  # variable will be changed
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader), # Changed
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        
        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)
            
            # Adjust learning rates according to the global step
            for _ in range(global_step):
                self.lr_scheduler.scheduler.step()
            
        # Do validation before start
        if self.args.validation_before_training:
            self.validate(global_step)
        
        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )
        
        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        last_validated_step = -1
        
        data_iter = iter(self.data_loader)
        large_scale_data_iter = iter(self.large_scale_data_loader)
        
        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")
            
            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]
            
            for step in range(self.args.train_steps):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                
                # Draw batches from both datasets
                try:
                    batch_1 = next(data_iter)
                except StopIteration:
                    logger.debug(f"Make New Iterator!")
                    data_iter = iter(self.data_loader)
                    batch_1 = next(data_iter)
                
                try:
                    batch_2 = next(large_scale_data_iter)
                except StopIteration:
                    logger.debug(f"Make New Iterator!")
                    large_scale_data_iter = iter(self.large_scale_data_loader)
                    batch_2 = next(large_scale_data_iter)
                
                # Merge batches to one batch
                batch = {k: torch.cat([batch_1[k], batch_2[k]], dim=0) for k in batch_1.keys()}
                
                with accelerator.accumulate(models_to_accumulate):
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs[f"grad_norm"] = grad_norm

                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
            
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)
                
                logs[f"loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)
                
                # Maybe run validation
                should_run_validation = (
                    self.args.do_validation and 
                    global_step % self.args.validation_steps == 0 and 
                    global_step != 0 and 
                    global_step != last_validated_step  # prevent duplicate validation
                )
                
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)
                    should_run_validation = False
                    last_validated_step = global_step

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break
            
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()
    
    
    def motion_reg_train(self) -> None:
        logger.info("Starting motion reg training")
        
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset), # Changed
            "train epochs": self.args.train_epochs, # variable will be changed
            "train steps": self.args.train_steps,  # variable will be changed
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader), # Changed
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        
        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)
            
            # Adjust learning rates according to the global step
            for _ in range(global_step):
                self.lr_scheduler.scheduler.step()
            
        # Do validation before start
        if self.args.validation_before_training:
            self.validate(global_step)
        
        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )
        
        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        last_validated_step = -1
        
        data_iter = iter(self.data_loader)
        large_scale_data_iter = iter(self.large_scale_data_loader)
        
        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")
            
            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]
            
            for step in range(self.args.train_steps):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                
                
                # Draw batch from each dataset
                if step % 2 == 0:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        logger.debug(f"Make New Iterator!")
                        data_iter = iter(self.data_loader)
                        batch = next(data_iter)
                    use_motion_reg = True
                else:
                    try:
                        batch = next(large_scale_data_iter)
                    except StopIteration:
                        logger.debug(f"Make New Iterator!")
                        large_scale_data_iter = iter(self.large_scale_data_loader)
                        batch = next(large_scale_data_iter)
                    use_motion_reg = False
                
                # print(f"Global step {global_step} | Step {step} | Using {'motion reg' if use_motion_reg else 'large scale data'} batch")
                
                # Merge batches to one batch
                # batch = {k: torch.cat([batch_1[k], batch_2[k]], dim=0) for k in batch_1.keys()}
                
                with accelerator.accumulate(models_to_accumulate):
                    loss = self.compute_loss(batch, use_motion_reg)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs[f"grad_norm"] = grad_norm

                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
            
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)
                
                logs[f"loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)
                
                # Maybe run validation
                should_run_validation = (
                    self.args.do_validation and 
                    global_step % self.args.validation_steps == 0 and 
                    global_step != 0 and 
                    global_step != last_validated_step  # prevent duplicate validation
                )
                
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)
                    should_run_validation = False
                    last_validated_step = global_step

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break
            
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()
    

    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")
        
        USE_FLOW = False
        if 't2f' in self.args.model_name or 'i2f' in self.args.model_name or 'tc2f' in self.args.model_name or 't2fv' in self.args.model_name or 'tf2v' in self.args.model_name:
            # print("Use Optical Flow!")
            USE_FLOW = True

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["transformer"])
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################

        all_processes_artifacts = []
        for i in range(num_validation_samples):
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on all processes but one
                if i % accelerator.num_processes != accelerator.process_index:
                    continue

            prompt = self.state.validation_prompts[i]
            image = self.state.validation_images[i]
            video = self.state.validation_videos[i]
            video_latent = self.state.validation_video_latents[i].permute(0,2,1,3,4) # [B,F,C,H,W] (e.g., [B, 13, 16, 60, 90])
            prompt_embedding = self.state.validation_prompt_embeddings[i]
            neg_prompt_embedding = self.state.validation_neg_prompt_embeddings[i]
            if USE_FLOW:
                flow_latent = self.state.validation_flow_latents[i].permute(0,2,1,3,4) # [B,F,C,H,W] (e.g., [B, 13, 16, 60, 90])
            
            plucker_embedding = None
            if self.args.use_camparam:
                _, _, _, height, width = video_latent.shape
                camparam = self.state.validation_camparams[i].to(self.accelerator.device) # [B,T,25]
                plucker_resize_scale = 16
                plucker_embedding = self.plucker_embedding_from_camparam(camparam, H=height*plucker_resize_scale, W=width*plucker_resize_scale, device=self.accelerator.device)
                plucker_embedding = plucker_embedding.to(dtype=self.state.weight_dtype)

            if image is not None:
                image = preprocess_image_with_resize(image, self.state.train_height, self.state.train_width)
                # Convert image tensor (C, H, W) to PIL images
                image = image.to(torch.uint8)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)

            if video is not None:
                video = preprocess_video_with_resize(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width
                )
                # Convert video tensor (F, C, H, W) to list of PIL images
                video = video.round().clamp(0, 255).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True, dtype=self.state.weight_dtype):
                        video_decoded = self.decode_latents(video_latent.to(self.accelerator.device)) # BFCHW
                        if USE_FLOW:
                            flow_decoded = self.decode_flow_latents(flow_latent.to(self.accelerator.device), sigma=0.1) # scale_factor for consecutive optical flow
                    video = ((video_decoded + 1.) / 2. * 255.)[0].float().clip(0., 255.).to(torch.uint8)
                    video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
                        

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            validation_artifacts = self.validation_step({"prompt": prompt, 'prompt_embedding': prompt_embedding.to(self.accelerator.device), 'neg_prompt_embedding': neg_prompt_embedding.to(self.accelerator.device), "image": image, "video": video, "flow_latent": flow_latent.to(self.accelerator.device) if USE_FLOW else None, "plucker_embedding": plucker_embedding if plucker_embedding is not None else None}, pipe)

            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue

            prompt_filename = string_to_filename(prompt)[:25]
            # Calculate hash of reversed prompt as a unique identifier
            reversed_prompt = prompt[::-1]
            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]

            artifacts = {
                "image": {"type": "image", "value": image},
                "video": {"type": "video", "value": video},
            }
            
            if USE_FLOW:
                artifacts.update({f"artifact_flow_{i}": {"type": 'flow', "value": flow_decoded}})
            
            
            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts.update({f"artifact_{i}": {"type": artifact_type, "value": artifact_value}})
                
                # if artifact_type == 'synthesized_flow':
                #     image_tensor = rearrange(torch.tensor(np.array(image)).to(flow_decoded.device, torch.float), 'h w c -> 1 c h w') # scale~(0,255) (B=1) C H W
                #     warped_video2 = generate_video_from_image_flow(image_tensor, rearrange(artifact_value.to(torch.float), '(b f) c h w -> b f c h w', f=artifact_value.size(0))) # BFCHW
                #     warped_video2 = rearrange(warped_video2, 'b f c h w -> (b f) c h w') # B=1
                #     frame_list2 = []
                #     for frame in warped_video2:
                #         frame = (frame.permute(1,2,0).float().detach().cpu().numpy()).astype(np.uint8).clip(0,255)
                #         frame_list2.append(Image.fromarray(frame))
                    
                #     artifacts.update({f"artifact_synthesized_flow_warped_video_{i}": {"type": 'synthesized_flow_warped_video', "value": frame_list2}})
                    
            
                
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video", "flow", "synthesized_video", "synthesized_flow"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}_{artifact_type}.{extension}"
                validation_path = self.args.output_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = str(validation_path / filename)

                if artifact_type == "image":
                    logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                    artifact_value = wandb.Image(filename)
                elif artifact_type == "video" or artifact_type == "synthesized_video":
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                    artifact_value = wandb.Video(filename, caption=f"[{artifact_type}]--{prompt}")
                elif artifact_type == "synthesized_flow" or artifact_type == "flow":
                    # TODO. RGB Visualization of optical flow. (F,2,H,W)
                    artifact_value_RGB = flow_to_color(artifact_value) # BF,C,H,W (B=1)

                    frame_list = []
                    for frame in artifact_value_RGB:
                        frame = (frame.permute(1,2,0).float().detach().cpu().numpy()).astype(np.uint8).clip(0,255)
                        frame_list.append(Image.fromarray(frame))
                    
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(frame_list, filename, fps=self.args.gen_fps)
                    artifact_value = wandb.Video(filename, caption=f"[{artifact_type}]--{prompt}")

                all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
                    video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
                    tracker.log(
                        {
                            tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                        },
                        step=step,
                    )

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()
    
    
    def pad_text_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args.
            embedding: (seq_len, C=4096)
        Output.
            embedding_padded: (seq_len_total, C=4096)
        """
        assert embedding.ndim == 2, embedding.shape[-1] == 4096
        
        return torch.cat([embedding, embedding.new_zeros(self.args.text_embedding_len - embedding.size(0), embedding.size(1))])
        
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args.
            latents: BFCHW
        Return.
            video: video decoded from latents, BFCHW
        """
        
        latents_mean = (
            torch.tensor(self.components.vae.config.latents_mean)
            .view(1, self.components.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        ) # BCFHW
        latents_std = 1.0 / torch.tensor(self.components.vae.config.latents_std).view(1, self.components.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        ) # BCFHW
        latents = rearrange(latents, 'b f c h w -> b c f h w') / latents_std + latents_mean
        # torch.Size([1, 48, 31, 44, 80])
        video = self.components.vae.decode(latents, return_dict=False)[0]
        
        return rearrange(video, 'b c f h w -> b f c h w')


    def decode_flow_latents(self, latents: torch.Tensor, sigma=0.1) -> torch.Tensor:
        """
        Args.
            latents: BFCHW
        Return.
            video: video decoded from latents, (BF)CHW
        """
        
        latents_mean = (
            torch.tensor(self.components.vae.config.latents_mean)
            .view(1, self.components.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        ) # BCFHW
        latents_std = 1.0 / torch.tensor(self.components.vae.config.latents_std).view(1, self.components.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        ) # BCFHW
        latents = rearrange(latents, 'b f c h w -> b c f h w') / latents_std + latents_mean
        flow_RGB_normalized = self.components.vae.decode(latents, return_dict=False)[0]
        flow_RGB_normalized = rearrange(flow_RGB_normalized, 'b c f h w -> (b f) c h w')
        
        if self.args.scale_factor is not None:
            flow = rgb_to_flow_ours_inverse(flow_RGB_normalized, scale = self.args.scale_factor)
        else:
            flow = rgb_hsv_flow_inverse(flow_RGB_normalized, sigma=sigma) # (BF)CHW
        
        return flow
        
        
    def compute_density_for_timestep_sampling(
        self,
        weighting_scheme: str,
        batch_size: int,
        logit_mean: float = None,
        logit_std: float = None,
        mode_scale: float = None,
        device = "cpu",
        generator = None,
    ):
        if weighting_scheme == "logit_normal":
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device=device, generator=generator)
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device=device, generator=generator)
        return u
    
    def compute_loss_weighting(self, weighting_scheme: str, sigmas=None):
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        elif weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
        else:
            weighting = torch.ones_like(sigmas)
        return weighting
    
    def get_sigmas(self, noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def get_training_weights(self, timesteps: torch.Tensor, num_inference_steps: int = 1000) -> torch.Tensor:
        """
        Compute per-timestep training weights like DiffSynth.

        Args:
            timesteps: [B] tensor of timesteps chosen during training
            num_inference_steps: number of total steps (same as scheduler.config.num_train_timesteps)

        Returns:
            weights: [B] tensor with per-sample weights
        """
        # timesteps are real-valued → normalize to [0, num_inference_steps]
        t = timesteps.to(dtype=torch.float32, device=self.accelerator.device)

        # Gaussian bump centered in the middle of the schedule
        y = torch.exp(-2.0 * ((t - num_inference_steps / 2) / num_inference_steps) ** 2)

        # shift so min=0, then normalize sum to num_inference_steps
        y_shifted = y - y.min()
        weights = y_shifted * (num_inference_steps / y_shifted.sum())

        return weights
    

    def load_components(self) -> Components:
        raise NotImplementedError

    def initialize_pipeline(self) -> DiffusionPipeline:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        # shape of output video: [B, C', F', H', W'], where B = 1
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        # shape of output text: [batch size, sequence length, embedding dimension]
        raise NotImplementedError

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        raise NotImplementedError
    
    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")
    
    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        # Set Lora weight
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                        
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                # Save Lora weight
                self.components.pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}")
            else:
                transformer_ = unwrap_model(self.accelerator, self.components.transformer).__class__.from_pretrained(
                    str(self.args.model_path), subfolder="transformer"
                )
                transformer_.add_adapter(transformer_lora_config)

            # Load Lora weight
            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)
    
    def filter_metadata(self, metadata: List[Dict[str, Any]], use_cycle_filtering=False, thres=85, thres_low=0) -> List[Dict[str, Any]]:
        """
        Filter metadata based on flow statistics.
        Keeps items whose flow_cycle_discrepancy is between the percentile thresholds [thres_low, thres].
        """
        assert 0 <= thres <= 100, "Threshold must be between 0 and 100"
        assert 0 <= thres_low <= 100, "Threshold_low must be between 0 and 100"
        assert thres_low < thres, "thres should be larger than thres_low"

        if use_cycle_filtering:
            cycle_error = np.array([line["flow_cycle_discrepancy"] for line in metadata])
            
            # High & Low percentile thresholds
            cycle_error_thres_high = np.percentile(cycle_error, thres)
            cycle_error_thres_low  = np.percentile(cycle_error, thres_low)

            # Keep items in range [low, high]
            metadata = [
                line for line in metadata 
                if cycle_error_thres_low <= line["flow_cycle_discrepancy"] <= cycle_error_thres_high
            ]

        return metadata

    
    def plucker_embedding_from_camparam(self, camparam, H, W, device, flip_flag=None):
        # camparam: B, V, 25
        B, V, _ = camparam.shape
        
        K_norm = camparam[:,:,:9].view(B,-1,3,3)        # (B,T,3,3), Normalized K
        E = camparam[:,:,9:].view(B,-1,4,4)        # (B,T,4,4), W2C
        
        intrinsics = K_norm.detach().clone()
        intrinsics[:,:,0:1] = intrinsics[:,:,0:1] * (W)
        intrinsics[:,:,1:2] = intrinsics[:,:,1:2] * (H)
        
        c2w = torch.inverse(E)
        
        # B, V, 1
        fx = intrinsics[:,:,0:1,0]
        fy = intrinsics[:,:,1:2,1]
        cx = intrinsics[:,:,0:1,2]
        cy = intrinsics[:,:,1:2,2]

        j, i = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
        )
        i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
        j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

        n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
        if n_flip > 0:
            j_flip, i_flip = custom_meshgrid(
                torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
                torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
            )
            i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
            j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
            i[:, flip_flag, ...] = i_flip
            j[:, flip_flag, ...] = j_flip

        # fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

        zs = torch.ones_like(i)                 # [B, V, HxW]
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)

        directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
        directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

        rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
        rays_o = c2w[..., :3, 3]                                        # B, V, 3
        rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
        # c2w @ dirctions
        rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
        plucker = torch.cat([rays_dxo, rays_d], dim=-1)
        plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
        plucker = plucker.permute(0, 4, 1, 2, 3)                        # B, 6, V, H, W
        
        return plucker
    

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union, Optional
import ftfy
import html
import re

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

def get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

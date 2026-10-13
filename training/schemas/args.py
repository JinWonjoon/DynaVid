import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, List, Literal, Tuple

from pydantic import BaseModel, ValidationInfo, field_validator


class Args(BaseModel):
    ########## Model ##########
    model_path: Path
    model_name: str
    model_type: Literal["t2v"]
    training_type: Literal["lora", "sft"] = "sft"
    output_dir: Path = Path("train_results/{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now()))
    report_to: Literal["tensorboard", "wandb", "all"] | None = None
    tracker_name: str = "finetrainer-cogvideo"
    run_name: str = "CogVideoX"
    train_resolution: Tuple[int, int, int]  # shape: (frames, height, width)
    do_mixed_batch_training: bool = False
    do_motion_regularization: bool = False
    resume_from_pretrained: Path | None = None
    text_embedding_len: int = 512
    FreezeExceptCamEmb: bool = False
    shift_training: float = 5.0
    noise_sigma: float = 0.0 # noise for TF2V
    train_attn1_only: bool = False
    train_flow_only: bool = False
    train_flow_and_attn1: bool = False
    use_separate_rope_emb: bool = False
    use_separate_modality_emb: bool = False
    use_multi_modal_cross_attn: bool = False
    use_gate: bool = False
    mm_mode: Literal["base", "pre_sa", "post_sa", "parallel"] = "base"
    use_separate_projection_attn_joint: bool = False  # For joint attn projection
    use_separate_projection: bool = False  # For projection
    sep_LN: bool = True
    sep_FFN: bool = True
    # sep_Attn: bool = False
    joint_attn_type: Literal["cross", "self", "none"] = "cross"
    FreezeVideoBranch: bool = False
    
    rebuttal_sample_100: bool = False
    rebuttal_sample_200: bool = False
    
    ########## JointGenerator ##########
    AddAttn: bool = False
    PretrainedFlowGen: bool = False
    FlowCondRGBGen: bool = False
    resume_from_pretrained: Path | None = None
    MotionGeneratorCkpt: Path | None = None
    
    ########## Data ###########
    data_root: Path
    use_cycle_filtering: bool = False
    filtering_threshold: int = 85
    filtering_threshold_low: int = 0
    large_scale_data_root: Path | None = None
    scale_factor: float = 29.5
    use_camparam: bool = False
    
    
    
    ########## Training #########
    resume_from_checkpoint: Path | None = None
    seed: int | None = None
    train_epochs: int
    train_steps: int | None = None
    checkpointing_steps: int = 200
    checkpointing_limit: int = 10

    batch_size: int
    gradient_accumulation_steps: int = 1
    uncond_prob_cam: float = 0.0
    guidance_scale_cam: float = 1.0
    
    mixed_precision: Literal["no", "fp16", "bf16"]

    learning_rate: float = 1e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    
    num_workers: int = 8
    pin_memory: bool = True

    gradient_checkpointing: bool = True
    enable_slicing: bool = True
    enable_tiling: bool = True
    nccl_timeout: int = 1800
    
    
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    
    ########## Validation ##########
    do_validation: bool = False
    validation_before_training: bool = False
    validation_steps: int | None  # if set, should be a multiple of checkpointing_steps
    validation_dir: Path | None  # if set do_validation, should not be None
    gen_fps: int = 15
    max_scene: int = 8
    num_inference_steps: int = 30
    additional_validation_root_path: Path | None

    
    
    @classmethod
    def parse_args(cls):
        """Parse command line arguments and return Args instance"""
        parser = argparse.ArgumentParser()

        # Required arguments
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--training_type", type=str, required=True)
        parser.add_argument("--do_mixed_batch_training", type=bool, required=False, default=False)
        parser.add_argument("--report_to", type=str, required=True)
        parser.add_argument("--run_name", type=str, required=False, default='CogVideoX')
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--train_resolution", type=str, required=True)
        parser.add_argument("--text_embedding_len", type=int, required=False, default=512)
        parser.add_argument("--resume_from_pretrained", type=str, default=None)
        parser.add_argument("--FreezeExceptCamEmb", type=bool, required=False, default=False)
        parser.add_argument("--shift_training", type=float, default=5.0)
        parser.add_argument("--noise_sigma", type=float, default=0.0)
        parser.add_argument("--do_motion_regularization", type=bool, required=False, default=False)
        parser.add_argument("--train_attn1_only", type=bool, required=False, default=False)
        parser.add_argument("--train_flow_only", type=bool, required=False, default=False)
        parser.add_argument("--train_flow_and_attn1", type=bool, required=False, default=False)
        parser.add_argument("--use_separate_rope_emb", type=bool, required=False, default=False)
        parser.add_argument("--use_separate_modality_emb", type=bool, required=False, default=False)
        parser.add_argument("--use_multi_modal_cross_attn", type=bool, required=False, default=False)
        parser.add_argument("--use_gate", type=bool, required=False, default=False)
        parser.add_argument("--mm_mode", type=str, required=False, default='base')
        parser.add_argument("--use_separate_projection_attn_joint", type=bool, required=False, default=False)
        parser.add_argument("--use_separate_projection", type=bool, required=False, default=False)
        parser.add_argument("--sep_LN", type=bool, required=False, default=True)
        parser.add_argument("--sep_FFN", type=bool, required=False, default=True)
        # parser.add_argument("--sep_Attn", type=bool, required=False, default=False)
        parser.add_argument("--joint_attn_type", type=str, required=False, default="cross")
        parser.add_argument("--FreezeVideoBranch", type=bool, required=False, default=False)
        
        parser.add_argument("--rebuttal_sample_100", type=bool, required=False, default=False)
        parser.add_argument("--rebuttal_sample_200", type=bool, required=False, default=False)

        # Joint Generator arguments
        parser.add_argument("--AddAttn", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--PretrainedFlowGen", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--FlowCondRGBGen", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--MotionGeneratorCkpt", type=str, default=None)
        
        # Training hyperparameters
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--train_epochs", type=int, default=10)
        parser.add_argument("--train_steps", type=int, default=None)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.95)
        parser.add_argument("--beta3", type=float, default=0.98)
        parser.add_argument("--epsilon", type=float, default=1e-8)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--uncond_prob_cam", type=float, default=0.0)
        parser.add_argument("--guidance_scale_cam", type=float, default=1.0)
        
        
        # Learning rate scheduler
        parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
        parser.add_argument("--lr_warmup_steps", type=int, default=100)
        parser.add_argument("--lr_num_cycles", type=int, default=1)
        parser.add_argument("--lr_power", type=float, default=1.0)
        
        # Data
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--pin_memory", type=bool, default=True)
        parser.add_argument("--data_root", type=str, required=True)
        parser.add_argument("--use_cycle_filtering", type=bool, default=False)
        parser.add_argument("--filtering_threshold", type=int, default=85)
        parser.add_argument("--filtering_threshold_low", type=int, default=0)
        parser.add_argument("--large_scale_data_root", type=str, default=None)
        parser.add_argument("--scale_factor", type=float, default=29.5)
        parser.add_argument("--use_camparam", type=bool, default=False)
        
        
        # Model configuration
        parser.add_argument("--mixed_precision", type=str, default="no")
        parser.add_argument("--gradient_checkpointing", type=bool, default=True)
        parser.add_argument("--enable_slicing", type=bool, default=True)
        parser.add_argument("--enable_tiling", type=bool, default=True)
        parser.add_argument("--nccl_timeout", type=int, default=1800)
        
        
        # Checkpointing
        parser.add_argument("--checkpointing_steps", type=int, default=200)
        parser.add_argument("--checkpointing_limit", type=int, default=10)
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)
        
        
        # Validation
        parser.add_argument("--do_validation", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--validation_before_training", type=bool, default=False)
        parser.add_argument("--validation_steps", type=int, default=None)
        parser.add_argument("--validation_dir", type=str, default=None)
        parser.add_argument("--gen_fps", type=int, default=15)
        parser.add_argument("--max_scene", type=int, default=8)
        parser.add_argument("--num_inference_steps", type=int, default=30)
        parser.add_argument("--additional_validation_root_path", type=str, default=None)
        args = parser.parse_args()

        # Convert video_resolution_buckets string to list of tuples
        frames, height, width = args.train_resolution.split("x")
        args.train_resolution = (int(frames), int(height), int(width))
        
            
        return cls(**vars(args))

#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set WANDB_API_KEY in your environment before running this script.

# Model Configuration
MODEL_ARGS=(
    --model_path "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    --model_name "wan-tc2f-vace"  # ["cogvideox-i2v" or "cogvideox-flovd"]
    --model_type "t2v" # ["t2v", "i2v", "i2vFlow", "if2v"]
    --training_type "sft"
    --MotionGeneratorCkpt "/workspace/logs/TC2F_VACE_RE10K_Train/DynaVid_TC2F_VACE_10Layers_RE10K_Train_BS16_lr1e-5_shift15-2025-10-21T08-05-03/checkpoint-3448/pytorch_model/mp_rank_00_model_states.pt"
    --shift_training 15
    --do_mixed_batch_training true
    # --FreezeExceptCamEmb true
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/local_home/workspace/DynaVid/output/TC2F_VACE_Synthetic_Final_Fixed"
    --report_to "wandb"
    --run_name "DynaVid_TC2F_VACE_10Layers_Synthetic_Final_BS16_lr1e-5_shift15_Fixed"  # Change run name as needed
)

# Data Configuration
DATA_ARGS=(
    --data_root "/workspace/data/Wan/DynaVid_CameraData_Final/"
    --train_resolution "121x704x1280"  # (frames x height x width), frames should be 8N+1
    --use_camparam true
    # --use_cycle_filtering True
    # --filtering_threshold 90
    --scale_factor 29.5
    --large_scale_data_root "/workspace/data/Wan/RE10K_Train/"
)


# Training Configuration
TRAIN_ARGS=(
    --train_epochs 40 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 2
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
    --learning_rate 1e-5
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 1000 # save checkpoint every x steps
    --checkpointing_limit 3 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/workspace_4T/VelDiff/output/JointGenerator_dev_Data200K_DEBUG/VelDiff_I2VF_jointgenerator_sft_Data200K_batch16_BothFilt_Thres50_DEBUG-2025-07-27T10-41-00/checkpoint-10/"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_before_training true
    --validation_dir "/workspace/data/Wan/DynaVid_CameraData_Final/"
    --validation_steps 1000  # should be multiple of checkpointing_steps
    --gen_fps 24
    --max_scene 4
    --num_inference_steps 30
)



# Combine all arguments and launch training
accelerate launch --config_file training/accelerate_config_mixed.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
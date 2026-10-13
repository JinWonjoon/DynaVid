#!/bin/bash

MODEL_NAME="wan-t2v"
CKPT_PATH_MOTION="/workspace/logs/T2FV_Shutterstock100K_Final_MotionReg/DynaVid_T2FV_Shutterstock100K_BS16_lr1e-5_shift5_Final_MotionReg-2025-10-27T10-57-02/checkpoint-6374/pytorch_model/mp_rank_00_model_states.pt"


# Pexels

DATA_ROOT="/workspace/data/Wan/Pexels_Eval/"
OUTPUT_PATH="./results/T2FV_MotionReg_timesteps/Pexels_Eval/"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/T2FV_timesteps.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 30 \
    --max_num_videos 16 \


# Mixamo

DATA_ROOT="/workspace/data/Wan/Mixamo_Eval_Final/"
OUTPUT_PATH="./results/T2FV_MotionReg_timesteps/Mixamo_Eval_Final/"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/T2FV_timesteps.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 30 \
    --max_num_videos 16 \

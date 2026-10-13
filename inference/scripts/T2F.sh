#!/bin/bash

MODEL_NAME="wan-t2v"
CKPT_PATH_MOTION="/workspace/workspace/checkpoints/DynaVid/T2F_Mixamo_Shutterstock100K_Final/mp_rank_00_model_states.pt"

DATA_ROOT="/workspace/data/Wan/Pexels_Eval/"
OUTPUT_PATH="./results/T2F/Pexels_Eval/"

N_PROCS=1 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/T2F.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \

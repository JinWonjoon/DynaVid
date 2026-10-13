#!/bin/bash

MODEL_NAME="wan-t2v"
CKPT_PATH_MOTION="/workspace/logs/T2FV_intermediate/checkpoint-4000/pytorch_model/mp_rank_00_model_states.pt"

DATA_ROOT="/workspace/data/Wan/Mixamo_Eval_Final/"
OUTPUT_PATH="./results/T2FV_MotionReg_intermediate_GuideEnd0.4/Mixamo_Eval_Final/"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/T2FV.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \

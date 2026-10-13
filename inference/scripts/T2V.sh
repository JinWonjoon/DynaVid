#!/bin/bash

# MODEL_NAME="wan-t2v"

# DATA_ROOT="/workspace/data/Wan/Pexels_Eval/"
# OUTPUT_PATH="./results/Wan-T2V/Pexels_Eval/"

# N_PROCS=4 # the number of GPUs
# RANDOM_PORT=$((49152 + RANDOM % 16384))

# python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Wan_T2V.py \
#     --model_name $MODEL_NAME \
#     --data_root $DATA_ROOT \
#     --output_path $OUTPUT_PATH \
#     --num_inference_steps 50 \



# DATA_ROOT="/workspace/data/Wan/Mixamo_Eval_Final/"
# OUTPUT_PATH="./results/Wan-T2V/Mixamo_Eval_Final/"

# N_PROCS=4 # the number of GPUs
# RANDOM_PORT=$((49152 + RANDOM % 16384))

# python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Wan_T2V.py \
#     --model_name $MODEL_NAME \
#     --data_root $DATA_ROOT \
#     --output_path $OUTPUT_PATH \
#     --num_inference_steps 50 \


MODEL_NAME="wan-t2v"

CKPT_PATH="./output/T2V_Shutterstock100K_filtering60-90/DynaVid_T2V_Shutterstock100K_filtering60-90-2025-11-16T07-17-51/checkpoint-1629/pytorch_model/mp_rank_00_model_states.pt"

DATA_ROOT="/workspace/data/Wan/Pexels_Eval/"
OUTPUT_PATH="./results/Wan-T2V_Shutterstock100K_filtering60-90/Pexels_Eval/"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Wan_T2V.py \
    --model_name $MODEL_NAME \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
    --ckpt_path $CKPT_PATH \



DATA_ROOT="/workspace/data/Wan/Mixamo_Eval_Final/"
OUTPUT_PATH="./results/Wan-T2V_Shutterstock100K_filtering60-90/Mixamo_Eval_Final/"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Wan_T2V.py \
    --model_name $MODEL_NAME \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
    --ckpt_path $CKPT_PATH \
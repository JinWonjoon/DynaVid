#!/bin/bash

MODEL_NAME="wan-t2v"

DATA_ROOT="/workspace/data/Wan/Pexels_Eval/"
OUTPUT_PATH="./results/Wan-T2V_Mixamo_MixedBatch_Final/Pexels_Eval/"
CKPT_PATH="./output/Wan_T2V_MixedBatch_Mixamo1K_Shutterstock100K_Final/Wan_T2V_MixedBatch_Mixamo1K_Shutterstock100K_Final-2025-10-21T08-36-27/checkpoint-4949/pytorch_model/mp_rank_00_model_states.pt"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Wan_T2V.py \
    --model_name $MODEL_NAME \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
    --ckpt_path $CKPT_PATH \


N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

DATA_ROOT="/workspace/data/Wan/Mixamo_Eval_Final/"
OUTPUT_PATH="./results/Wan-T2V_Mixamo_MixedBatch_Final/Mixamo_Eval_Final/"
CKPT_PATH="./output/Wan_T2V_MixedBatch_Mixamo1K_Shutterstock100K_Final/Wan_T2V_MixedBatch_Mixamo1K_Shutterstock100K_Final-2025-10-21T08-36-27/checkpoint-4949/pytorch_model/mp_rank_00_model_states.pt"

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Wan_T2V.py \
    --model_name $MODEL_NAME \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
    --ckpt_path $CKPT_PATH \
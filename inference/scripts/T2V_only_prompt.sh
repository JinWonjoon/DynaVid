#!/bin/bash

MODEL_NAME="wan-t2v"

METADATA_PATH="/workspace/data/Wan/DynaVid_Analysis_Prompts/metadata_animals4.jsonl"
OUTPUT_PATH="./results/Wan-T2V/OnlyPrompt/"

N_PROCS=1 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Wan_T2V_only_prompt.py \
    --model_name $MODEL_NAME \
    --metadata_path $METADATA_PATH \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \

#!/bin/bash

MODEL_NAME="wan-t2v"
CKPT_PATH_MOTION="/workspace/workspace/checkpoints/DynaVid/T2F_Mixamo_Shutterstock100K_Final/mp_rank_00_model_states.pt" # TODO. Will be changed!!
CKPT_PATH_VIDEO="/workspace/workspace/checkpoints/DynaVid/TF2V_VACE_6Layers_Shutterstock100K_thres90_4887step/mp_rank_00_model_states.pt"

# DATA_ROOT="/workspace/data/Wan/Pexels_Eval/"
# OUTPUT_PATH="./results/Sequential_Generation_thres90_guidance0.4/Pexels_Eval/"

NOISE_SIGMA=0.0

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))


METADATA_PATH="/workspace/data/Wan/DynaVid_Analysis_Prompts/metadata_supple.jsonl"
OUTPUT_PATH="./results/Analysis_Sequential_Generation_onlyPrompt/Supple_1118_NoNegFlow/"

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Sequential_gen_onlyPrompt.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --ckpt_path_video $CKPT_PATH_VIDEO \
    --metadata_path $METADATA_PATH \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
    --noise_sigma $NOISE_SIGMA \
    --guidance_end 0.4


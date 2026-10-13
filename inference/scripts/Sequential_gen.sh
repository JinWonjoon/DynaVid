#!/bin/bash

MODEL_NAME="wan-t2v"
CKPT_PATH_MOTION="/workspace/workspace/checkpoints/DynaVid/T2F_Mixamo_Shutterstock100K_Final/mp_rank_00_model_states.pt" # TODO. Will be changed!!
CKPT_PATH_VIDEO="/workspace/workspace/checkpoints/DynaVid/TF2V_VACE_6Layers_Shutterstock100K_thres90_4887step/mp_rank_00_model_states.pt"

# DATA_ROOT="/workspace/data/Wan/Pexels_Eval/"
# OUTPUT_PATH="./results/Sequential_Generation_thres90_guidance0.4/Pexels_Eval/"

NOISE_SIGMA=0.0

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))


DATA_ROOT="/workspace/data/Wan/Pexels_Eval/"
OUTPUT_PATH="./results/Sequential_Generation_VACE_6Layers_thres90_guidance0.2/Pexels_Eval/"

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Sequential_gen.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --ckpt_path_video $CKPT_PATH_VIDEO \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
    --noise_sigma $NOISE_SIGMA \
    --guidance_end 0.2


DATA_ROOT="/workspace/data/Wan/Mixamo_Eval_Final/"
OUTPUT_PATH="./results/Sequential_Generation_VACE_6Layers_thres90_guidance0.2/Mixamo_Eval_Final/"

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Sequential_gen.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --ckpt_path_video $CKPT_PATH_VIDEO \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
    --noise_sigma $NOISE_SIGMA \
    --guidance_end 0.2


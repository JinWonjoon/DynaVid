#!/bin/bash

MODEL_NAME="wan-t2v"
CKPT_PATH_MOTION="/workspace/workspace/checkpoints/DynaVid/T2F_Mixamo_Shutterstock100K_Final/mp_rank_00_model_states.pt" # TODO. Will be changed!!
CKPT_PATH_VIDEO="./output/TF2V_VACE_10Layers_Shutterstock100K_thres90/DynaVid_TF2V_VACE_10Layers_Shutterstock100K_thres90_BS16_noise_sigma0.0-2025-11-01T05-58-21/checkpoint-4887/pytorch_model/mp_rank_00_model_states.pt"

NOISE_SIGMA=0.0 
DATA_ROOT="/workspace/data/Wan/Pexels_Eval/"
OUTPUT_PATH="./results/Sequential_Generation_thres90_noise0.0/Pexels_Eval/"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Sequential_gen.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --ckpt_path_video $CKPT_PATH_VIDEO \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
    --noise_sigma $NOISE_SIGMA \


DATA_ROOT="/workspace/data/Wan/Mixamo_Eval_Final/"
OUTPUT_PATH="./results/Sequential_Generation_thres90_noise0.0/Mixamo_Eval_Final/"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Sequential_gen.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --ckpt_path_video $CKPT_PATH_VIDEO \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
    --noise_sigma $NOISE_SIGMA \
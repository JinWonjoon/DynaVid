#!/bin/bash

MODEL_NAME="wan-t2v"
CKPT_PATH_MOTION="${CKPT_PATH_MOTION:?Set CKPT_PATH_MOTION to your motion checkpoint}"
CKPT_PATH_VIDEO="/workspace/workspace/checkpoints/DynaVid/TF2V_VACE_6Layers_Shutterstock100K_thres90_4887step/mp_rank_00_model_states.pt"

DATA_ROOT="/workspace/data/Wan/RE10K_Test/"
OUTPUT_PATH="./results/Sequential_Gen_camera_Final_Fixed_1110/RE10K_Test/"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Sequential_gen_camera.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --ckpt_path_video $CKPT_PATH_VIDEO \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \


DATA_ROOT="/workspace/data/Wan/DynaVid_CameraData_Final/"
OUTPUT_PATH="./results/Sequential_Gen_camera_Final_1119_Supple/DynaVid_CameraData_Final/"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/Sequential_gen_camera.py \
    --model_name $MODEL_NAME \
    --ckpt_path_motion $CKPT_PATH_MOTION \
    --ckpt_path_video $CKPT_PATH_VIDEO \
    --data_root $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
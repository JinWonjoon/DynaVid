#!/bin/bash

MODEL_NAME="wan-tc2v-vace"

# DATA_ROOT="/workspace/data/Wan/RE10K_Test/"
# CKPT_PATH="output/TC2V_VACE_RE10K_TEST/DynaVid_TC2V_VACE_10Layers_RE10K_TEST_BS8_lr1e-4_shift15-2025-10-09T12-03-16/checkpoint-8000/pytorch_model/mp_rank_00_model_states.pt"
# OUTPUT_PATH="./results/TC2V_shift15_InferShift15_guidance_end_0.4_step8K/RE10K_Test/"

# NUM_VIDEO=16


DATA_ROOT="/workspace/data/Wan/DynaVid_CameraData_Final/"
CKPT_PATH="/workspace/logs/TC2V_VACE_RE10K_Train/DynaVid_TC2V_VACE_10Layers_RE10K_Train_BS16_lr1e-5_shift15-2025-10-20T07-30-35/checkpoint-3448/pytorch_model/mp_rank_00_model_states.pt"
OUTPUT_PATH="./results/TC2V_Shift15_InferShift15/DynaVid_CameraData_Final/"

N_PROCS=4 # the number of GPUs
RANDOM_PORT=$((49152 + RANDOM % 16384))

# python inference/flovd_FVSM_demo.py \
python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/TC2V.py \
    --model_name $MODEL_NAME \
    --data_root $DATA_ROOT \
    --ckpt_path $CKPT_PATH \
    --output_path $OUTPUT_PATH \
    --num_inference_steps 50 \
    --guidance_end 0.1 \
    --infer_shift 15.0 \


# DATA_ROOT="/workspace/data/Wan/RE10K_Test/"
# CKPT_PATH="/workspace/logs/TC2V_VACE_RE10K_Train/DynaVid_TC2V_VACE_10Layers_RE10K_Train_BS16_lr1e-5_shift15-2025-10-20T07-30-35/checkpoint-3448/pytorch_model/mp_rank_00_model_states.pt"
# OUTPUT_PATH="./results/TC2V_Shift15_InferShift15/RE10K_Test/"

# N_PROCS=4 # the number of GPUs
# RANDOM_PORT=$((49152 + RANDOM % 16384))

# # python inference/flovd_FVSM_demo.py \
# python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference/TC2V.py \
#     --model_name $MODEL_NAME \
#     --data_root $DATA_ROOT \
#     --ckpt_path $CKPT_PATH \
#     --output_path $OUTPUT_PATH \
#     --num_inference_steps 50 \
#     --guidance_end 0.1 \
#     --infer_shift 15.0 \

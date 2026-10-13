# DATA_ROOT_PATH="/workspace_4T/DynaVid/results/T2FV_MotionReg_timesteps/Pexels_Eval"
# SAVE_PATH="./eval_results/Pexels_eval.csv"

# python compute_reward_flow_video_re.py \
#     --root_path ${DATA_ROOT_PATH} \
#     --save_path ${SAVE_PATH} \
#     --mag_thresh 2.0


DATA_ROOT_PATH="/workspace_4T/DynaVid/results/T2FV_MotionReg_timesteps/Mixamo_Eval_Final"
SAVE_PATH="./eval_results/Mixamo_Eval_Final.csv"

python compute_reward_flow_video_re.py \
    --root_path ${DATA_ROOT_PATH} \
    --save_path ${SAVE_PATH} \
    --mag_thresh 2.0
DATA_ROOT_PATH="/workspace_4T/DynaVid/results/T2FV_MotionReg_timesteps/Pexels_Eval"
SAVE_PATH="./eval_results/Gen_Scene/Pexels_eval/SSIM.csv"

python compute_reward_flow_video_gen_scene_ssim.py \
    --root_path ${DATA_ROOT_PATH} \
    --save_path ${SAVE_PATH} \
    --mag_thresh 2.0


DATA_ROOT_PATH="/workspace_4T/DynaVid/results/T2FV_MotionReg_timesteps/Mixamo_Eval_Final"
SAVE_PATH="./eval_results/Gen_Scene/Mixamo_Eval_Final/SSIM.csv"

python compute_reward_flow_video_gen_scene_ssim.py \
    --root_path ${DATA_ROOT_PATH} \
    --save_path ${SAVE_PATH} \
    --mag_thresh 2.0
# DATA_ROOT_PATH="/workspace_4T/local_data/DynaVid_Analysis_Data/Pexels_Eval/"
# SAVE_PATH="./eval_results/Dataset_GT/Pexels_eval.csv"

# python compute_reward_flow_video_data.py \
#     --root_path ${DATA_ROOT_PATH} \
#     --save_path ${SAVE_PATH} \
#     --mag_thresh 2.0


DATA_ROOT_PATH="/workspace_4T/local_data/DynaVid_Analysis_Data/Mixamo_Eval_Final/"
SAVE_PATH="./eval_results/Dataset_GT/Mixamo_Eval_Final.csv"

python compute_reward_flow_video_data.py \
    --root_path ${DATA_ROOT_PATH} \
    --save_path ${SAVE_PATH} \
    --mag_thresh 2.0
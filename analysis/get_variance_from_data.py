import torch
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from safetensors.torch import load_file, save_file
from pathlib import Path
import json
from tqdm import tqdm

import pdb

def filter_metadata(metadata: List[Dict[str, Any]], use_cycle_filtering=False, thres=90) -> List[Dict[str, Any]]:
    """
    Filter metadata based on the flow statistics.
    Eliminates most erroneous data based on the reconstruction error and flow cycle discrepancy. (default 15% of the data)
    """
    assert thres >= 0 and thres <= 100, "Threshold must be between 0 and 100"
    
    if use_cycle_filtering:
        cycle_error = np.array([line["flow_cycle_discrepancy"] for line in metadata])
        cycle_error_thres = np.percentile(cycle_error, thres)
    
    if use_cycle_filtering:
        metadata = [line for line in metadata if line["flow_cycle_discrepancy"] <= cycle_error_thres]
    
    return metadata

if __name__ == "__main__":
    
    """
    Compute SNR based on the signal power obtained from the video latent data.
    """
    
    root_dir = Path("/workspace/data/Wan/Shutterstock_100K/")

    # latent_dir = Path("/workspace/data/Wan/Shutterstock_100K/flow_latent_WAFT/")  # 여기 안에 latent_000.pt 이런 식이라고 가정
    # latent_files = sorted(latent_dir.glob("*.safetensors"))  # npy면 *.npy

    metadata_path = root_dir / "metadata.jsonl"
    

    metadata = []
    # load videos and prompt from metadata
    with open(metadata_path, "r") as f:
        for line in f:
            metadata.append( json.loads(line) )
            
    metadata = filter_metadata(metadata, use_cycle_filtering=True, thres=90)
    
    # metadata_thres50 = filter_metadata(metadata, use_cycle_filtering=True, thres=50)
    
    # flow_mean = [x['flow_magnitude_mean'] for x in metadata]
    # tmp = [x for x in flow_max if x > 300]
    # pdb.set_trace()
            
    flow_path_list = [root_dir / 'flow_latent_WAFT' / (line['hash_code'] + '.safetensors') for line in metadata]


    sum_mean, sum_var, count = 0.0, 0.0, 0

    for f in tqdm(flow_path_list[:1000]):
        z = load_file(f)['flow_latent']  # shape: (C, T, H, W) or (T, C, H, W) 어떤거든 상관 X
        z = z.float()
        m = z.mean().item()
        v = ((z-m)**2).mean().item()
        
        sum_mean += m
        sum_var += v
        count += 1

    global_mean = sum_mean / count
    global_var = sum_var / count

    print("global_mean:", global_mean)
    print("global_var:", global_var)

    pdb.set_trace()

    # E[z] == 0 가정하지 않고, 실제 var(z) 계산 (1000개)
    # global_mean: -0.08364875829248195
    # global_var: 2.5899988787174224

    # E[z] == 0 가정했을 때. (약간 근사치임. 실제로 E[z]!=0)
    # signal_power (E[z^2]): 2.599205231666565 (100개)
    # signal_power (E[z^2]): 135538.56108326363 (1000개, outlier 너무 많음)
    # signal_power (E[z^2]): 2.600289835333824 (1000개, cycle filtering 후)
    
    
    #------------------------------------------------------------------------------------------------
    """
    Compute SNR based on the signal power obtained from the video latent data.
    """
    
    # root_dir = Path("/workspace/data/Wan/Shutterstock_100K/")

    # # latent_dir = Path("/workspace/data/Wan/Shutterstock_100K/flow_latent_WAFT/")  # 여기 안에 latent_000.pt 이런 식이라고 가정
    # # latent_files = sorted(latent_dir.glob("*.safetensors"))  # npy면 *.npy

    # metadata_path = root_dir / "metadata.jsonl"
    

    # metadata = []
    # # load videos and prompt from metadata
    # with open(metadata_path, "r") as f:
    #     for line in f:
    #         metadata.append( json.loads(line) )
            
    # metadata = filter_metadata(metadata, use_cycle_filtering=True, thres=90)
    
    # # flow_max = [x['flow_magnitude_max'] for x in metadata]
    # # tmp = [x for x in flow_max if x > 300]
    # # pdb.set_trace()
            
    # video_path_list = [root_dir / 'video_latent' / (line['hash_code'] + '.safetensors') for line in metadata]


    # sum_sq = 0.0
    # count = 0
    # mean_list = []
    # sum_sq_list = []

    # for f in tqdm(video_path_list[:100]):
    #     z = load_file(f)['video_latent']  # shape: (C, T, H, W) or (T, C, H, W) 어떤거든 상관 X
    #     z = z.float()
    #     mean = z.mean().item()
    #     mean_list.append( mean )
        
    #     mean_sq = (z ** 2).mean().item()  # 파일 하나의 mean(z^2)
    #     sum_sq += mean_sq
    #     sum_sq_list.append( mean_sq )
    #     count += 1

    # signal_power = sum_sq / count  # = E[z^2]
    # print("signal_power (E[z^2]):", signal_power)


    # # signal_power (E[z^2]): 1.2318683928251266 (100개)
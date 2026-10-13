import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pdb

NUM_SAMPLE=1000

# RE10K
# data_root = "/workspace/data/Wan/RE10K_Train/cam_param"
# dataset_name = "RE10K_Train"

# DynaVid_Camera_New_1203
data_root = "/workspace_4T/local_data/DynaVid_Camera_New_1203/datasets_organized/cam_param"
dataset_name = "DynaVid_CameraData_New_1203_Normalized"

# DynaVid_Prev.
# data_root = "/workspace/data/Wan/DynaVid_CameraData_Final/cam_param"
# dataset_name = "DynaVid_CameraData_Prev"




# 모든 .npz 파일 수집
npz_files = sorted(glob.glob(os.path.join(data_root, "*.npz")))[:NUM_SAMPLE]

all_magnitudes = []

for path in tqdm(npz_files):
    data = np.load(path)
    
    # extrinsics: (B, 4, 4)
    extrinsics = data["extrinsics"]  
    
    # translation vector: (B, 3)
    t = extrinsics[:, :3, 3]
    
    # magnitude: (B,) -> 1D Avg.
    mag_avg = np.linalg.norm(t, axis=1).mean()
    
    # 리스트에 저장
    all_magnitudes.append(mag_avg)

# 하나의 array로 합치기
# all_magnitudes = np.concatenate(all_magnitudes, axis=0)
all_magnitudes = np.array(all_magnitudes)

print("Total samples:", len(all_magnitudes))
print("Magnitude stats: mean =", all_magnitudes.mean(), ", max =", all_magnitudes.max())

# -----------------------------
# 🔥 Histogram Plot & Save
# -----------------------------
plt.figure(figsize=(8, 5))

# 전체 sample 수
N = len(all_magnitudes)

# weights = 각 sample마다 1/N을 부여 → bin 높이가 "전체 대비 비율"이 됨
plt.hist(all_magnitudes, bins=50, weights=np.ones(N) / N)

plt.xlabel("Translation Magnitude")
plt.ylabel("Ratio")
plt.title("Histogram of Camera Translation Magnitudes (Normalized)")
plt.grid(True)
plt.ylim(0, 1)  # 세로축 0 ~ 1 범위 고정
plt.tight_layout()

os.makedirs("camera_histogram", exist_ok=True)
save_path = f"camera_histogram/translation_magnitude_hist_{dataset_name}.png"
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved histogram to {save_path}")
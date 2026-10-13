import os
import torch
import torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio
from safetensors.torch import load_file
import csv

import pdb

def load_video_mp4(path, device="cuda"):
    """
    Load MP4 → [T, C, H, W], float32, [0,1]
    """
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frame = torch.from_numpy(frame).permute(2,0,1)  # HWC → CHW
        frames.append(frame)
    reader.close()

    video = torch.stack(frames).float().to(device) / 255.0
    return video  # [T,C,H,W]


def load_flow_safetensor(path, key="flow", device="cuda"):
    """
    Safetensor → [T, 2, H, W], float32
    """
    data = load_file(path, device=device)
    flow = data[key].float()
    return flow

# ================================
# 1. Forward bilinear splatting
# ================================
def forward_bilinear_splatting_general(image, flow, mask=None):
    """
    Forward warping (splatting) with bilinear interpolation for an entire batch.

    Args:
      image: (B, C, H, W)  # source image (C arbitrary)
      flow:  (B, 2, H, W)  # forward flow (dx, dy)
      mask:  (B, 1, H, W)  # 1: valid, 0: invalid (optional)

    Returns:
      warped: (B, C, H, W) # forward-warped result
    """
    device = image.device
    B, C, H, W = image.shape

    if mask is None:
        mask = torch.ones(B, 1, H, W, device=device, dtype=image.dtype)

    assert flow.shape == (B, 2, H, W), "flow는 (B,2,H,W) 형태여야 합니다."
    assert mask.shape == (B, 1, H, W), "mask는 (B,1,H,W) 형태여야 합니다."

    # (B,C,H,W) -> (B,H,W,C)
    image_bhwc = image.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
    flow_bhw2  = flow.permute(0, 2, 3, 1).contiguous()   # (B,H,W,2)
    mask_bhw   = mask.view(B, H, W)                      # (B,H,W)

    # Flatten
    image_flat = image_bhwc.view(-1, C)   # (B*H*W, C)
    flow_flat  = flow_bhw2.view(-1, 2)    # (B*H*W, 2)
    mask_flat  = mask_bhw.view(-1)        # (B*H*W,)

    # base coordinates
    b_grid = torch.arange(B, device=device).view(B,1,1).expand(-1,H,W)  # (B,H,W)
    y_grid = torch.arange(H, device=device).view(1,H,1).expand(B,-1,W)
    x_grid = torch.arange(W, device=device).view(1,1,W).expand(B,H,-1)

    b_idx = b_grid.flatten()  # (B*H*W)
    y_idx = y_grid.flatten()
    x_idx = x_grid.flatten()

    # apply flow: (x+dx, y+dy)
    dx = flow_flat[:, 0]
    dy = flow_flat[:, 1]
    tx = x_idx.to(dx.dtype) + dx
    ty = y_idx.to(dy.dtype) + dy

    # floor / ceil
    tx0 = torch.floor(tx).long()
    tx1 = tx0 + 1
    ty0 = torch.floor(ty).long()
    ty1 = ty0 + 1

    alpha = tx - tx0.to(tx.dtype)  # (B*H*W)
    beta  = ty - ty0.to(ty.dtype)

    # validity
    valid = ((mask_flat == 1) &
             (tx0 >= 0) & (tx1 < W) &
             (ty0 >= 0) & (ty1 < H))
    valid_idx = valid.nonzero(as_tuple=True)[0]  # (N,)

    v_b     = b_idx[valid_idx]
    v_x0    = tx0[valid_idx]
    v_x1    = tx1[valid_idx]
    v_y0    = ty0[valid_idx]
    v_y1    = ty1[valid_idx]
    v_alpha = alpha[valid_idx]
    v_beta  = beta[valid_idx]
    v_src   = image_flat[valid_idx]  # (N,C)

    # bilinear weights
    w00 = (1 - v_alpha) * (1 - v_beta)
    w01 = v_alpha       * (1 - v_beta)
    w10 = (1 - v_alpha) * v_beta
    w11 = v_alpha       * v_beta

    # output buffers
    warped_bhwc = torch.zeros_like(image_bhwc)          # (B,H,W,C)
    weight_map  = torch.zeros((B, H, W), device=device, dtype=image.dtype)

    warped_flat = warped_bhwc.view(-1, C)  # (B*H*W,C)
    weight_flat = weight_map.view(-1)      # (B*H*W,)

    def flatten_index(b, y, x):
        return b*(H*W) + (y * W) + x

    i00 = flatten_index(v_b, v_y0, v_x0)
    i01 = flatten_index(v_b, v_y0, v_x1)
    i10 = flatten_index(v_b, v_y1, v_x0)
    i11 = flatten_index(v_b, v_y1, v_x1)

    # scatter_add
    warped_flat.index_add_(0, i00, w00.unsqueeze(-1) * v_src)
    warped_flat.index_add_(0, i01, w01.unsqueeze(-1) * v_src)
    warped_flat.index_add_(0, i10, w10.unsqueeze(-1) * v_src)
    warped_flat.index_add_(0, i11, w11.unsqueeze(-1) * v_src)

    weight_flat.index_add_(0, i00, w00)
    weight_flat.index_add_(0, i01, w01)
    weight_flat.index_add_(0, i10, w10)
    weight_flat.index_add_(0, i11, w11)

    # normalize
    w_valid = (weight_flat > 0)
    warped_flat[w_valid] /= weight_flat[w_valid].unsqueeze(-1)

    warped_bhwc = warped_flat.view(B, H, W, C)
    warped = warped_bhwc.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
    
    return warped


# =========================================================
# 2. Photometric error computation using forward flow
# =========================================================
def compute_photometric_error_forward_consistent(video, flow, mask=None):
    """
    video: [T, C, H, W]
    flow : [T, 2, H, W]   # forward flow, F[t+1]: V_t -> V_{t+1}
    mask : [T, 1, H, W] or None
    """
    device = video.device
    T, C, H, W = video.shape

    errors = []

    for t in range(T - 1):
        Vn   = video[t].unsqueeze(0)      # (1,C,H,W)
        Vnp1 = video[t+1].unsqueeze(0)    # (1,C,H,W)
        Ft1  = flow[t+1].unsqueeze(0)     # (1,2,H,W)

        if mask is not None:
            Mn = mask[t].unsqueeze(0)     # (1,1,H,W)
        else:
            Mn = None

        Vnp1_pred = forward_bilinear_splatting_general(Vn, Ft1, mask=Mn)

        err = torch.abs(Vnp1_pred - Vnp1).mean()
        errors.append(err.item())

    errors = np.array(errors)
    return errors, float(errors.mean())

def compute_identity_error(video):
    """
    video: [T, C, H, W]
    return: per-frame identity photometric errors, mean
    """
    T = video.shape[0]
    errors = []
    for t in range(T - 1):
        err = torch.abs(video[t] - video[t+1]).mean()
        errors.append(err.item())
    errors = np.array(errors)
    return errors, float(errors.mean())

def compute_zero_flow_warp_error(video):
    """
    video: [T, C, H, W]
    forward warp using flow = 0
    """
    T, C, H, W = video.shape
    zero_flow = torch.zeros(1, 2, H, W, device=video.device)  # (1,2,H,W)
    errors = []

    for t in range(T - 1):
        Vn   = video[t].unsqueeze(0)
        Vnp1 = video[t+1].unsqueeze(0)
        
        # warp Vn using zero flow
        Vnp1_pred = forward_bilinear_splatting_general(Vn, zero_flow)
        
        err = torch.abs(Vnp1_pred - Vnp1).mean()
        errors.append(err.item())

    errors = np.array(errors)
    return errors, float(errors.mean())


def compute_photometric_error_forward_motion_mask(video, flow, mag_thresh=2.0):
    """
    video: [T,C,H,W], float
    flow : [T,2,H,W]
    mag_thresh: threshold for motion mask (pixel units)
    """

    T, C, H, W = video.shape
    errors = []

    for t in range(T - 1):
        Vn   = video[t].unsqueeze(0)
        Vnp1 = video[t+1].unsqueeze(0)
        Ft1  = flow[t+1].unsqueeze(0)

        # forward warp
        Vnp1_pred = forward_bilinear_splatting_general(Vn, Ft1)

        # magnitude mask
        u = Ft1[:,0]   # [1,H,W]
        v = Ft1[:,1]
        mag = torch.sqrt(u*u + v*v)  # [1,H,W]
        mask = (mag > mag_thresh).float()  # [1,H,W]

        if mask.sum() < 1:
            # no-motion region → skip or treat as zero
            continue

        # masked photometric error
        err = (torch.abs(Vnp1_pred - Vnp1) * mask.unsqueeze(1)).sum() / mask.sum()
        errors.append(err.item())

    if len(errors) == 0:
        return np.array([0.0]), 0.0

    errors = np.array(errors)
    return errors, float(errors.mean())


def compute_ssim(x, y, C1=0.01**2, C2=0.03**2):
    """
    x, y: (B, C, H, W), range [0,1]
    """
    mu_x = F.avg_pool2d(x, 3, 1, padding=1)
    mu_y = F.avg_pool2d(y, 3, 1, padding=1)

    sigma_x  = F.avg_pool2d(x * x, 3, 1, padding=1) - mu_x * mu_x
    sigma_y  = F.avg_pool2d(y * y, 3, 1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, padding=1) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)

    return num / (den + 1e-12)  # SSIM map

def compute_masked_ssim_error(video, flow, mag_thresh=2.0):
    """
    video: [T, C, H, W]
    flow : [T, 2, H, W]
    """
    T = video.shape[0]
    device = video.device

    ssim_errors = []

    for t in range(T - 1):
        Vt   = video[t:t+1]     # (1,C,H,W)
        Vtp1 = video[t+1:t+2]   # (1,C,H,W)
        Ft1  = flow[t+1:t+2]    # (1,2,H,W)

        # forward warp prediction
        Vtp1_pred = forward_bilinear_splatting_general(Vt, Ft1)

        # Normalize to [0,1]
        Vtp1_pred_clamped = torch.clamp(Vtp1_pred, 0.0, 1.0)
        Vtp1_clamped      = torch.clamp(Vtp1, 0.0, 1.0)

        # compute SSIM
        ssim_map = compute_ssim(Vtp1_pred_clamped, Vtp1_clamped)  # (1,3,H,W) or (1,1,H,W)
        ssim_map = ssim_map.mean(dim=1, keepdim=True)  # reduce channel → (1,1,H,W)

        # motion mask
        mag = torch.norm(Ft1, dim=1, keepdim=True)  # (1,1,H,W)
        motion_mask = (mag > mag_thresh).float()

        # apply mask
        masked_ssim = (ssim_map * motion_mask).sum() / (motion_mask.sum() + 1e-6)

        # SSIM → error
        ssim_error = 1.0 - masked_ssim
        ssim_errors.append(ssim_error.item())

    return ssim_errors, float(np.mean(ssim_errors))

# =========================================================
# Example Usage (video, flow 텐서 이미 로드되어 있다고 가정)
# =========================================================
if __name__ == "__main__":
    device = "cuda"

    # root_path = "/workspace_4T/DynaVid/results/T2FV_MotionReg_timesteps/Pexels_Eval/"
    root_path = "/workspace_4T/DynaVid/results/T2FV_MotionReg_timesteps/Mixamo_Eval_Final/"
    scene_name_list = os.listdir(os.path.join(root_path, "generated_videos"))
    timestep_list = [146, 908, 999]

    # Random Flow
    flow_random_all        = (torch.randn(121,2,704,1280) * 5.0).to(device)
    flow_random_all_scale1 = (torch.randn(121,2,704,1280) * 1.0).to(device)

    identity_errors = []
    zero_flow_errors = []
    flow_warp_errors = []
    motion_mask_errors = []
    masked_ssim_errors = []       # <<< 새로 추가됨
    random_flow_errors = []
    random_flow_errors_scale1 = []

    for timestep in timestep_list:
        for scene_name in scene_name_list:

            video_path = os.path.join(root_path, "generated_videos", scene_name, f"{timestep}.mp4")
            flow_path  = os.path.join(root_path, "generated_flow_tensors", scene_name, f"{timestep}.safetensors")

            if not os.path.exists(video_path) or not os.path.exists(flow_path):
                continue

            video = load_video_mp4(video_path).to(device)
            flow  = load_flow_safetensor(flow_path).to(device)

            # 1) Identity
            _, id_mean = compute_identity_error(video)
            identity_errors.append(id_mean)

            # 2) Zero flow warp
            _, zf_mean = compute_zero_flow_warp_error(video)
            zero_flow_errors.append(zf_mean)

            # 3) Real flow warp (photometric)
            _, fw_mean = compute_photometric_error_forward_consistent(video, flow)
            flow_warp_errors.append(fw_mean)

            # 4) Motion-mask photometric error
            _, mm_mean = compute_photometric_error_forward_motion_mask(video, flow, mag_thresh=2.0)
            motion_mask_errors.append(mm_mean)

            # 5) Masked SSIM Error  ←← 추가됨
            _, ssim_mean = compute_masked_ssim_error(video, flow, mag_thresh=2.0)
            masked_ssim_errors.append(ssim_mean)

            # 6) Random flow scale=5
            _, rand_mean = compute_photometric_error_forward_consistent(video, flow_random_all)
            random_flow_errors.append(rand_mean)

            # 7) Random flow scale=1
            _, rand1_mean = compute_photometric_error_forward_consistent(video, flow_random_all_scale1)
            random_flow_errors_scale1.append(rand1_mean)

        # Print summary per timestep
        print(f"\n===== Overall Results - Timestep: {timestep} =====")
        print(f"Identity Error: {np.mean(identity_errors):.6f}")
        print(f"Zero-Flow Warp Error: {np.mean(zero_flow_errors):.6f}")
        print(f"Flow Warp Error: {np.mean(flow_warp_errors):.6f}")
        print(f"Motion-Mask Flow Error: {np.mean(motion_mask_errors):.6f}")
        print(f"Masked SSIM Error: {np.mean(masked_ssim_errors):.6f}")  # <<< 추가됨
        print(f"Random-Flow Warp Error (scale=5): {np.mean(random_flow_errors):.6f}")
        print(f"Random-Flow Warp Error (scale=1): {np.mean(random_flow_errors_scale1):.6f}")
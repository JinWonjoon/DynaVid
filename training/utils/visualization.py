#--------------------------------------------------------------------------------------------------------------
# Codes borrowed from https://github.com/ChristophReich1996/Optical-Flow-Visualization-PyTorch

from typing import Optional, Union

import torch
import math
from math import pi as PI


def get_color_wheel(device: torch.device) -> torch.Tensor:
    """
    Generates the color wheel.
    :param device: (torch.device) Device to be used
    :return: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    """
    # Set constants
    RY: int = 15
    YG: int = 6
    GC: int = 4
    CB: int = 11
    BM: int = 13
    MR: int = 6
    # Init color wheel
    color_wheel: torch.Tensor = torch.zeros((RY + YG + GC + CB + BM + MR, 3), dtype=torch.float32)
    # Init counter
    counter: int = 0
    # RY
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    counter: int = counter + RY
    # YG
    color_wheel[counter:counter + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    color_wheel[counter:counter + YG, 1] = 255
    counter: int = counter + YG
    # GC
    color_wheel[counter:counter + GC, 1] = 255
    color_wheel[counter:counter + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    counter: int = counter + GC
    # CB
    color_wheel[counter:counter + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)
    color_wheel[counter:counter + CB, 2] = 255
    counter: int = counter + CB
    # BM
    color_wheel[counter:counter + BM, 2] = 255
    color_wheel[counter:counter + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    counter: int = counter + BM
    # MR
    color_wheel[counter:counter + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)
    color_wheel[counter:counter + MR, 0] = 255
    # To device
    color_wheel: torch.Tensor = color_wheel.to(device)
    return color_wheel


def _flow_hw_to_color(flow_vertical: torch.Tensor, flow_horizontal: torch.Tensor,
                      color_wheel: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Private function applies the flow color wheel to flow components (vertical and horizontal).
    :param flow_vertical: (torch.Tensor) Vertical flow of the shape [height, width]
    :param flow_horizontal: (torch.Tensor) Horizontal flow of the shape [height, width]
    :param color_wheel: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    :param: device: (torch.device) Device to be used
    :return: (torch.Tensor) Visualized flow of the shape [3, height, width]
    """
    # Get shapes
    _, height, width = flow_vertical.shape
    # Init flow image
    flow_image: torch.Tensor = torch.zeros(3, height, width, dtype=torch.float32, device=device)
    # Get number of colors
    number_of_colors: int = color_wheel.shape[0]
    # Compute norm, angle and factors
    flow_norm: torch.Tensor = (flow_vertical ** 2 + flow_horizontal ** 2).sqrt()
    angle: torch.Tensor = torch.atan2(- flow_vertical, - flow_horizontal) / PI
    fk: torch.Tensor = (angle + 1.) / 2. * (number_of_colors - 1.)
    k0: torch.Tensor = torch.floor(fk).long()
    k1: torch.Tensor = k0 + 1
    k1[k1 == number_of_colors] = 0
    f: torch.Tensor = fk - k0
    # Iterate over color components
    for index in range(color_wheel.shape[1]):
        # Get component of all colors
        tmp: torch.Tensor = color_wheel[:, index]
        # Get colors
        color_0: torch.Tensor = tmp[k0] / 255.
        color_1: torch.Tensor = tmp[k1] / 255.
        # Compute color
        color: torch.Tensor = (1. - f) * color_0 + f * color_1
        # Get color index
        color_index: torch.Tensor = flow_norm <= 1
        # Set color saturation
        color[color_index] = 1 - flow_norm[color_index] * (1. - color[color_index])
        color[~color_index] = color[~color_index] * 0.75
        # Set color in image
        flow_image[index] = torch.floor(255 * color)
    return flow_image


def flow_to_color(flow: torch.Tensor, clip_flow: Optional[Union[float, torch.Tensor]] = None, 
                  normalize_over_video: bool = False) -> torch.Tensor:
    """
    Function converts a given optical flow map into the classical color schema.
    :param flow: (torch.Tensor) Optical flow tensor of the shape [batch size (optional), 2, height, width].
    :param clip_flow: (Optional[Union[float, torch.Tensor]]) Max value of flow values for clipping (default None).
    :param normalize_over_video: (bool) If true scale is normalized over the whole video (batch).
    :return: (torch.Tensor) Flow visualization (float tensor) with the shape [batch size (if used), 3, height, width].
    """
    # Check parameter types
    assert torch.is_tensor(flow), "Given flow map must be a torch.Tensor, {} given".format(type(flow))
    assert torch.is_tensor(clip_flow) or isinstance(clip_flow, float) or clip_flow is None, \
        "Given clip_flow parameter must be a float, a torch.Tensor, or None, {} given".format(type(clip_flow))
    # Check shapes
    assert flow.ndimension() in [3, 4], \
        "Given flow must be a 3D or 4D tensor, given tensor shape {}.".format(flow.shape)
    if torch.is_tensor(clip_flow):
        assert clip_flow.ndimension() == 0, \
            "Given clip_flow tensor must be a scalar, given tensor shape {}.".format(clip_flow.shape)
    # Manage batch dimension
    batch_dimension: bool = True
    if flow.ndimension() == 3:
        flow = flow[None]
        batch_dimension: bool = False
    # Save shape
    batch_size, _, height, width = flow.shape
    # Check flow dimension
    assert flow.shape[1] == 2, "Flow dimension must have the shape 2 but tensor with {} given".format(flow.shape[1])
    # Save device
    device: torch.device = flow.device
    # Clip flow if utilized
    if clip_flow is not None:
        flow = flow.clip(max=clip_flow)
    # Get horizontal and vertical flow
    flow_vertical: torch.Tensor = flow[:, 0:1]
    flow_horizontal: torch.Tensor = flow[:, 1:2]
    # Get max norm of flow
    flow_max_norm: torch.Tensor = (flow_vertical ** 2 + flow_horizontal ** 2).sqrt().view(batch_size, -1).max(dim=-1)[0]
    flow_max_norm: torch.Tensor = flow_max_norm.view(batch_size, 1, 1, 1)
    if normalize_over_video:
        flow_max_norm: Tensor = flow_max_norm.max(dim=0, keepdim=True)[0]
    # Normalize flow
    flow_vertical: torch.Tensor = flow_vertical / (flow_max_norm + 1e-05)
    flow_horizontal: torch.Tensor = flow_horizontal / (flow_max_norm + 1e-05)
    # Get color wheel
    color_wheel: torch.Tensor = get_color_wheel(device=device)
    # Init flow image
    flow_image = torch.zeros(batch_size, 3, height, width, device=device)
    # Iterate over batch dimension
    for index in range(batch_size):
        flow_image[index] = _flow_hw_to_color(flow_vertical=flow_vertical[index],
                                              flow_horizontal=flow_horizontal[index], color_wheel=color_wheel,
                                              device=device)
    return flow_image if batch_dimension else flow_image[0]


@torch.no_grad()
def rgb_hsv_flow_inverse(
    rgb: torch.Tensor,
    sigma: float = 0.15,
    convention: str = "uv",   # "uv" for atan2(v,u), "neg_uv" for atan2(-v,-u)
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Invert VideoJAM-style HSV mapping: RGB[-1,1] -> optical flow (u,v).
    Assumes forward mapping used: H=(atan2(v,u)+pi)/(2pi), S=1, V=min(1, ||flow||/(sigma*sqrt(H^2+W^2))).

    Args:
        rgb: (B,3,H,W), float in [-1,1]
        H_img, W_img: image height/width used in forward mapping
        sigma: same sigma used in forward mapping
        convention: "uv" or "neg_uv" (must match forward)
    Returns:
        flow: (B,2,H,W), float (u,v)
    """
    assert rgb.ndim == 4 and rgb.size(1) == 3, "rgb must be (B,3,H,W)"
    B, _, H, W = rgb.shape
    H_img = H
    W_img = W
    # assert H == H_img and W == W_img, "Input size must match forward H,W"

    # [-1,1] -> [0,1]
    rgb01 = (rgb.clamp(-1.0, 1.0) + 1.0) * 0.5
    r, g, b = rgb01[:,0], rgb01[:,1], rgb01[:,2]

    # RGB -> HSV (standard)
    cmax = torch.max(torch.max(r, g), b)
    cmin = torch.min(torch.min(r, g), b)
    delta = (cmax - cmin).clamp_min(eps)

    # Hue in [0,1]
    # When delta==0, hue is undefined; set to 0 (flow mag will be 0 if V==0 anyway).
    hue = torch.zeros_like(cmax)

    mask_r = (cmax == r)
    mask_g = (cmax == g)
    mask_b = (cmax == b)

    hue = torch.where(mask_r, ((g - b) / delta) % 6.0, hue)
    hue = torch.where(mask_g, ((b - r) / delta) + 2.0, hue)
    hue = torch.where(mask_b, ((r - g) / delta) + 4.0, hue)
    hue = (hue / 6.0)  # scale to [0,1]

    # Value = cmax (since S=1 in forward; small deviations ok)
    V = cmax.clamp(0.0, 1.0)

    # Recover angle and magnitude
    alpha = (hue * 2.0 * math.pi) - math.pi          # [-pi,pi]
    diag = math.sqrt(H_img*H_img + W_img*W_img)
    mag = V * (sigma * (diag + eps))                 # unclamped magnitude

    # flow components
    u = mag * torch.cos(alpha)
    v = mag * torch.sin(alpha)

    if convention == "uv":
        # forward used atan2(v,u) -> already correct
        pass
    elif convention == "neg_uv":
        # forward used atan2(-v, -u) i.e., angle of (-u,-v)
        # So original (u0,v0) satisfy: atan2(v0,u0) == atan2(-v,-u)
        # one consistent mapping is simply to negate both components
        u, v = -u, -v
    else:
        raise ValueError("convention must be 'uv' or 'neg_uv'")

    # If forward had clamp V = min(1, ...), then mag above threshold is unrecoverable.
    # You get lower-bound magnitude only.
    flow = torch.stack([u, v], dim=1)
    return flow



@torch.no_grad()
def rgb_to_flow_ours_inverse(
    rgb: torch.Tensor,
    scale: float,
    p: float = 0.5,
    m_max: float = 1.0,
    eps: float = 1e-8
):
    """
    Inverse of flow_to_rgb_ours:
    Input RGB in [-1,1].
    Returns reconstructed flow [B,2,H,W].
    """
    # [-1,1] → [0,1] for HSV conversion
    rgb = (rgb + 1) / 2

    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

    # -----------------------------
    # 1) RGB -> HSV
    # -----------------------------
    maxc, _ = torch.max(rgb, dim=1, keepdim=True)
    minc, _ = torch.min(rgb, dim=1, keepdim=True)
    v = maxc
    delta = maxc - minc + eps

    # Saturation
    s = delta / (maxc + eps)

    # Hue
    rc = (maxc - r) / (delta + eps)
    gc = (maxc - g) / (delta + eps)
    bc = (maxc - b) / (delta + eps)

    h = torch.zeros_like(maxc)
    mask = (maxc == r)
    h[mask] = (bc - gc)[mask]
    mask = (maxc == g)
    h[mask] = 2.0 + (rc - bc)[mask]
    mask = (maxc == b)
    h[mask] = 4.0 + (gc - rc)[mask]

    h = (h / 6.0) % 1.0  # [0,1]

    # -----------------------------
    # 2) HSV -> angle, magnitude
    # -----------------------------
    alpha = h * 2 * torch.pi - torch.pi  # [-pi, pi]
    m = s  # ours forward에서 s = m/m_max, v=1

    # ours 방식 역변환: r = (m^1/p) * scale
    r = torch.pow(m_max * m, 1.0 / p) * scale

    # -----------------------------
    # 3) Polar -> Cartesian flow
    # -----------------------------
    u = r * torch.cos(alpha)
    v = r * torch.sin(alpha)
    flow = torch.cat([u, v], dim=1)

    return flow
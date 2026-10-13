from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import argparse
import logging
import os
import sys
from typing import Literal, Optional
from pathlib import Path
import json
from datetime import timedelta
import random
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from einops import rearrange, repeat
import math
import numpy as np
from PIL import Image

import torch
from diffusers.utils import export_to_video, load_image, load_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanPipeline,
    WanTransformer3DModel,
    UniPCMultistepScheduler,
)

sys.path.append(os.path.abspath(os.path.join(sys.path[0], "../")))
from training.modules.transformer_wan_vace_tf2v import WanVACETransformer3DModel_TF2V
from training.pipeline.wan_vace_tf2v_pipeline import WanVACEPipeline_TF2V
from training.datasets.utils import preprocess_image_with_resize, preprocess_video_with_resize
from transformers import T5TokenizerFast, UMT5EncoderModel
from training.constants import LOG_LEVEL, LOG_NAME
from training.utils import custom_meshgrid, flow_to_color
from training.trainer import get_t5_prompt_embeds
from inference.utils import decode_flow_latents

from torch.utils.data import Dataset
from torchvision import transforms

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import pdb


os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)


def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30), **kwargs)

    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))

    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    # https://github.com/pytorch/pytorch/issues/98763
    # torch.cuda.set_device(local_rank)

    return local_rank



class ImgFlowVideoDataset_Inference(Dataset):
    def __init__(
        self, 
        max_num_frames: int, 
        height: int, 
        width: int, 
        data_root: str,
        max_num_videos: int = None,
        use_camparam: bool = False,
        use_flow: bool = False,
    ) -> None:
        data_root = Path(data_root)
        metadata_path = data_root / "metadata.jsonl"
        self.use_camparam = use_camparam
        self.use_flow = use_flow
        
        assert metadata_path.is_file(), "For this dataset type, you need metadata.jsonl in the root path"
        
        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append( json.loads(line) )
        
        if max_num_videos is not None:
            metadata = metadata[:max_num_videos]
        
        self.prompts = [x["prompt"] for x in metadata]
        self.hash_code = [x["hash_code"] for x in metadata]
        
        prompt_embedding_dir_path = "prompt_embeddings"
        flow_embedding_dir_path = f"flow_latent_WAFT"
        camparam_dir_path = f"cam_param"
        
        self.prompt_embeddings = [data_root / prompt_embedding_dir_path / (x["hash_code"] + '.safetensors') for x in metadata]
        self.videos = [data_root / "video_latent" / (x["hash_code"] + '.safetensors') for x in metadata]
        self.images = [data_root / "first_frames" /  (x["hash_code"] + '.png') for x in metadata]
        if self.use_flow:
            self.flows = [data_root / flow_embedding_dir_path/ (x["hash_code"] + '.safetensors') for x in metadata]
        if use_camparam:
            self.camparam = [data_root / camparam_dir_path/ (x["hash_code"] + '.npz') for x in metadata]
        
        # Check if number of prompts matches number of videos and images
        if self.use_flow:
            if not (len(self.videos) == len(self.prompts) == len(self.images) == len(self.flows)):
                raise ValueError(
                    f"Expected length of prompts, videos and images to be the same but found {len(self.prompts)=}, {len(self.videos)=}, {len(self.images)=} and {len(self.flows)=}. Please ensure that the number of caption prompts, videos and images match in your dataset."
                )
        else:
            if not (len(self.videos) == len(self.prompts) == len(self.images)):
                raise ValueError(
                    f"Expected length of prompts, videos and images to be the same but found {len(self.prompts)=}, {len(self.videos)=}, {len(self.images)=}. Please ensure that the number of caption prompts, videos and images match in your dataset."
                )
        
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms
        
        self.length = len(self.videos)
        print(f"Dataset size: {self.length}")
        
        
        
    def __len__(self) -> int:
        return self.length
    
    def load_data_pair(self, index):
        # prompt = self.prompts[index]
        prompt_embedding_path = self.prompt_embeddings[index]
        encoded_video_path = self.videos[index]
        
        encoded_flow = "null"
        
        if self.use_flow:
            encoded_flow_path = self.flows[index]
        
        prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"] 
        encoded_video = load_file(encoded_video_path)["video_latent"] # CFHW
        if self.use_flow:
            encoded_flow = load_file(encoded_flow_path)["flow_latent"] # CFHW
        
        prompt_embedding = pad_text_embedding(prompt_embedding)
        encoded_video = rearrange(encoded_video, 'f c h w -> c f h w') # FCHW
        if self.use_flow:
            encoded_flow = rearrange(encoded_flow, 'f c h w -> c f h w') # FCHW
            
        # if self.use_flow:
        #     assert encoded_flow.shape == (48, 31, 44, 80) and encoded_video.shape == (48, 31, 44, 80) and prompt_embedding.shape == (512, 4096), f"For Wan-TI2V-5B, we use 121 frames, which would be 31 frames when encoded by VAE. || flow : {encoded_flow.shape} | video : {encoded_video.shape} | text : {prompt_embedding.shape} ||"
        # else:
        #     assert encoded_video.shape == (48, 31, 44, 80) and prompt_embedding.shape == (512, 4096), f"For Wan-TI2V-5B, we use 121 frames, which would be 31 frames when encoded by VAE. || flow : {encoded_flow.shape} | video : {encoded_video.shape} | text : {prompt_embedding.shape} ||"
        
        return prompt_embedding, encoded_video, encoded_flow
    
    def load_camparam(self, index):
        camparam_path = self.camparam[index]
        camparam_all = np.load(camparam_path)
        intrinsics, extrinsics = camparam_all['intrinsics'], camparam_all['extrinsics'] # THW
        if intrinsics.ndim == 2:
            intrinsics = repeat(camparam_all['intrinsics'], 'h w -> n h w', n=camparam_all['extrinsics'].shape[0])
        
        intrinsics = torch.tensor(intrinsics)
        extrinsics = torch.tensor(extrinsics)
        extrinsics0_inv = torch.inverse(extrinsics[0])              # (4,4)
        extrinsics = torch.matmul(extrinsics, extrinsics0_inv.unsqueeze(0)) # (T,4,4)
        
        camparam = torch.cat([rearrange(intrinsics, 'n h w -> n (h w)'), rearrange(extrinsics, 'n h w -> n (h w)')], dim=-1)
        return camparam
        

    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            try:
                prompt_embedding, encoded_video, encoded_flow = self.load_data_pair(index)
                if self.use_camparam:
                    camparam = self.load_camparam(index)
                break
            except Exception as e:
                print(f"Error loading {self.prompt_embeddings[index]}: {str(e)}")
                index = random.randint(0, self.length - 1)
            
        image_path = self.images[index]
        prompt = self.prompts[index]
        hash_code = self.hash_code[index]
        
        _, image = self.preprocess(None, image_path)
        image = self.image_transform(image)
        

        # shape of encoded_video: [C, F, H, W]
        # shape and scale of image: [C, H, W], [-1,1]
        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "encoded_flow": encoded_flow if self.use_flow else "null",
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
            "camparam": camparam if self.use_camparam else 'null',
            "prompt": prompt,
            "hash_code": hash_code
        }
    
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image
    
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


def load_motion_generator_pipeline(backbone_path, ckpt_path, device, dtype):
    
    tokenizer = T5TokenizerFast.from_pretrained(backbone_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(backbone_path, subfolder="text_encoder")
    transformer = WanTransformer3DModel.from_pretrained(backbone_path, subfolder="transformer")
    vae = AutoencoderKLWan.from_pretrained(backbone_path, subfolder="vae")
    scheduler_tmp = FlowMatchEulerDiscreteScheduler.from_pretrained(backbone_path, shift=5.0, subfolder="scheduler")
    scheduler = UniPCMultistepScheduler.from_config(scheduler_tmp.config, flow_shift=5.0)
    
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location='cpu')['module']
        missing, unexpected = transformer.load_state_dict(sd)
        assert len(missing) == 0 and len(unexpected) == 0
    
    pipe = WanPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
    
    pipe = pipe.to(device, dtype)
    
    return pipe




def generate_video(
    # prompt: str,
    model_name: str,
    ckpt_path_motion: str,
    launcher: str,
    port: int,
    data_root: str,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 16,
    max_num_videos: int = None,
    use_dynamic_cfg: bool = False,
    ckpt_path: str = None,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """
    
    # Distributed
    local_rank = init_dist(launcher=launcher, port=port)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0
    
    torch.manual_seed(seed)
    random.seed(seed)
    
    if is_main_process:
        # os.makedirs(os.path.join(output_path, 'generated_videos'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'generated_flows'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'original_videos'), exist_ok=True)
        # os.makedirs(os.path.join(output_path, 'syn_flow_warped_videos'), exist_ok=True)
        # os.makedirs(os.path.join(output_path, 'flow_conditioned_generated_videos'), exist_ok=True)
        # os.makedirs(os.path.join(output_path, 'flow_conditioned_generated_flows'), exist_ok=True)
        # os.makedirs(os.path.join(output_path, 'input_flow_warped_videos'), exist_ok=True)


    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    model_name = model_name.lower()
    # desired_resolution = RESOLUTION_MAP[model_name]
    if 'tc2' in model_name:
        # Camera-Controlled Video Geenration.
        # Could be changed..
        desired_resolution = (704, 1280)
        use_camparam = True 
    else:
        desired_resolution = (704, 1280)
        use_camparam = False
    
    if width is None or height is None:
        height, width = desired_resolution
        logging.info(f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m")
    

    """
        # Prepare Dataset Class..
    """
    # image = load_image(image=image_or_video_path)
        
    # prompt
    # first image
    # camera parameters
    dataset = ImgFlowVideoDataset_Inference(
        max_num_frames=num_frames,
        height=height,
        width=width,
        data_root=data_root,
        max_num_videos=max_num_videos,
        use_camparam=use_camparam,
    )
    
    
    distributed_sampler = DistributedSampler(
        dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=False,
        seed=seed,
    )
    
    # DataLoaders creation:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    

    """
        # Prepare Pipeline
    """
    
    print(f'Constructing pipeline')
    pipe_motion_generator = load_motion_generator_pipeline(backbone_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers", ckpt_path=ckpt_path_motion, device=local_rank, dtype=dtype)
    
    print(f'Done loading pipeline')
    
    #--------------------------------------------------------------------------------------------------------


    dataloader.sampler.set_epoch(1)
    dist.barrier()
    
    original_video_path = os.path.join(output_path, 'original_videos')
    # output_video_path = os.path.join(output_path, 'generated_videos')
    output_flow_path = os.path.join(output_path, 'generated_flows')
    
    # output_syn_warped_video_path = os.path.join(output_path, 'syn_flow_warped_videos')
    # output_flow_cond_video_path = os.path.join(output_path, 'flow_conditioned_generated_videos')
    # output_flow_cond_flow_path = os.path.join(output_path, 'flow_conditioned_generated_flows')
    # output_inputflow_warped_video_path = os.path.join(output_path, 'input_flow_warped_videos')
    

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            orig_device = pipe_motion_generator.text_encoder.device
            pipe_motion_generator.text_encoder.to(local_rank, dtype=dtype)
            
            neg_prompt_embedding = get_t5_prompt_embeds(
                pipe_motion_generator.tokenizer,
                pipe_motion_generator.text_encoder,
                sample_neg_prompt,
                device=local_rank,
                dtype=dtype,
            )
            pipe_motion_generator.text_encoder.to(orig_device, dtype=dtype)
    
    
    
    data_iter = iter(dataloader)
    for step in tqdm(range(0, len(dataloader))):
        batch = next(data_iter)
        
        hash_code = batch["hash_code"][0]
        prompt = batch["prompt"][0]
        # image = batch["image"].to(local_rank)
        prompt_embedding = batch["prompt_embedding"].to(local_rank)
        # flow_latent = rearrange(batch["encoded_flow"].to(local_rank), 'b c f h w -> b f c h w') # BCFHW -> BFCHW
        prompt_short = prompt[:30].strip() + f"_{hash_code}"
        video_latent = rearrange(batch["encoded_video"].to(local_rank), 'b c f h w -> b f c h w')
        
        
        # image_torch_255 = ((image.detach().clone()+1)/2. * 255.).squeeze(0)
        
        # if step < 10:
        #     step += 1
        #     continue
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                video_original = decode_latents(video_latent.to(local_rank), pipe_motion_generator.vae)
                
                flow_latent_generate = pipe_motion_generator(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    prompt=None,
                    prompt_embeds=prompt_embedding,
                    negative_prompt_embeds=neg_prompt_embedding.to(local_rank),
                    generator=torch.Generator().manual_seed(seed),
                    num_inference_steps=num_inference_steps,
                    output_type='latent',
                    guidance_scale=guidance_scale,
                ).frames # BCFHW
                
                
                flow_generate = decode_flow_latents(pipe_motion_generator.vae.to(local_rank), rearrange(flow_latent_generate, 'b c f h w -> b f c h w').to(local_rank)) # B F C H W -> BF,C,H,W 

        
        # Save logs
        # 0) Original video
        save_path = os.path.join(original_video_path, f"{prompt_short}.mp4")
        video_original = (video_original[0].permute(0,2,3,1).to(torch.float32)/2 + 0.5).clip(0., 1.).cpu().numpy()
        export_to_video(video_original, save_path, fps=fps)
        
        
        # # 1) Synthesized video
        # save_path = os.path.join(output_video_path, f"{prompt_short}.mp4")
        # video_generate = (video_generate[0].permute(0,2,3,1).to(torch.float32)/2 + 0.5).clip(0., 1.).cpu().numpy()
        # export_to_video(video_generate, save_path, fps=fps)
        
        # 2) Synthesized flow
        save_path = os.path.join(output_flow_path, f"{prompt_short}.mp4")
        save_flow(flow_generate, filename=save_path, fps=fps)
        
        
        
        # # 3) Flow-Warped Video
        # save_path = os.path.join(output_syn_warped_video_path, f"{prompt_short}.mp4")
        # save_flow_warped_video(image_torch_255, flow_generate, filename=save_path, fps=fps)
        
        dist.barrier()
        
        step += 1


#--------------------------------------------------------------------------------------------------

def save_flow(flow, filename, fps=16):
    # flow: (BF)CHW, arbitrary scale
    flow_RGB = flow_to_color(flow) # BF,C,H,W (B=1)

    frame_list = []
    for frame in flow_RGB:
        frame = (frame.permute(1,2,0).float().detach().cpu().numpy()).astype(np.uint8).clip(0,255)
        frame_list.append(Image.fromarray(frame))
    
    export_to_video(frame_list, filename, fps=fps)
    

def decode_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    """
    Args.
        latents: BFCHW
    Return.
        video: video decoded from latents, BFCHW
    """
    
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    ) # BCFHW
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    ) # BCFHW
    latents = rearrange(latents, 'b f c h w -> b c f h w') / latents_std + latents_mean
    # torch.Size([1, 48, 31, 44, 80])
    video = vae.decode(latents, return_dict=False)[0]
    
    return rearrange(video, 'b c f h w -> b f c h w')


def plucker_embedding_from_camparam(camparam, H, W, device, flip_flag=None):
    # camparam: B, V, 25
    B, V, _ = camparam.shape
    
    K_norm = camparam[:,:,:9].view(B,-1,3,3)        # (B,T,3,3), Normalized K
    E = camparam[:,:,9:].view(B,-1,4,4)        # (B,T,4,4), W2C
    
    intrinsics = K_norm.detach().clone()
    intrinsics[:,:,0:1] = intrinsics[:,:,0:1] * (W)
    intrinsics[:,:,1:2] = intrinsics[:,:,1:2] * (H)
    
    c2w = torch.inverse(E)
    
    # B, V, 1
    fx = intrinsics[:,:,0:1,0]
    fy = intrinsics[:,:,1:2,1]
    cx = intrinsics[:,:,0:1,2]
    cy = intrinsics[:,:,1:2,2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    # fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    plucker = plucker.permute(0, 4, 1, 2, 3)                        # B, 6, V, H, W
    
    return plucker

def pad_text_embedding(embedding: torch.Tensor, text_embedding_len=512) -> torch.Tensor:
    """
    Args.
        embedding: (seq_len, C=4096)
    Output.
        embedding_padded: (seq_len_total, C=4096)
    """
    assert embedding.ndim == 2, embedding.shape[-1] == 4096
    
    return torch.cat([embedding, embedding.new_zeros(text_embedding_len - embedding.size(0), embedding.size(1))])
        

#--------------------------------------------------------------------------------------------------


# def encode_video(video: torch.Tensor, vae) -> torch.Tensor:
#     # shape of input video: [B, C, F, H, W]
#     video = video.to(vae.device, dtype=vae.dtype)
#     latent_dist = vae.encode(video).latent_dist
#     latent = latent_dist.sample() * vae.config.scaling_factor
#     return latent

# def encode_flow(flow, vae, flow_scale_factor):
#     # flow: BF,C,H,W
#     # flow_scale_factor [sf_x, sf_y]
#     assert flow.ndim == 4
#     num_frames, _, height, width = flow.shape

#     # Normalize optical flow
#     # ndim: 4 -> 5
#     flow = rearrange(flow, '(b f) c h w -> b f c h w', b=1)
#     flow_norm = adaptive_normalize(flow, flow_scale_factor[0], flow_scale_factor[1])

#     # ndim: 5 -> 4
#     flow_norm = rearrange(flow_norm, 'b f c h w -> (b f) c h w', b=1)

#     # Duplicate mean value for third channel
#     num_frames, _, H, W = flow_norm.shape
#     flow_norm_extended = torch.empty((num_frames, 3, height, width)).to(flow_norm)
#     flow_norm_extended[:,:2] = flow_norm
#     flow_norm_extended[:,-1:] = flow_norm.mean(dim=1, keepdim=True)
#     flow_norm_extended = rearrange(flow_norm_extended, '(b f) c h w -> b c f h w', f=num_frames)

#     return encode_video(flow_norm_extended, vae)


# def decode_flow(flow_latent, vae, flow_scale_factor):
#     flow_latent = flow_latent.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
#     flow_latent = 1 / vae.config.scaling_factor * flow_latent
    
#     flow = vae.decode(flow_latent).sample # BCFHW

#     # discard third channel (which is a mean value of f_x and f_y)
#     flow = flow[:,:2].detach().clone()

#     # Unnormalize optical flow
#     flow = rearrange(flow, 'b c f h w -> b f c h w')
#     flow = adaptive_unnormalize(flow, flow_scale_factor[0], flow_scale_factor[1])

#     flow = rearrange(flow, 'b f c h w -> (b f) c h w')
#     return flow # BF,C,H,W

# def adaptive_normalize(flow, sf_x, sf_y):
#     # x: BFCHW, optical flow
#     assert flow.ndim == 5, 'Set the shape of the flow input as (B, F, C, H, W)'
#     assert sf_x is not None and sf_y is not None
#     b, f, c, h, w = flow.shape
    
#     max_clip_x = math.sqrt(w/sf_x) * 1.0
#     max_clip_y = math.sqrt(h/sf_y) * 1.0
    
#     flow_norm = flow.detach().clone()
#     flow_x = flow[:, :, 0].detach().clone()
#     flow_y = flow[:, :, 1].detach().clone()
    
#     flow_x_norm = torch.sign(flow_x) * torch.sqrt(torch.abs(flow_x)/sf_x + 1e-7)
#     flow_y_norm = torch.sign(flow_y) * torch.sqrt(torch.abs(flow_y)/sf_y + 1e-7)

#     flow_norm[:, :, 0] = torch.clamp(flow_x_norm, min=-max_clip_x, max=max_clip_x)
#     flow_norm[:, :, 1] = torch.clamp(flow_y_norm, min=-max_clip_y, max=max_clip_y)

#     return flow_norm


# def adaptive_unnormalize(flow, sf_x, sf_y):
#     # x: BFCHW, optical flow
#     assert flow.ndim == 5, 'Set the shape of the flow input as (B, F, C, H, W)'
#     assert sf_x is not None and sf_y is not None
    
#     flow_orig = flow.detach().clone()
#     flow_x = flow[:, :, 0].detach().clone()
#     flow_y = flow[:, :, 1].detach().clone()
    
#     flow_orig[:, :, 0] = torch.sign(flow_x) * sf_x * (flow_x**2 - 1e-7)
#     flow_orig[:, :, 1] = torch.sign(flow_y) * sf_y * (flow_y**2 - 1e-7)
    
#     return flow_orig

#--------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    # parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument("--model_name", type=str, required=True, help="Model Name, [t2v, t2f, t2fv, tc2v, tc2fv]",)
    parser.add_argument("--ckpt_path_motion", type=str, required=True, help="T2F path",)
    parser.add_argument("--image_path", type=str, default=None, help="The path of the image to be used as the background of the video",)
    parser.add_argument("--data_root", type=str, required=True, help="The path of the dataset root",)
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument("--height", type=int, default=None, help="The height of the generated video")
    parser.add_argument("--fps", type=int, default=24, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--max_num_videos", type=int, default=None, help="# of videos for inference")
    parser.add_argument("--use_dynamic_cfg", action='store_true')
    parser.add_argument("--ckpt_path", type=str, default=None)
    
    # DDP args
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of the distributed processes.")
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Replica rank on the current node. This field is required '
                             'by `torch.distributed.launch`.')
    parser.add_argument("--global_seed", default=42, type=int,
                        help="seed")
    parser.add_argument("--port", type=int)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")


    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    
    generate_video(
        # prompt=args.prompt,
        model_name=args.model_name,
        ckpt_path_motion=args.ckpt_path_motion,
        launcher=args.launcher,
        port=args.port,
        data_root=args.data_root,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_path=args.image_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        fps=args.fps,
        max_num_videos=args.max_num_videos,
        use_dynamic_cfg=args.use_dynamic_cfg,
        ckpt_path=args.ckpt_path,
    )

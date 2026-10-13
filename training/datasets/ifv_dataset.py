import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import json
import random
import os
import numpy as np
from einops import rearrange, repeat

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from training.constants import LOG_LEVEL, LOG_NAME

from .utils import (
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)

import pdb

if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class ImgFlowVideoDataset(Dataset):
    """
    A dataset class for (image,flow)-to-video generation or image-to-flow_video that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(
        self, 
        max_num_frames: int, 
        height: int, 
        width: int, 
        data_root: str,
        device: torch.device,
        trainer: "Trainer" = None,
        do_forced_filtering: bool = False,
        use_camparam: bool = False,
        *args, 
        **kwargs
    ) -> None:
        data_root = Path(data_root)
        
        if do_forced_filtering == False and trainer.args.rebuttal_sample_100:
            metadata_path = data_root / "metadata_100_sampled.jsonl"
        elif do_forced_filtering == False and trainer.args.rebuttal_sample_200:
            metadata_path = data_root / "metadata_200_sampled.jsonl"
        else:
            metadata_path = data_root / "metadata.jsonl"
        self.use_camparam = use_camparam
        
        assert metadata_path.is_file(), "For this dataset type, you need metadata.jsonl in the root path"
        
        # Load metadata
        # metadata = {
        #     "video_path": ...,
        #     "hash_code": ...,
        #     "prompt": ...,
        # }
        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append( json.loads(line) )
        
        if hasattr(trainer, "filter_metadata"):
            use_cycle_filtering = trainer.args.use_cycle_filtering
            filtering_threshold = trainer.args.filtering_threshold
            filtering_threshold_low = trainer.args.filtering_threshold_low
            
            if do_forced_filtering:
                # Hard coding.. Just use this now.
                use_cycle_filtering = True
                filtering_threshold = 90
                filtering_threshold_low = 0
                
            metadata = trainer.filter_metadata(metadata, use_cycle_filtering=use_cycle_filtering, thres=filtering_threshold, thres_low=filtering_threshold_low)

        self.prompts = [x["prompt"] for x in metadata]
        
        prompt_embedding_dir_path = "prompt_embeddings"
        # flow_embedding_dir_path = "flow_latent"
        # if trainer.args.scale_factor is not None:
        #     assert isinstance(trainer.args.scale_factor, float), "Use scale factor of float!"
        #     flow_embedding_dir_path = f"flow_latent_{trainer.args.scale_factor}_WAFT"
        flow_embedding_dir_path = f"flow_latent_WAFT"
        camparam_dir_path = f"cam_param"

        self.prompt_embeddings = [data_root / prompt_embedding_dir_path / (x["hash_code"] + '.safetensors') for x in metadata]
        self.videos = [data_root / "video_latent" / (x["hash_code"] + '.safetensors') for x in metadata]
        self.images = [data_root / "first_frames" /  (x["hash_code"] + '.png') for x in metadata]
        self.flows = [data_root / flow_embedding_dir_path/ (x["hash_code"] + '.safetensors') for x in metadata]
        if use_camparam:
            self.camparam = [data_root / camparam_dir_path/ (x["hash_code"] + '.npz') for x in metadata]
        

        self.trainer = trainer

        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        
        
        
        # Check if number of prompts matches number of videos and images
        if not (len(self.videos) == len(self.prompts) == len(self.images) == len(self.flows)):
            raise ValueError(
                f"Expected length of prompts, videos and images to be the same but found {len(self.prompts)=}, {len(self.videos)=}, {len(self.images)=} and {len(self.flows)=}. Please ensure that the number of caption prompts, videos and images match in your dataset."
            )

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms
        
        self.length = len(self.videos)

        print(f"Dataset size: {self.length}")
        
        # self.__getitem__(0)
        # for _ in range(1000):
        #     batch = self.__getitem__(random.randint(0, self.length-1))
        #     camparam = batch["camparam"]
            
        #     print(camparam[:,9:].reshape(-1,4,4)[:,:3,-1])
        #     pdb.set_trace()


    def __len__(self) -> int:
        return self.length
    
    def load_data_pair(self, index):
        # prompt = self.prompts[index]
        prompt_embedding_path = self.prompt_embeddings[index]
        encoded_video_path = self.videos[index]
        encoded_flow_path = self.flows[index]
        
        prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"] 
        encoded_video = load_file(encoded_video_path)["video_latent"] # CFHW
        encoded_flow = load_file(encoded_flow_path)["flow_latent"] # CFHW
        
        prompt_embedding = self.trainer.pad_text_embedding(prompt_embedding)
        encoded_video = rearrange(encoded_video, 'f c h w -> c f h w') # FCHW
        encoded_flow = rearrange(encoded_flow, 'f c h w -> c f h w') # FCHW
        
        assert encoded_flow.shape == (48, 31, 44, 80) and encoded_video.shape == (48, 31, 44, 80) and prompt_embedding.shape == (512, 4096), f"For Wan-TI2V-5B, we use 121 frames, which would be 31 frames when encoded by VAE. || flow : {encoded_flow.shape} | video : {encoded_video.shape} | text : {prompt_embedding.shape} ||"
        
        return prompt_embedding, encoded_video, encoded_flow
    
    def load_camparam(self, index):
        camparam_path = self.camparam[index]
        camparam_all = np.load(camparam_path)
        intrinsics, extrinsics = camparam_all['intrinsics'], camparam_all['extrinsics'] # THW
        if intrinsics.ndim == 2:
            intrinsics = repeat(camparam_all['intrinsics'], 'h w -> n h w', n=camparam_all['extrinsics'].shape[0])
        elif intrinsics.ndim ==3:
            # Blender data.
            # Normalize intrinsics
            if intrinsics[:,0] > 3.0:   # if not normalized so far, normalize.
                intrinsics[:,0] = intrinsics[:,0] / self.width
                intrinsics[:,1] = intrinsics[:,1] / self.height
            
        intrinsics = torch.tensor(intrinsics)
        extrinsics = torch.tensor(extrinsics)
        extrinsics0_inv = torch.inverse(extrinsics[0])              # (4,4)
        extrinsics = torch.matmul(extrinsics, extrinsics0_inv.unsqueeze(0)) # (T,4,4)
        
        camparam = torch.cat([rearrange(intrinsics, 'n h w -> n (h w)'), rearrange(extrinsics, 'n h w -> n (h w)')], dim=-1)
        return camparam
        
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # prompt_embedding, encoded_video, encoded_flow = self.load_data_pair(index)
        # if self.use_camparam:
        #     camparam = self.load_camparam(index)

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
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
        
        _, image = self.preprocess(None, image_path)
        image = self.image_transform(image)
        

        # shape of encoded_video: [C, F, H, W]
        # shape and scale of image: [C, H, W], [-1,1]
        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "encoded_flow": encoded_flow,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
            "camparam": camparam if self.use_camparam else None,
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
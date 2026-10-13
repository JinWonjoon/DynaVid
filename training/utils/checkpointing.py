import os
from pathlib import Path
from typing import Tuple
from datetime import datetime

from accelerate.logging import get_logger

from training.constants import LOG_LEVEL, LOG_NAME

from ..utils.file_utils import delete_files, find_files

import pdb

logger = get_logger(LOG_NAME, LOG_LEVEL)


def get_latest_ckpt_path_to_resume_from(
    resume_from_checkpoint: str | None, num_update_steps_per_epoch: int
) -> Tuple[str | None, int, int, int]:
    
    """
    resume_from_checkpoint: path to log directory (i.e., output_dir)
    """
    
    # import torch.distributed as dist
    # if dist.get_rank() == 0:
    #     pdb.set_trace()
    # dist.barrier()
    if resume_from_checkpoint is not None:
        exp_list = sorted([d for d in resume_from_checkpoint.iterdir() if d.is_dir()], reverse=False)
        if len(exp_list) < 2:
            ckpt_list = []
        else:
            exp_latest = exp_list[-2] # [-1] denotes current one.
            
            ckpt_list = sorted([d for d in exp_latest.iterdir() if d.is_dir()], reverse=False)
            ckpt_list = [d for d in ckpt_list if 'checkpoint' in str(d)]
    else:
        ckpt_list = []
    
    if len(ckpt_list) == 0:
        resume_from_checkpoint = None
    else:
        resume_from_checkpoint = ckpt_list[-1]
    
    if resume_from_checkpoint is None:
        initial_global_step = 0
        global_step = 0
        first_epoch = 0
        resume_from_checkpoint_path = None
    else:
        resume_from_checkpoint_path = Path(resume_from_checkpoint)
        if not resume_from_checkpoint_path.exists():
            logger.info(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            initial_global_step = 0
            global_step = 0
            first_epoch = 0
            resume_from_checkpoint_path = None
        else:
            logger.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            global_step = int(resume_from_checkpoint_path.name.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    return resume_from_checkpoint_path, initial_global_step, global_step, first_epoch


def get_intermediate_ckpt_path(checkpointing_limit: int, step: int, output_dir: str) -> str:
    # before saving state, check if this save would set us over the `checkpointing_limit`
    if checkpointing_limit is not None:
        checkpoints = find_files(output_dir, prefix="checkpoint")

        # before we save the new checkpoint, we need to have at_most `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpointing_limit:
            num_to_remove = len(checkpoints) - checkpointing_limit + 1
            checkpoints_to_remove = checkpoints[0:num_to_remove]
            delete_files(checkpoints_to_remove)

    logger.info(f"Checkpointing at step {step}")
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    logger.info(f"Saving state to {save_path}")
    return save_path

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Union

from accelerate.logging import get_logger

from training.constants import LOG_LEVEL, LOG_NAME


logger = get_logger(LOG_NAME, LOG_LEVEL)


def find_files(dir: Union[str, Path], prefix: str = "checkpoint") -> List[str]:
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.exists():
        return []
    checkpoints = os.listdir(dir.as_posix())
    checkpoints = [c for c in checkpoints if c.startswith(prefix)]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    checkpoints = [dir / c for c in checkpoints]
    return checkpoints


def delete_files(dirs: Union[str, List[str], Path, List[Path]]) -> None:
    if not isinstance(dirs, list):
        dirs = [dirs]
    dirs = [Path(d) if isinstance(d, str) else d for d in dirs]
    logger.info(f"Deleting files: {dirs}")
    for dir in dirs:
        if not dir.exists():
            continue
        shutil.rmtree(dir, ignore_errors=True)


# def delete_files(dirs: Union[str, List[str], Path, List[Path]]) -> None:
#     if not isinstance(dirs, list):
#         dirs = [dirs]
#     dirs = [Path(d) if isinstance(d, str) else d for d in dirs]
#     dirs = [d / 'pytorch_model' for d in dirs]
    
#     # Remove all weights except checkpoint
#     logger.info(f"Deleting files: {dirs}")
#     for dir in dirs:
#         if not dir.exists():
#             continue
#         # shutil.rmtree(dir, ignore_errors=True)
        
#         for root_, dir_, files_ in os.walk(dir):
#             for file_ in files_:
#                 if not ('model_states.pt' in file_):
#                     file_path_ = os.path.join(root_, file_)
#                     try:
#                         os.remove(file_path_)
#                     except:
#                         pass
    


def string_to_filename(s: str) -> str:
    return (
        s.replace(" ", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace(".", "-")
        .replace(",", "-")
        .replace(";", "-")
        .replace("!", "-")
        .replace("?", "-")
    )

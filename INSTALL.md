# Installation and Code Layout

DynaVid builds on Wan2.2 and uses PyTorch, Diffusers, and VACE-based motion control. Use a Linux environment with an NVIDIA GPU, a compatible CUDA toolkit, and Python 3.10 or newer. Model weights and datasets are not included.

## Installation

Install a CUDA-compatible PyTorch build first, then install the runtime dependencies and package:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

For training, also install the training dependencies:

```bash
pip install -e '.[train]'
```

Flash Attention requires a compatible CUDA toolchain. If its isolated build fails, install it separately with `pip install flash_attn==2.7.3 --no-build-isolation`.

## Code Layout

- `train.py` and `training/`: training entry point, datasets, models, pipelines, and training configurations.
- `inference/`: motion generation, motion-guided video generation, and sequential human/camera generation.
- `analysis/`: analysis scripts used during development.
- `wan/`, `generate.py`, `examples/`, and `tests/`: inherited Wan components, examples, and GPU smoke tests.

Existing shell scripts in `training/training_scripts/` and `inference/scripts/` retain experiment settings. Set dataset, model, checkpoint, and output paths for your environment before running them. They reference local research paths and do not download DynaVid checkpoints automatically. The scripts are research configurations, rather than a turnkey reproduction package.

## Third-Party Code

The Wan implementation and its Apache 2.0 license are retained in `wan/` and `LICENSE.txt`.

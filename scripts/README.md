# Scripts

This folder contains some scripts

## `generate_requirements_rtd.py`

The documentation, built by [Read the Docs (RTD)](https://readthedocs.org/), first 
installs `docs/requirements-rtd.txt` before installing `pystiche`. This has two 
reasons:

1. Installing PyTorch distributionswith CUDA support exceeds their memory limit. Thus, 
   we need to make sure to install it with CPU support only. 
2. The additional dependencies to build the documentation only live in `tox.ini`. Thus, 
   we need to extract them.

This script automatically populates `docs/requirements-rtd.txt`. It requires

- `pyyaml` and
- `light-the-torch>=0.2`

to be installed.

## `perform_model_optimization`

Trainings script for the model-optimization example. Requires the root directory of the 
dataset used for training as positional argument.

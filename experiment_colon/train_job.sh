#!/bin/bash
#SBATCH -p ls6
#SBATCH --gres=gpu:1

python tools/train.py configs/swin-b_vpt/1-shot_colon.py "$@"

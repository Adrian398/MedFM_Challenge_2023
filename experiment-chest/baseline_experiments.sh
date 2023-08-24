#!/bin/bash
# Activate the virtual environment

source /home/ls6/hekalo/Git/medfm-challenge/venv/bin/activate

# export python path
export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/hekalo/job_output/medfm-%j.out"

# experiment numbers
exp_num=1
exp_suffix="baseline_test"

configs=(
  "configs/clip-b_vpt/1-shot_chest.py"
  "configs/clip-b_vpt/5-shot_chest.py"
  "configs/clip-b_vpt/10-shot_chest.py"
  "configs/dinov2-b_vpt/1-shot_chest.py"
  "configs/dinov2-b_vpt/5-shot_chest.py"
  "configs/dinov2-b_vpt/10-shot_chest.py"
  "configs/eva-b_vpt/1-shot_chest.py"
  "configs/eva-b_vpt/5-shot_chest.py"
  "configs/eva-b_vpt/10-shot_chest.py"
  "configs/swin-b_vpt/1-shot_chest.py"
  "configs/swin-b_vpt/5-shot_chest.py"
  "configs/swin-b_vpt/10-shot_chest.py"
  "configs/vit-b_vpt/1-shot_chest.py"
  "configs/vit-b_vpt/5-shot_chest.py"
  "configs/vit-b_vpt/10-shot_chest.py"
)

configs_v2=(
  "configs/swinv2-b/1-shot_chest.py"
  "configs/swinv2-b/5-shot_chest.py"
  "configs/swinv2-b/10-shot_chest.py"
)

for config in "${configs[@]}"; do
  sbatch -p ls6 --gres=gpu:1 --nodelist=gpu8a --wrap="python tools/train.py $config --exp_num=$exp_num --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output
done

# for config_v2 in "${configs_v2[@]}"; do
#   sbatch -p ls6 --gres=gpu:rtx3090:1 --wrap="python tools/train.py $config_v2 --exp_num=$exp_num --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output
# done

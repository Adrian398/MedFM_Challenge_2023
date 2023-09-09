#!/bin/bash
# Activate the virtual environment

# export python path
# export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/krenzer/job_output/slurm-%j.out"

# experiment numbers
exp_num=1
exp_suffix="baseline_test"

configs=(
  "configs/clip-b_vpt/1-shot_endo.py"
  "configs/clip-b_vpt/5-shot_endo.py"
  "configs/clip-b_vpt/10-shot_endo.py"
  "configs/dinov2-b_vpt/1-shot_endo.py"
  "configs/dinov2-b_vpt/5-shot_endo.py"
  "configs/dinov2-b_vpt/10-shot_endo.py"
  "configs/eva-b_vpt/1-shot_endo.py"
  "configs/eva-b_vpt/5-shot_endo.py"
  "configs/eva-b_vpt/10-shot_endo.py"
  #"configs/swin-b_vpt/1-shot_endo.py"
  #"configs/swin-b_vpt/5-shot_endo.py"
  #"configs/swin-b_vpt/10-shot_endo.py"
)

configs_v2=(
  #"configs/swinv2-b/1-shot_endo.py"
  #"configs/swinv2-b/5-shot_endo.py"
  #"configs/swinv2-b/10-shot_endo.py"
)

for config in "${configs[@]}"; do
  print(config)
  sbatch -p ls6 --gres=gpu:1 --nodelist=gpu8a --wrap="python tools/train.py $config --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output
done

# for config_v2 in "${configs_v2[@]}"; do
#   sbatch -p ls6 --gres=gpu:rtx3090:1 --wrap="python tools/train.py $config_v2 --exp_num=$exp_num --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output
# done
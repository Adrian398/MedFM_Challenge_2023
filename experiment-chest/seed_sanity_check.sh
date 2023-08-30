#!/bin/bash
# Activate the virtual environment

source /home/ls6/hekalo/Git/medfm-challenge/venv/bin/activate

# export python path
export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/hekalo/job_output/medfm-%j.out"

# experiment numbers
exp_num=1
exp_suffix="seed_sanity_check"

configs=(
  "configs/clip-b_vpt/1-shot_chest.py"
  "configs/vit-b_vpt/1-shot_chest.py"
  "configs/swin-b_vpt/1-shot_chest.py"
  "configs/swinv2-t/1-shot_chest.py"
)

swinv2_config="configs/swinv2-b/1-shot_chest.py"

for config in "${configs[@]}"; do
  sbatch -p ls6 --gres=gpu:1 --nodelist=gpu8a --wrap="python tools/train.py $config --exp_num=$exp_num --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output
done

sbatch -p ls6 --gres=gpu:rtx3090:1 --wrap="python tools/train.py $swinv2_config --exp_num=$exp_num --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output

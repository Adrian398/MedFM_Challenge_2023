#!/bin/bash
# Activate the virtual environment

source /home/ls6/hekalo/Git/medfm-challenge/venv/bin/activate

# export python path
export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/hekalo/job_output/medfm-%j.out"

# experiment numbers
exp_num=1
exp_suffix="swinv2_lr_test"

#configs=(
#  "configs/swinv2-b/1-shot_chest.py"
#  "configs/swinv2-b/5-shot_chest.py"
#  "configs/swinv2-b/10-shot_chest.py"
#)
config="configs/swinv2-b/1-shot_chest.py"

lrs=(
  5e-3
  5e-4
  5e-5
)

for lr in "${lrs[@]}"; do
  sbatch -p ls6 --gres=gpu:rtx3090:1 --wrap="python tools/train.py $config --exp_num=$exp_num --remove_timestamp --lr=$lr --exp_suffix=$exp_suffix" -o $log_output
done

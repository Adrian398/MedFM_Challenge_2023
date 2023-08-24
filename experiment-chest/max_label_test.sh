#!/bin/bash
# Activate the virtual environment

source /home/ls6/hekalo/Git/medfm-challenge/venv/bin/activate

# export python path
export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/hekalo/job_output/slurm-%j.out"

# experiment numbers
exp_nums=(
20
21
)

exp_suffix="min_max_label_test"

config="configs/clip-b_vpt/10-shot_chest.py"
#configs=(
#  "configs/clip-b_vpt/1-shot_chest.py"
#  "configs/clip-b_vpt/5-shot_chest.py"
#  "configs/clip-b_vpt/10-shot_chest.py"
#)

for exp_num in "${exp_nums[@]}"; do
  sbatch -p ls6 --gres=gpu:1 --nodelist=gpu8a --wrap="python tools/train.py $config --exp_num=$exp_num --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output
done

# for config_v2 in "${configs_v2[@]}"; do
#   sbatch -p ls6 --gres=gpu:rtx3090:1 --wrap="python tools/train.py $config_v2 --exp_num=$exp_num --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output
# done
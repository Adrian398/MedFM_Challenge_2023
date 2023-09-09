#!/bin/bash
# Activate the virtual environment

source /home/ls6/hekalo/Git/medfm-challenge/venv/bin/activate

# export python path
export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/hekalo/job_output/medfm-%j.out"

# experiment numbers
exp_nums=(
1
2
3
4
5
)

exp_suffix="new_train_pipeline"

configs=(
  "configs/swin-b_vpt/1-shot_chest.py"
  "configs/swin-b_vpt/5-shot_chest.py"
  "configs/swin-b_vpt/10-shot_chest.py"
)

for config in "${configs[@]}"; do
  for exp_num in "${exp_nums[@]}"; do
    sbatch -p ls6 --gres=gpu:1 --nodelist=gpu8a --wrap="python tools/train.py $config --exp_num=$exp_num --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output
  done
done

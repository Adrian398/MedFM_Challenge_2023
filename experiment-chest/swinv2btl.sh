#!/bin/bash
# Activate the virtual environment

source /home/ls6/hekalo/Git/medfm-challenge/venv/bin/activate

# export python path
export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/hekalo/job_output/medfm-%j.out"

# experiment numbers
exp_nums=(1 2 3 4 5)
exp_suffix="swinv2_btl_seed3"
batch_size=2
seed=3

configs=(
  "configs/swinv2-b/1-shot_chest.py"
  "configs/swinv2-b/5-shot_chest.py"
  "configs/swinv2-b/10-shot_chest.py"
  "configs/swinv2-t/1-shot_chest.py"
  "configs/swinv2-t/5-shot_chest.py"
  "configs/swinv2-t/10-shot_chest.py"
  "configs/swinv2-l/1-shot_chest.py"
  "configs/swinv2-l/5-shot_chest.py"
  "configs/swinv2-l/10-shot_chest.py"
)

for config in "${configs[@]}"; do
  for exp_num in "${exp_nums[@]}"; do
    sbatch -p ls6 --gres=gpu:rtx3090:1 --wrap="python tools/train.py $config --seed=$seed --exp_num=$exp_num --exp_suffix=$exp_suffix --train_bs=$batch_size" -o $log_output
  done
done

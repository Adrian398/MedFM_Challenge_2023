#!/bin/bash
# Activate the virtual environment

source /home/ls6/hekalo/Git/medfm-challenge/venv/bin/activate

# export python path
export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/hekalo/job_output/medfm-%j.out"

# experiment numbers
exp_nums=(1 2 3 4 5)
exp_suffix="seed_1"
batch_size=4
seed=1

configs=(
  "configs/densenet121/1-shot_chest.py"
  "configs/densenet121/5-shot_chest.py"
  "configs/densenet121/10-shot_chest.py"
)

for config in "${configs[@]}"; do
  for exp_num in "${exp_nums[@]}"; do
    sbatch -p ls6 --gres=gpu:1 --nodelist=gpu8a --wrap="python tools/train.py $config --seed=$seed --exp_num=$exp_num --exp_suffix=$exp_suffix --train_bs=$batch_size" -o $log_output
  done
done

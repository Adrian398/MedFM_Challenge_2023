#!/bin/bash
# Activate the virtual environment

source /home/ls6/hekalo/Git/medfm-challenge/venv/bin/activate

# export python path
export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/hekalo/job_output/medfm-%j.out"

# experiment numbers
exp_num=1
exp_suffix="swinv2_tests"
batch_size=(1 2 4 8 16)
seeds=(0 1 2 3 4 5)

configs=(
  "configs/swinv2-b/1-shot_chest.py"
  "configs/swinv2-t/1-shot_chest.py"
  "configs/swinv2-l/1-shot_chest.py"
)

for config in "${configs[@]}"; do
  for seed in "${seeds[@]}"; do
    for bs in "${batch_size[@]}"; do
      sbatch -p ls6 --gres=gpu:1 --nodelist=gpu1a --wrap="python tools/train.py $config --seed=$seed --exp_num=$exp_num --exp_suffix=$exp_suffix --train_bs=$bs" -o $log_output
    done
  done
done

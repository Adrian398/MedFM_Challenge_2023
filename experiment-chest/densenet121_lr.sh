#!/bin/bash
# Activate the virtual environment

source /home/ls6/hekalo/Git/medfm-challenge/venv/bin/activate

# export python path
export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/hekalo/job_output/densenetlr-%j.out"

# experiment numbers
exp_num=1
exp_suffix="densenet121_lr"
seed=1337
lrs=(1e-4 3e-5 1e-5 3e-6 1e-6 3e-7 1e-7)

config="configs/densenet121/1-shot_chest.py"

for lr in "${lrs[@]}"; do
  sbatch -p ls6 --gres=gpu:1 --nodelist=gpu8a --wrap="python tools/train.py $config --seed=$seed --exp_num=$exp_num --exp_suffix=$exp_suffix --lr=$lr" -o $log_output
done

#!/bin/bash
# Activate the virtual environment

source /home/ls6/hekalo/Git/medfm-challenge/venv/bin/activate

# export python path
export PYTHONPATH="$PWD:$PYTHONPATH"

log_output="/home/ls6/hekalo/job_output/medfm-%j.out"

# experiment numbers
exp_num=1
exp_suffix="ensemble_soup_test"
seed=1337
lr=1e-6

configs=(
  "configs/clip-b_vpt/5-shot_chest.py"
  "configs/swin-b_vpt/5-shot_chest.py"
  "configs/dinov2-b_vpt/5-shot_chest.py"
)

configs_2=(
  "configs/swinv2-b/5-shot_chest.py"
  "configs/resnet101/5-shot_chest.py"
)

batch_sizes=(1 2 4 8)

for config in "${configs[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    sbatch -p ls6 --gres=gpu:1 --nodelist=gpu8a --wrap="python tools/train.py $config --lr=$lr --train_bs=$batch_size --seed=$seed --exp_num=$exp_num --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output
  done
done

for config_2 in "${configs_2[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    sbatch -p ls6 --gres=gpu:rtx3090:1 --wrap="python tools/train.py $config_2 --train_bs=$batch_size --lr=$lr --seed=$seed --exp_num=$exp_num --remove_timestamp --exp_suffix=$exp_suffix" -o $log_output
  done
done


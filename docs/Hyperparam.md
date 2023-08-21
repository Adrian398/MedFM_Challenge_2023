# Scheduler Overview
ConstantLR, LinearLR, ExponentialLR, ...
https://mmengine.readthedocs.io/en/latest/api/optim.html

```
sbatch -p ls6 --gres=gpu:1 --pty python tools/train.py configs/swin-b_vpt/1-shot_colon.py --lr 0.01
&& sbatch -p ls6 --gres=gpu:1 --pty python tools/train.py configs/swin-b_vpt/1-shot_colon.py --lr 0.001
&& sbatch -p ls6 --gres=gpu:1 --pty python tools/train.py configs/swin-b_vpt/1-shot_colon.py --lr 0.0001
```

```
sbatch experiment_colon/train_job.sh --lr 0.0001
```

sbatch -p ls6 --gres=gpu:rtx4090:1 --wrap="python tools/train.py configs/swin-b_vpt/1-shot_colon.py --lr 0.001"


srun -p ls6 --gres=gpu:1 --pty "python tools/train.py configs/swin-b_vpt/1-shot_colon.py --lr 0.001"

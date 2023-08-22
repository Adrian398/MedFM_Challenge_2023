
## General:

- Validation / saving checkpoints takes a lot of time, adjust ``val_interval`` in ``train_cfg`` and ``interval`` in default_hooks/checkpoint accordingly


- When using cosine annealing (default in swin schedule), the learning rate decrease will orient itself on ``max_epochs``, i.e. larger max_epoch => slower decrease of LR

## For Swin + Colon 1-shot:

- Adding the following Augmentation did not improve results 
````python
dict(type='RandomFlip', prob=0.5, direction='vertical'),
dict(type='ColorJitter', hue=0.1, brightness=0.2, contrast=0.2, saturation=0.2)
#Same for higher values like:
dict(type='ColorJitter', hue=0.3, brightness=0.4, contrast=0.4, saturation=0.4),
````
	
 - on ``gpu1c``, (4090), validation batch size of ``128`` works, ``256`` => out of memory


- in config, uncommment ``optim_wrapper=...`` so that Adam optimizer is used, as defined in 
	swin_schedule, which is imported at the top => training more stable


- PromptedSwinTransformer parameter ``window_size`` can't be changed without changing other parameters like padding


- Changing visual prompt length ``vpl`` from 5 to 50 => no noticeable difference

### Interval combos
which work well for colon for certain batch sizes (more data in higher n-shot => decrease val interval since more learning happens in less epochs):

- 1-shot, BS 8, val 250 or 125
- 5-shot, BS 8, val 50 or 25
    

# Chest Findings

## First Experiment Runs

### Setup

- all parameters kept the same for 1-shot, 5-shot, 10-shot
- epoch number to 2000 with cosine scaling learning rate
- kept all other default parameters
- used experiment 2
- used Swin and ViT
- used some standard approaches for data augmentation:

````python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='NumpyToPIL', to_rgb=True),
    dict(type='torchvision/RandomAffine', degrees=(-15, 15), translate=(0.05, 0.05), fill=128),
    dict(type='PILToNumpy', to_bgr=True),
    dict(
        type='RandomResizedCrop',
        scale=384,
        crop_ratio_range=(0.9, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
````

### Results

- ViT performs a lot better for 1-shot and 10-shot than Swin
- on 5-shot performance is similar
- best epochs vary a lot, but the higher the n-shot, the higher the best epoch --> longer training necessary
- mAP went continuously up towards the end despite not being best epoch anymore --> keep training?
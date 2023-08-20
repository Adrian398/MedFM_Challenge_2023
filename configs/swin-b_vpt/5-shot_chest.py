_base_ = [
    '../datasets/chest.py',
    '../swin_schedule.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 5e-2
n = 1
vpl = 5
dataset = 'chest'
exp_num = 1
nshot = 5
run_name = f'in21k-swin-b_vpt-{vpl}_bs4_lr{lr}_{nshot}-shot_{dataset}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedSwinTransformer',
        prompt_length=vpl,
        arch='base',
        img_size=384,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth',
            prefix='backbone',
        ),
        stage_cfgs=dict(block_cfgs=dict(window_size=12))),
    neck=None,
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=1024,
    ))

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

train_dataloader = dict(
    batch_size=4, 
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
        pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=8,  
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
)

test_dataloader = dict(
    batch_size=4,  
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt'),
)

optim_wrapper = dict(optimizer=dict(lr=lr))

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=50),
)

work_dir = f'work_dirs/exp{exp_num}/{run_name}'

from configs.chest_config import *


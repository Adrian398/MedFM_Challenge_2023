_base_ = [
    '../datasets/chest.py',
    '../schedules/chest.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 1e-3
vpl = 1  
dataset = 'chest'
exp_num = 1
nshot = 10

run_name = f'eva02-b_{vpl}_bs4_lr{lr}_{nshot}-shot_{dataset}'
work_dir = f'work_dirs/chest/{nshot}-shot/{run_name}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedViTEVA02',
        prompt_length=vpl,
        patch_size=14,
        sub_ln=True,
        final_norm=False,
        out_type='avg_featmap',
        arch='b',
        img_size=448,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-base-p14_in21k-pre_in21k-medft_3rdparty_in1k-448px_20230505-5cd4d87f.pth',
            prefix='backbone',),
        ),
    neck=None,
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=768,
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=448,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]


train_dataloader = dict(
    batch_size=1, 
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
    pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=2,  
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt',
    pipeline=test_pipeline),
)

test_dataloader = dict(
    batch_size=2,  
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt',
    pipeline=test_pipeline),
)

optim_wrapper = dict(optimizer=dict(lr=lr))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=50),
)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

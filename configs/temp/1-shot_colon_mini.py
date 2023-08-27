_base_ = [
    '../datasets/colon.py',
    '../swin_schedule.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 5e-3
train_bs = 8
dataset = 'colon'
model_name = 'swin'
exp_num = 2
nshot = 1

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/temp/{nshot}-shot/{run_name}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedSwinTransformer',
        prompt_length=5,
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
        type='LinearClsHead',
        num_classes=2,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=384,
        backend='pillow',
        interpolation='bicubic'
    ),
    dict(
        type='Normalize',
        mean=bgr_mean,
        std=bgr_std,
        to_rgb=False
    ),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    num_workers=16,
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
        pipeline=train_pipeline
    )
)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

val_dataloader = dict(
    batch_size=32,
    num_workers=1,
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt',
        pipeline=val_pipeline
    )
)

test_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt'),
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

val_evaluator = [
    dict(type='Aggregate'),
    dict(type='AveragePrecision'),
    dict(type='AUC')
]
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=25000, max_keep_ckpts=-1, save_last=False),
    logger=dict(interval=10),
)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

optimizer = dict(
    type='AdamW',
    lr=lr,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
)

param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=1e-3,
    #     by_epoch=True,
    #     end=1
    # ),
    # dict(
    #     type='CosineAnnealingLR',
    #     eta_min=1e-5,
    #     by_epoch=True,
    #     begin=1)
]

randomness = dict(seed=0)
train_cfg = dict(by_epoch=True, val_interval=5, max_epochs=50)
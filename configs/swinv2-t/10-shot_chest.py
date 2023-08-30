_base_ = [
    '../datasets/colon.py',
    '../schedules/chest.py',
    'mmpretrain::_base_/models/swin_transformer_v2/tiny_256.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py'
]

lr = 5e-4
train_bs = 8
dataset = 'chest'
model_name = 'swinv2-t'
exp_num = 1
nshot = 10
seed = 2049
randomness = dict(seed=seed)

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}'
work_dir = f'work_dirs/colon/{nshot}-shot/{run_name}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        img_size=256,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-tiny-w16_3rdparty_in1k-256px_20220803-9651cdd7.pth',
            prefix='backbone',
        ),
        window_size=[16, 16, 16, 8],
        drop_path_rate=0.2,
        #pretrained_window_sizes=[12, 12, 12, 6]
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1024,
        loss=dict(type='LabelSmoothLoss', loss_weight=1.0),
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='NumpyToPIL', to_rgb=True),
    dict(type='torchvision/RandomAffine', degrees=(-15, 15), translate=(0.05, 0.05), fill=128),
    dict(type='PILToNumpy', to_bgr=True),
    dict(type='RandomResizedCrop', scale=256, crop_ratio_range=(0.9, 1.0), backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    num_workers=4,
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
        pipeline=train_pipeline
    )
)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=256, backend='pillow', interpolation='bicubic'),
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

val_evaluator = [
    dict(type='Aggregate'),
    dict(type='AveragePrecision'),
    dict(type='AUC')
]
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=250,
        max_keep_ckpts=1,
        save_best="Aggregate", rule="greater"
    ),
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
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=1
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        begin=1)
]

train_cfg = dict(by_epoch=True, val_interval=25, max_epochs=1000)

auto_scale_lr = dict(base_batch_size=1024)

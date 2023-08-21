_base_ = [
    'mmpretrain::_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'mmpretrain::_base_/models/swin_transformer_v2/base_384.py',
    'mmpretrain::_base_/default_runtime.py',
    '../datasets/colon.py',
    '../custom_imports.py'
]

lr = ''
train_bs = 16
dataset = 'colon'
model_name = 'swinv2'
exp_num = 1
nshot = 1

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/colon/{nshot}-shot/{run_name}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        img_size=384,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w24_in21k-pre_3rdparty_in1k-384px_20220803-44eb70f8.pth',
            prefix='backbone',
        ),
        window_size=[24, 24, 24, 12],
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6]),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1024,
        loss=dict(type='LabelSmoothLoss', loss_weight=1.0),
    )
)

bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(
        type='Resize',
        scale=384,
        backend='pillow',
        interpolation='bicubic'
    ),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=8,
        magnitude_std=0.7,
        hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')
    ),
    dict(type='RandomGrayscale', prob=0.5, keep_channels=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
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

val_dataloader = dict(
    batch_size=32,
    num_workers=1,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
)

val_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='AUC')
]
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=250, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=10),
)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(by_epoch=True, val_interval=50, max_epochs=1000)

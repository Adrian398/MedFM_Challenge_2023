_base_ = [
    '../../datasets/chest.py',
    '../../schedules/chest.py',
    'mmpretrain::_base_/default_runtime.py',
    'mmpretrain::_base_/models/efficientnet_v2/efficientnetv2_s.py',
    '../../custom_imports.py',
]


# Pre-trained Checkpoint Path
checkpoints = {
    "s": "https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_in21k-pre-3rdparty_in1k_20221220-7a7c8475.pth",
    "m": "https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_in21k-pre-3rdparty_in1k_20221220-a1013a04.pth",
    "l": "https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-l_in21k-pre-3rdparty_in1k_20221220-63df0efd.pth",
    "xl": "https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-xl_in21k-pre-3rdparty_in1k_20221220-583ac18b.pth"
}

train_scales = {
    "s": 300,
    "m": 384,
    "l": 384,
    "xl": 384
}

test_scales = {
    "s": 384,
    "m": 480,
    "l": 480,
    "xl": 512
}

lr = 1e-4
train_bs = 8
val_bs = 64
dataset = 'chest'
model_name = 'efficientnetv2'
arch = 'm'
exp_num = 1
nshot = 1

run_name = f'{model_name}_{arch}_bs{train_bs}_lr{lr}_exp{exp_num}'
work_dir = f'work_dirs/{dataset}/{nshot}-shot/{run_name}'

model = dict(
    backbone=dict(
        arch=arch,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoints[arch], prefix='backbone')
    ),
    neck=None,
    head=dict(
        type='CSRAClsHead',
        num_classes=19,
        in_channels=1280,
        num_heads=1,
        lam=0.1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        topk=None))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='NumpyToPIL', to_rgb=True),
    dict(type='torchvision/RandomAffine', degrees=(-15, 15), translate=(0.05, 0.05), fill=128),
    dict(type='PILToNumpy', to_bgr=True),
    dict(type='EfficientNetRandomCrop', scale=train_scales[arch], crop_ratio_range=(0.9, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=test_scales[arch], crop_padding=0),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
                 pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=val_bs,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt',
                 pipeline=test_pipeline)
)

test_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt',
                 pipeline=test_pipeline)
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=250, max_keep_ckpts=1, save_best="Aggregate", rule="greater"),
    logger=dict(interval=10),
)

optim_wrapper = dict(optimizer=dict(lr=lr))

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(by_epoch=True, val_interval=25, max_epochs=500)

randomness = dict(seed=0)

auto_scale_lr = dict(base_batch_size=1024)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth'
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'medfmc.datasets.medical_datasets',
        'medfmc.evaluation.metrics.auc',
        'medfmc.models',
    ])
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=2,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset = 'colon'
dataset_type = 'Colon'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmpretrain',
        interval=250,
        max_keep_ckpts=1,
        rule='greater',
        save_best='Aggregate',
        type='CheckpointHook'),
    logger=dict(_scope_='mmpretrain', interval=10, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmpretrain', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmpretrain', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmpretrain', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmpretrain', enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
exp_num = 4
launcher = 'none'
load_from = None
log_level = 'INFO'
lr = 1e-06
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth',
            prefix='backbone',
            type='Pretrained'),
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        lam=0.1,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        num_classes=2,
        num_heads=1,
        type='CSRAClsHead'),
    neck=None,
    type='ImageClassifier')
model_name = 'resnet101'
nshot = 1
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=1e-06,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
        flat_decay_mult=0.0,
        norm_decay_mult=0.0))
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), eps=1e-08, lr=1e-06, type='AdamW', weight_decay=0.05)
param_scheduler = [
    dict(by_epoch=True, end=1, start_factor=1, type='LinearLR'),
    dict(begin=1, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(seed=1)
resume = False
run_name = 'resnet101_bs32_lr1e-06e-06_exp4__seed_test_'
test_cfg = dict()
test_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/test_WithLabel.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=448, type='Resize'),
            dict(
                meta_keys=(
                    'sample_idx',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'gt_label_difficult',
                ),
                type='PackInputs'),
        ],
        type='Colon'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='Aggregate'),
    dict(type='AveragePrecision'),
    dict(type='AUC'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=448, type='Resize'),
    dict(
        meta_keys=(
            'sample_idx',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'gt_label_difficult',
        ),
        type='PackInputs'),
]
train_bs = 32
train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=50)
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/colon_1-shot_train_exp2.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                crop_ratio_range=(
                    0.7,
                    1.0,
                ),
                scale=448,
                type='RandomResizedCrop'),
            dict(
                brightness=0.4,
                contrast=0.4,
                hue=0.3,
                saturation=0.4,
                type='ColorJitter'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(direction='vertical', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='Colon'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_evaluator = [
    dict(type='Aggregate'),
    dict(type='AveragePrecision'),
    dict(type='AUC'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(crop_ratio_range=(
        0.7,
        1.0,
    ), scale=448, type='RandomResizedCrop'),
    dict(
        brightness=0.4,
        contrast=0.4,
        hue=0.3,
        saturation=0.4,
        type='ColorJitter'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_bs = 256
val_cfg = dict()
val_dataloader = dict(
    batch_size=256,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/colon_1-shot_val_exp2.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=448, type='Resize'),
            dict(
                meta_keys=(
                    'sample_idx',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'gt_label_difficult',
                ),
                type='PackInputs'),
        ],
        type='Colon'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='Aggregate'),
    dict(type='AveragePrecision'),
    dict(type='AUC'),
]
vis_backends = [
    dict(_scope_='mmpretrain', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmpretrain',
    type='Visualizer',
    vis_backends=[
        dict(type='TensorboardVisBackend'),
    ])
work_dir = '/scratch/medfm/medfm-challenge/work_dirs/colon/1-shot/resnet101_bs32_lr1e-06e-06_exp2__seed_test_20230831-153001'

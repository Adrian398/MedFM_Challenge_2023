auto_scale_lr = dict(base_batch_size=1024)
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
    num_classes=19,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_onehot=True,
    to_rgb=True)
dataset = 'chest'
dataset_type = 'Chest19'
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
exp_num = 1
launcher = 'none'
load_from = '/scratch/medfm/medfm-challenge/work_dirs/chest/1-shot/swinv2-t_bs2_lr1e-06_exp1_swinv2_tests20230912-144809/best_Aggregate_epoch_100.pth'
log_level = 'INFO'
lr = 1e-06
model = dict(
    _scope_='mmpretrain',
    backbone=dict(
        arch='tiny',
        drop_path_rate=0.2,
        img_size=256,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-tiny-w16_3rdparty_in1k-256px_20220803-9651cdd7.pth',
            prefix='backbone',
            type='Pretrained'),
        pretrained_window_sizes=[
            12,
            12,
            12,
            6,
        ],
        type='SwinTransformerV2',
        window_size=[
            16,
            16,
            16,
            8,
        ]),
    head=dict(
        in_channels=768,
        lam=0.1,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        num_classes=19,
        num_heads=1,
        type='CSRAClsHead'),
    neck=None,
    type='ImageClassifier')
model_name = 'swinv2-t'
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
randomness = dict(seed=3)
resume = False
run_name = 'swinv2-t_bs2_lr1e-06_exp1_swinv2_tests'
seed = 2049
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/chest/test_WithLabel.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/chest/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=256,
                type='Resize'),
            dict(type='PackInputs'),
        ],
        type='Chest19'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AveragePrecision'),
    dict(average='macro', type='MultiLabelMetric'),
    dict(average='micro', type='MultiLabelMetric'),
    dict(multilabel=True, type='AUC'),
    dict(multilabel=True, type='Aggregate'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', interpolation='bicubic', scale=256, type='Resize'),
    dict(type='PackInputs'),
]
train_bs = 2
train_cfg = dict(by_epoch=True, max_epochs=125, val_interval=10)
train_dataloader = dict(
    batch_size=2,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/chest/chest_1-shot_train_exp1.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/chest/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(to_rgb=True, type='NumpyToPIL'),
            dict(
                degrees=(
                    -15,
                    15,
                ),
                fill=128,
                translate=(
                    0.05,
                    0.05,
                ),
                type='torchvision/RandomAffine'),
            dict(to_bgr=True, type='PILToNumpy'),
            dict(
                backend='pillow',
                crop_ratio_range=(
                    0.9,
                    1.0,
                ),
                interpolation='bicubic',
                scale=256,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='Chest19'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_evaluator = [
    dict(type='AveragePrecision'),
    dict(average='macro', type='MultiLabelMetric'),
    dict(average='micro', type='MultiLabelMetric'),
    dict(multilabel=True, type='AUC'),
    dict(multilabel=True, type='Aggregate'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(to_rgb=True, type='NumpyToPIL'),
    dict(
        degrees=(
            -15,
            15,
        ),
        fill=128,
        translate=(
            0.05,
            0.05,
        ),
        type='torchvision/RandomAffine'),
    dict(to_bgr=True, type='PILToNumpy'),
    dict(
        backend='pillow',
        crop_ratio_range=(
            0.9,
            1.0,
        ),
        interpolation='bicubic',
        scale=256,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_bs = 32
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/chest/chest_1-shot_val_exp1.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/chest/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=256,
                type='Resize'),
            dict(type='PackInputs'),
        ],
        type='Chest19'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AveragePrecision'),
    dict(average='macro', type='MultiLabelMetric'),
    dict(average='micro', type='MultiLabelMetric'),
    dict(multilabel=True, type='AUC'),
    dict(multilabel=True, type='Aggregate'),
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
work_dir = '/scratch/medfm/medfm-challenge/work_dirs/chest/1-shot/swinv2-t_bs2_lr1e-06_exp1_swinv2_tests20230912-144809'

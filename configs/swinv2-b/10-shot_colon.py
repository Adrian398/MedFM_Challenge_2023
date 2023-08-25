auto_scale_lr = dict(base_batch_size=1024)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
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
    checkpoint=dict(type='CheckpointHook', interval=250, max_keep_ckpts=2, save_best="Aggregate", rule="greater"),
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
load_from = None
log_level = 'INFO'
lr = 0.05
model = dict(
    _scope_='mmpretrain',
    backbone=dict(
        arch='base',
        drop_path_rate=0.2,
        img_size=384,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w24_in21k-pre_3rdparty_in1k-384px_20220803-44eb70f8.pth',
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
            24,
            24,
            24,
            12,
        ]),
    head=dict(
        cal_acc=False,
        in_channels=1024,
        init_cfg=None,
        loss=dict(
            label_smooth_val=0.1,
            loss_weight=1.0,
            mode='original',
            type='LabelSmoothLoss'),
        num_classes=2,
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
model_name = 'swinv2'
nshot = 10
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.001,
        type='AdamW',
        weight_decay=0.01),
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
    ), eps=1e-08, lr=0.05, type='AdamW', weight_decay=0.01)
param_scheduler = [
    dict(by_epoch=True, end=1, start_factor=1, type='LinearLR'),
    dict(begin=1, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
run_name = 'swinv2_bs8_lr0.05_exp1_'
test_cfg = dict()
test_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/test_WithLabel.txt',
        data_prefix='/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=384,
                type='Resize'),
            dict(type='PackInputs'),
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
    dict(backend='pillow', interpolation='bicubic', scale=384, type='Resize'),
    dict(type='PackInputs'),
]
train_bs = 8
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=25)
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/colon_10-shot_train_exp1.txt',
        data_prefix='/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=384,
                type='Resize'),
            dict(type='PackInputs'),
        ],
        type='Colon'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', interpolation='bicubic', scale=384, type='Resize'),
    dict(
        type='Normalize',
        mean=bgr_mean,
        std=bgr_std,
        to_rgb=False
    ),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/colon_10-shot_val_exp1.txt',
        data_prefix='/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=384,
                type='Resize'),
            dict(type='PackInputs'),
        ],
        type='Colon'),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='AUC'),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', interpolation='bicubic', scale=384, type='Resize'),
    dict(type='PackInputs'),
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
work_dir = '/scratch/medfm/medfm-challenge/work_dirs/colon/10-shot/swinv2_bs8_lr0.05_exp1_'

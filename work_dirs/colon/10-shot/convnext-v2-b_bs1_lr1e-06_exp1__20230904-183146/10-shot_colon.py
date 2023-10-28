auto_scale_lr = dict(base_batch_size=1024)
custom_hooks = [
    dict(momentum=0.0001, priority='ABOVE_NORMAL', type='EMAHook'),
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
    checkpoint=dict(
        _scope_='mmpretrain',
        interval=250,
        max_keep_ckpts=1,
        rule='greater',
        save_best='accuracy/top1',
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
load_from = '/scratch/medfm/medfm-challenge/work_dirs/colon/10-shot/convnext-v2-b_bs1_lr1e-06_exp1__20230904-183146/best_accuracy_top1_epoch_60.pth'
log_level = 'INFO'
lr = 0.0025
model = dict(
    _scope_='mmpretrain',
    backbone=dict(
        arch='base',
        drop_path_rate=0.1,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth',
            prefix='backbone',
            type='Pretrained'),
        layer_scale_init_value=0.0,
        type='ConvNeXt',
        use_grn=True),
    head=dict(
        in_channels=1024,
        init_cfg=None,
        loss=dict(label_smooth_val=0.1, type='LabelSmoothLoss'),
        num_classes=2,
        type='LinearClsHead'),
    init_cfg=dict(
        bias=0.0, layer=[
            'Conv2d',
            'Linear',
        ], std=0.02, type='TruncNormal'),
    train_cfg=dict(augments=[
        dict(alpha=0.8, type='Mixup'),
        dict(alpha=1.0, type='CutMix'),
    ]),
    type='ImageClassifier')
model_name = 'convnext-v2-b'
nshot = 10
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=1e-06,
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
    ), eps=1e-08, lr=1e-06, type='AdamW', weight_decay=0.01)
param_scheduler = [
    dict(by_epoch=True, end=1, start_factor=1, type='LinearLR'),
    dict(begin=1, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(seed=0)
resume = False
run_name = 'convnext-v2-b_bs1_lr1e-06_exp1__'
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/test_WithLabel.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
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
    dict(type='AveragePrecision'),
    dict(topk=(1, ), type='Accuracy'),
    dict(items=[
        'precision',
        'recall',
    ], type='SingleLabelMetric'),
    dict(type='Aggregate'),
    dict(type='AUC'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', interpolation='bicubic', scale=384, type='Resize'),
    dict(type='PackInputs'),
]
train_bs = 1
train_cfg = dict(by_epoch=True, max_epochs=125, val_interval=10)
train_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/colon_10-shot_train_exp1.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=384,
                type='RandomResizedCrop'),
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
    dict(type='AveragePrecision'),
    dict(topk=(1, ), type='Accuracy'),
    dict(items=[
        'precision',
        'recall',
    ], type='SingleLabelMetric'),
    dict(type='Aggregate'),
    dict(type='AUC'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=384,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_bs = 96
val_cfg = dict()
val_dataloader = dict(
    batch_size=96,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/colon_10-shot_val_exp1.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
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
val_evaluator = [
    dict(type='AveragePrecision'),
    dict(topk=(1, ), type='Accuracy'),
    dict(items=[
        'precision',
        'recall',
    ], type='SingleLabelMetric'),
    dict(type='Aggregate'),
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
work_dir = '/scratch/medfm/medfm-challenge/work_dirs/colon/10-shot/convnext-v2-b_bs1_lr1e-06_exp1__20230904-183146'

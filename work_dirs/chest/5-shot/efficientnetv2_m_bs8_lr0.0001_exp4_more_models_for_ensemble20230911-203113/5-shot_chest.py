arch = 'm'
auto_scale_lr = dict(base_batch_size=1024)
checkpoints = dict(
    l=
    'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-l_in21k-pre-3rdparty_in1k_20221220-63df0efd.pth',
    m=
    'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_in21k-pre-3rdparty_in1k_20221220-a1013a04.pth',
    s=
    'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_in21k-pre-3rdparty_in1k_20221220-7a7c8475.pth',
    xl=
    'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-xl_in21k-pre-3rdparty_in1k_20221220-583ac18b.pth'
)
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
load_from = '/scratch/medfm/medfm-challenge/work_dirs/chest/5-shot/efficientnetv2_m_bs8_lr0.0001_exp4_more_models_for_ensemble20230911-203113/best_Aggregate_epoch_75.pth'
log_level = 'INFO'
lr = 0.0001
model = dict(
    _scope_='mmpretrain',
    backbone=dict(
        arch='m',
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_in21k-pre-3rdparty_in1k_20221220-a1013a04.pth',
            prefix='backbone',
            type='Pretrained'),
        type='EfficientNetV2'),
    head=dict(
        in_channels=1280,
        lam=0.1,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        num_classes=19,
        num_heads=1,
        topk=None,
        type='CSRAClsHead'),
    neck=None,
    type='ImageClassifier')
model_name = 'efficientnetv2'
nshot = 5
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
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
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        start_factor=0.001,
        type='LinearLR'),
    dict(begin=1, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(seed=0)
resume = False
run_name = 'efficientnetv2_m_bs8_lr0.0001_exp1_more_models_for_ensemble'
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
            dict(crop_padding=0, crop_size=480, type='EfficientNetCenterCrop'),
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
    dict(crop_padding=0, crop_size=480, type='EfficientNetCenterCrop'),
    dict(type='PackInputs'),
]
test_scales = dict(l=480, m=480, s=384, xl=512)
train_bs = 8
train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=25)
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/chest/chest_5-shot_train_exp4.txt',
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
                crop_ratio_range=(
                    0.9,
                    1.0,
                ),
                scale=384,
                type='EfficientNetRandomCrop'),
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
        crop_ratio_range=(
            0.9,
            1.0,
        ),
        scale=384,
        type='EfficientNetRandomCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
train_scales = dict(l=384, m=384, s=300, xl=384)
val_bs = 64
val_cfg = dict()
val_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/chest/chest_5-shot_val_exp4.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/chest/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(crop_padding=0, crop_size=480, type='EfficientNetCenterCrop'),
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
work_dir = '/scratch/medfm/medfm-challenge/work_dirs/chest/5-shot/efficientnetv2_m_bs8_lr0.0001_exp4_more_models_for_ensemble20230911-203113'

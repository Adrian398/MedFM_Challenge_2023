auto_scale_lr = dict(base_batch_size=1024)
cos_end_lr = 1e-06
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
    num_classes=4,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_onehot=True,
    to_rgb=True)
dataset = 'endo'
dataset_type = 'Endoscopy'
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
exp_num = 3
launcher = 'none'
load_from = '/scratch/medfm/medfm-challenge/work_dirs/endo/1-shot/swin-b_vpt_bs8_lr0.0005_exp5__pre_processed20230912-103005/best_Aggregate_epoch_15.pth'
log_level = 'INFO'
lr = 0.0005
model = dict(
    backbone=dict(
        arch='base',
        img_size=384,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth',
            prefix='backbone',
            type='Pretrained'),
        prompt_length=5,
        stage_cfgs=dict(block_cfgs=dict(window_size=12)),
        type='PromptedSwinTransformer'),
    head=dict(in_channels=1024, num_classes=4, type='MultiLabelLinearClsHead'),
    neck=None,
    type='ImageClassifier')
model_name = 'swin-b_vpt'
nshot = 1
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0005,
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
        gamma=0.5,
        milestones=[
            100,
            200,
            300,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=5)
resume = False
run_name = 'swin-b_vpt_bs8_lr0.0005_exp3__pre_processed'
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/endo/test_WithLabel.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/endo/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=384,
                type='Resize'),
            dict(type='PackInputs'),
        ],
        type='Endoscopy'),
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
    dict(backend='pillow', interpolation='bicubic', scale=384, type='Resize'),
    dict(type='PackInputs'),
]
train_bs = 8
train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=15)
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/endo/endo_1-shot_train_exp5.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/endo/pre_processed_images',
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
        type='Endoscopy'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_evaluator = [
    dict(type='AveragePrecision'),
    dict(average='macro', type='MultiLabelMetric'),
    dict(average='micro', type='MultiLabelMetric'),
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
val_cfg = dict()
val_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/endo/endo_1-shot_val_exp5.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/endo/pre_processed_images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=384,
                type='Resize'),
            dict(type='PackInputs'),
        ],
        type='Endoscopy'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AveragePrecision'),
    dict(average='macro', type='MultiLabelMetric'),
    dict(average='micro', type='MultiLabelMetric'),
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
vpl = 5
warmup_lr = 0.001
work_dir = '/scratch/medfm/medfm-challenge/work_dirs/endo/1-shot/swin-b_vpt_bs8_lr0.0005_exp5__pre_processed20230912-103005'

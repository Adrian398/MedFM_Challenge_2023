auto_scale_lr = dict(base_batch_size=1024, enable=False)
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
cos_end_lr = 1e-04
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
        save_best='auto',
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
load_from = None
log_level = 'INFO'
lr = 0.005
model = dict(
    backbone=dict(
        arch='base',
        drop_rate=0.1,
        img_size=384,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth',
            prefix='backbone',
            type='Pretrained'),
        prompt_length=5,
        stage_cfgs=dict(block_cfgs=dict(window_size=12)),
        type='PromptedSwinTransformer'),
    head=dict(
        in_channels=1024,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=2,
        type='LinearClsHead'),
    neck=None,
    type='ImageClassifier')
model_name = 'swin'
nshot = 1
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.005,
        type='AdamW',
        weight_decay=0.005),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
        flat_decay_mult=0.0,
        norm_decay_mult=0.0))
param_scheduler = [
    dict(by_epoch=True, milestones=[
        1,
        2,
    ], type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
run_name = 'swin_bs16_lr0.005_exp1_'
test_cfg = dict()
test_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/test_WithLabel.txt',
        data_prefix='data/MedFMC_train/colon/images',
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
    dict(type='AUC'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', interpolation='bicubic', scale=384, type='Resize'),
    dict(type='PackInputs'),
]
train_bs = 16
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=100)
train_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/colon_1-shot_train_exp1.txt',
        data_prefix='data/MedFMC_train/colon/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=384,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='Colon'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        erase_prob=0.25,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(
        hparams=dict(interpolation='bicubic', pad_val=[
            104,
            116,
            124,
        ]),
        magnitude_level=9,
        magnitude_std=0.5,
        num_policies=2,
        policies='timm_increasing',
        total_level=10,
        type='RandAugment'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=255,
        type='RandomResizedCrop'),
    dict(type='RandomPatchWithLabels'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(keep_channels=True, prob=0.5, type='RandomGrayscale'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/colon/colon_1-shot_val_exp1.txt',
        data_prefix='data/MedFMC_train/colon/images',
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
work_dir = 'work_dirs/colon/1-shot/swin_bs16_lr0.005_exp1_20230820-221921'

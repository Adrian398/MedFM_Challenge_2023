auto_scale_lr = dict(base_batch_size=1024)
bgr_mean = [
    123.675,
    116.28,
    103.53,
]
bgr_std = [
    58.395,
    57.12,
    57.375,
]
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
        interval=100,
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
model_name = 'swin'
nshot = 10
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
            50,
            70,
            85,
            100,
            110,
            120,
            125,
            130,
            135,
            140,
            145,
            150,
            155,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
run_name = 'swin_bs8_lr0.0005_exp1_'
test_cfg = dict()
test_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/endo/test_WithLabel.txt',
        data_prefix='/scratch/medfm/data/MedFMC_train/endo/images',
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
    dict(type='AUC'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', interpolation='bicubic', scale=1028, type='Resize'),
    dict(type='PackInputs'),
]
train_bs = 8
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=20)
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/endo/endo_10-shot_train_exp1.txt',
        data_prefix='/scratch/medfm/data/MedFMC_train/endo/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                erase_prob=0.25,
                fill_color=[
                    123.675,
                    116.28,
                    103.53,
                ],
                fill_std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='RandomErasing'),
            dict(
                hparams=dict(
                    interpolation='bicubic', pad_val=[
                        124,
                        116,
                        104,
                    ]),
                magnitude_level=8,
                magnitude_std=0.7,
                num_policies=2,
                policies='timm_increasing',
                total_level=10,
                type='RandAugment'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=384,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(keep_channels=True, prob=0.5, type='RandomGrayscale'),
            dict(direction='vertical', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='Endoscopy'),
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        erase_prob=0.25,
        fill_color=[
            123.675,
            116.28,
            103.53,
        ],
        fill_std=[
            58.395,
            57.12,
            57.375,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(
        hparams=dict(interpolation='bicubic', pad_val=[
            124,
            116,
            104,
        ]),
        magnitude_level=8,
        magnitude_std=0.7,
        num_policies=2,
        policies='timm_increasing',
        total_level=10,
        type='RandAugment'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=384,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(keep_channels=True, prob=0.5, type='RandomGrayscale'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/endo/endo_10-shot_val_exp1.txt',
        data_prefix='/scratch/medfm/data/MedFMC_train/endo/images',
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
work_dir = '/scratch/medfm/medfm-challenge/work_dirs/endo/10-shot/swin_bs8_lr0.0005_exp1_'

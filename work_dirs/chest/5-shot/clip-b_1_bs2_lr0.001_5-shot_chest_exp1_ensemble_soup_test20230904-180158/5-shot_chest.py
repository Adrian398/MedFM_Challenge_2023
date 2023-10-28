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
        interval=1,
        max_keep_ckpts=1,
        save_best='auto',
        type='CheckpointHook'),
    logger=dict(_scope_='mmpretrain', interval=50, type='LoggerHook'),
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
load_from = '/scratch/medfm/medfm-challenge/work_dirs/chest/5-shot/clip-b_1_bs2_lr0.001_5-shot_chest_exp1_ensemble_soup_test20230904-180158/best_multi-label_mAP_epoch_156.pth'
log_level = 'INFO'
lr = 0.001
model = dict(
    backbone=dict(
        arch='b',
        img_size=384,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_laion2b-in12k-pre_3rdparty_in1k-384px_20221220-84ed0cc0.pth',
            prefix='backbone',
            type='Pretrained'),
        out_type='cls_token',
        patch_size=16,
        pre_norm=True,
        prompt_length=1,
        type='PromptedViT'),
    head=dict(in_channels=768, num_classes=19, type='MultiLabelLinearClsHead'),
    neck=None,
    type='ImageClassifier')
nshot = 5
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.001,
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
randomness = dict(seed=1337)
resume = False
run_name = 'clip-b_1_bs2_lr0.001_5-shot_chest_exp1_ensemble_soup_test'
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
                edge='short',
                interpolation='bicubic',
                scale=384,
                type='ResizeEdge'),
            dict(crop_size=384, type='CenterCrop'),
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
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=384,
        type='ResizeEdge'),
    dict(crop_size=384, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_bs = 2
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
train_dataloader = dict(
    batch_size=2,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/chest_new/chest_5-shot_train_exp1.txt',
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
                scale=384,
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
        scale=384,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='data_anns/MedFMC/chest_new/chest_5-shot_val_exp1.txt',
        data_prefix=
        '/scratch/medfm/medfm-challenge/data/MedFMC_train/chest/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=384,
                type='ResizeEdge'),
            dict(crop_size=384, type='CenterCrop'),
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
vpl = 1
work_dir = '/scratch/medfm/medfm-challenge/work_dirs/chest/5-shot/clip-b_1_bs2_lr0.001_5-shot_chest_exp1_ensemble_soup_test20230904-180158'

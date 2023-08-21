_base_ = [
    'mmpretrain::_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'medfmc.datasets.medical_datasets',
        'medfmc.evaluation.metrics.auc',
        'medfmc.models'
    ],
    allow_failed_imports=False)

# general
task = 'colon'
dataset_type = 'Colon'
model_name = 'swin'
exp_num = 1
n_shot = 1
train_bs = 4
data_prefix = '/scratch/medfm/data/MedFMC_train/colon/images'

# data loading and preprocessing
data_preprocessor = dict(
    num_classes=2,
    mean=[123.675, 116.28, 103.53],  # RGB format normalization parameters
    std=[58.395, 57.12, 57.375],
    to_rgb=True,  # convert image from BGR to RGB
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    dataset=dict(ann_file=f'data_anns/MedFMC/{task}/{task}_{n_shot}-shot_train_exp{exp_num}.txt',
                 data_prefix=data_prefix,
                 dataset_type=dataset_type,
                 pipeline=train_pipeline),
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

val_dataloader = dict(
    batch_size=64,
    dataset=dict(pipeline=test_pipeline,
                 data_prefix=data_prefix,
                 dataset_type=dataset_type,
                 ann_file=f'data_anns/MedFMC/{task}/{task}_{n_shot}-shot_val_exp{exp_num}.txt'),
)

test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=data_prefix,
        dataset_type=dataset_type,
        ann_file=f'data_anns/MedFMC/{task}/test_WithLabel.txt'),
)

# metrics
val_evaluator = [dict(type='Accuracy', topk=(1,)), dict(type='AUC')]
test_evaluator = val_evaluator

# model
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedSwinTransformer',
        prompt_length=5,
        arch='base',
        img_size=384,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth',
            prefix='backbone',
        ),
        stage_cfgs=dict(block_cfgs=dict(window_size=12))),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=50),
)

# optimizer
lr = 5e-2
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=1,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=1)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=1500, val_interval=250)
val_cfg = dict()
test_cfg = dict()

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/colon/{n_shot}-shot/{run_name}'

_base_ = [
    '../datasets/chest.py',
    '../schedules/chest.py',
    'mmpretrain::_base_/default_runtime.py',
    'mmpretrain::_base_/models/efficientformer-l1.py',
    '../custom_imports.py',
]

lr = 1e-5
train_bs = 4
val_bs = 128
dataset = 'chest'
model_name = 'efficientformer'
arch = "l1"
exp_num = 1
nshot = 1

run_name = f'{model_name}_{arch}_bs{train_bs}_lr{lr}_exp{exp_num}'
work_dir = f'work_dirs/{dataset}/{nshot}-shot/{run_name}'

model = dict(
    backbone=dict(
        arch=arch
    ),
    head=dict(
        type='EfficientFormerClsHead',
        num_classes=19))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=224, crop_ratio_range=(0.7, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=train_bs,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
                 pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=val_bs,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt',
                 pipeline=test_pipeline)
)

test_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt',
                 pipeline=test_pipeline)
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=250, max_keep_ckpts=1, save_best="auto", rule="greater"),
    logger=dict(interval=10),
)

optim_wrapper = dict(optimizer=dict(lr=lr))

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(by_epoch=True, val_interval=25, max_epochs=1000)

randomness = dict(seed=0)

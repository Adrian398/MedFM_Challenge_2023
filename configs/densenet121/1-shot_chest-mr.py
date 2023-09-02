_base_ = [
    '../datasets/chest.py',
    '../schedules/adamw_inverted_cosine_lr.py',
    'mmpretrain::_base_/models/densenet/densenet121.py',
    #'mmpretrain::_base_/schedules/imagenet_bs256.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

# Pre-trained Checkpoint Path
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/densenet/densenet121_4xb256_in1k_20220426-07450f99.pth'  # noqa

lr = 1e-6
train_bs = 16
val_bs = 64
dataset = 'chest'
model_name = 'densenet121'
exp_num = 1
nshot = 1

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/{dataset}/{nshot}-shot/{run_name}'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone')
    ),
    head=dict(
        in_channels=1024,
        num_classes=19,
        type='MultiLabelLinearClsHead'),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=256, crop_ratio_range=(0.7, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=256),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt')
)

val_dataloader = dict(
    batch_size=val_bs,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt')
)

test_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt')
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=250, max_keep_ckpts=1, save_best="auto", rule="greater"),
    logger=dict(interval=10),
)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(by_epoch=True, val_interval=25, max_epochs=500)

randomness = dict(seed=0)

auto_scale_lr = dict(base_batch_size=train_bs)


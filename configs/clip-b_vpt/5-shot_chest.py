_base_ = [
    '../datasets/chest.py',
    '../schedules/chest.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 1e-3
vpl = 1
dataset = 'chest'
exp_num = 1
nshot = 5
seed = 2049
randomness = dict(seed=seed)

run_name = f'clip-b_{vpl}_bs4_lr{lr}_{nshot}-shot_{dataset}_exp{exp_num}'
work_dir = f'work_dirs/chest/{nshot}-shot/{run_name}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedViT',
        prompt_length=vpl,
        patch_size=16,
        out_type='cls_token',
        arch='b',
        pre_norm=True,
        img_size=384,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_laion2b-in12k-pre_3rdparty_in1k-384px_20221220-84ed0cc0.pth',
            prefix='backbone',
        ),
        ),
    neck=None,
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=768,
    ))

# data settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='NumpyToPIL', to_rgb=True),
    dict(type='torchvision/RandomAffine', degrees=(-15, 15), translate=(0.05, 0.05), fill=128),
    dict(type='PILToNumpy', to_bgr=True),
    dict(type='RandomResizedCrop', scale=384, crop_ratio_range=(0.9, 1.0), backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=384,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}_new/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
        pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}_new/{dataset}_{nshot}-shot_val_exp{exp_num}.txt',
        pipeline=test_pipeline),
)

test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt',
        pipeline=test_pipeline),
)

optim_wrapper = dict(optimizer=dict(lr=lr), clip_grad=dict(max_norm=1.0))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=50),
)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

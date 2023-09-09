_base_ = [
    '../datasets/colon.py',
    '../swin_schedule.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 5e-3
vpl = 1
dataset = 'colon'
exp_num = 1
nshot = 1
run_name = f'dinov2-b_{nshot}-shot_ptokens-{vpl}_{dataset}'

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedViT',
        prompt_length=vpl,
        layer_scale_init_value=1e-5,
        out_type='avg_all',
        img_size=518,
        patch_size=14,
        arch='base',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth',
            prefix='backbone'),
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

val_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='Accuracy', topk=(1,)),
    dict(type='AUC')
]
test_evaluator = val_evaluator

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=518,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=518, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
        pipeline=train_pipeline),
)
visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(by_epoch=True, val_interval=10, max_epochs=100)
val_dataloader = dict(
    batch_size=24,
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt',
        pipeline=test_pipeline),
)

test_dataloader = dict(
    batch_size=24,
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt',
        pipeline=test_pipeline),
)
optim_wrapper = dict(optimizer=dict(lr=lr))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=50),
)

work_dir = f'work_dirs/dinov2-b/exp{exp_num}/{run_name}'

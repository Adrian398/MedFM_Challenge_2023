


_base_ = [
    '../datasets/endoscopy.py',
    '../swin_schedule.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

warmup_lr = 1e-3
lr = 5e-4
cos_end_lr = 1e-6
train_bs = 8
vpl = 5
dataset = 'endo'
model_name = 'swin'
exp_num = 1
nshot = 10
run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/endo/{nshot}-shot/{run_name}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedSwinTransformer',
        prompt_length=vpl,
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
        type='MultiLabelLinearClsHead',
        num_classes=4,
        in_channels=1024,
    ))

bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=8,
        magnitude_std=0.7,
        hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')
    ),
    dict(
        type='RandomResizedCrop',
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomGrayscale', prob=0.5, keep_channels=True),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    num_workers=16, 
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',  pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=8,  
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
)

test_dataloader = dict(
    batch_size=4,  
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt'),
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=1028, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=50),
)

# optim_wrapper = dict(optimizer=dict(lr=lr))

val_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='AUC')
]
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=100, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=10),
)


visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

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

param_scheduler = [
    dict(type='MultiStepLR',
         milestones=[50, 70, 85, 100, 110, 120, 125, 130, 135, 140, 145, 150, 155],
         by_epoch=True,
         gamma=0.5)
]

train_cfg = dict(by_epoch=True, val_interval=20, max_epochs=200)





#auto_scale_lr = dict(base_batch_size=1024, enable=False)
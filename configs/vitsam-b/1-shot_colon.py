_base_ = [
    '../datasets/colon.py',
    '../swin_schedule.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 5e-4
train_bs = 2
dataset = 'colon'
exp_num = 1
nshot = 1

model_name = 'vitsam'
run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/colon/{nshot}-shot/{run_name}'

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ViTSAM',
        arch='base',
        img_size=1024,
        patch_size=16,
        out_channels=256,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=2,  # For binary classification: tumor or no tumor.
        in_channels=256,  # Matches the out_channels of the backbone.
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

rgb_mean = [123.675, 116.28, 103.53]
rgb_std = [58.395, 57.12, 57.375]

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=rgb_mean,
    std=rgb_std,
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=1024, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomGrayscale', prob=0.5, keep_channels=True),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Normalize', mean=rgb_mean, std=rgb_std, to_rgb=True),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    num_workers=1,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
                 pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=1024, backend='pillow', interpolation='bicubic'),
    dict(type='Normalize', mean=rgb_mean, std=rgb_std, to_rgb=True),  # Add this line
    dict(type='PackInputs'),
]

test_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt', pipeline=test_pipeline),
)

val_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='AUC')
]
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=250, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=10),
)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

optimizer = dict(
    type='AdamW',
    lr=1e-4,  # Initial learning rate. You might need to adjust this based on your dataset and task.
    weight_decay=0.01,  # Weight decay (L2 penalty)
    betas=(0.9, 0.999),  # Default betas values for AdamW
    eps=1e-8  # A small epsilon for numerical stability
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)
# param_scheduler = [
#     dict(
#         policy='CosineAnnealing',  # Cosine Annealing scheduler
#         warmup='linear',  # Linear warmup helps stabilize training in the initial epochs
#         warmup_iters=1000,  # Number of iterations for warmup. You can adjust this value.
#         warmup_ratio=0.001,  # Ratio of the initial learning rate used for warmup
#         min_lr_ratio=1e-5  # Minimum learning rate value, as a ratio of the initial learning rate.
#     )
#     dict(
#        type='CosineAnnealingLR',
#        eta_min=1e-5,
#        by_epoch=True,
#        begin=1)
# ]

lr_config = dict(
    policy='CosineAnnealing',  # Cosine Annealing scheduler
    warmup='linear',  # Linear warmup helps stabilize training in the initial epochs
    warmup_iters=1000,  # Number of iterations for warmup. You can adjust this value.
    warmup_ratio=0.001,  # Ratio of the initial learning rate used for warmup
    min_lr_ratio=1e-5  # Minimum learning rate value, as a ratio of the initial learning rate.
)

train_cfg = dict(by_epoch=True, val_interval=50, max_epochs=1000)
auto_scale_lr = dict(base_batch_size=1024, enable=False)

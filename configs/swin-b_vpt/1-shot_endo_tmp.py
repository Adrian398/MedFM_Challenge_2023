_base_ = [
    '../datasets/endoscopy.py',
    '../swin_schedule.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

warmup_lr = 1e-3
lr = 1e-4
cos_end_lr = 1e-6
train_bs = 8
vpl = 5
dataset = 'endo'
model_name = 'swin'
exp_num = 3
nshot = 1
seed = 123
randomness = dict(seed=seed)
mt = "PromptedSwinTransformer"
run_name = f'endo_seed{seed}_{mt}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/temp/{nshot}-shot/{run_name}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type=mt,
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

train_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt'),
)

val_dataloader = dict(
    batch_size=16,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
)

test_dataloader = dict(
    batch_size=4,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt'),
)

# optim_wrapper = dict(optimizer=dict(lr=lr))

val_evaluator = [
    dict(type='Aggregate'),
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='AUC')
]
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=500, max_keep_ckpts=-1, save_last=False),
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
    # dict(type='MultiStepLR',
    #      milestones=[100, 200, 300],
    #      by_epoch=True,
    #      gamma=0.5)
]
randomness = dict(seed=seed)
train_cfg = dict(by_epoch=True, val_interval=10, max_epochs=500)

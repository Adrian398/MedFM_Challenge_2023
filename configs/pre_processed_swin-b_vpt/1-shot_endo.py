_base_ = [
    '../datasets/endoscopy_pre_processed.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

train_bs = 8
auto_scale_lr = dict(base_batch_size=1024)

vpl = 5
cos_end_lr = 1e-06
dataset = 'endo'
warmup_lr = 0.001
lr = 5e-4
model_name = 'swin-b_vpt'
exp_num = 3
nshot = 1

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/{dataset}/{nshot}-shot/{run_name}'

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

optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=lr,
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
            100,
            200,
            300,
        ],
        type='MultiStepLR'),
]

# Train Dataset
train_dataloader = dict(
    batch_size=train_bs,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt'),
)

# Validation Dataset
val_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
)

# Test Dataset
test_dataloader = dict(
    batch_size=64,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt')
)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=15)
val_cfg = dict()
test_cfg = dict()

randomness = dict(seed=0)

default_hooks = dict(
    checkpoint=dict(interval=250, max_keep_ckpts=1, save_best='Aggregate', rule="greater"),
    logger=dict(interval=10),
)

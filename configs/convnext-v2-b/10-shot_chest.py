_base_ = [
    'mmpretrain::_base_/models/convnext_v2/base.py',
    '../datasets/chest.py',
    '../schedules/adamw_inverted_cosine_lr.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 1e-6
train_bs = 8
val_bs = 96
dataset = 'chest'
model_name = 'convnext-v2-b'
exp_num = 1
nshot = 5

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/{dataset}/{nshot}-shot/{run_name}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='base',
        init_cfg=dict(
            prefix='backbone',
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth'
        )
    ),
    head=dict(
        in_channels=1024,
        num_classes=19,
        type='MultiLabelLinearClsHead'
    )
)

train_dataloader = dict(
    batch_size=train_bs,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt'),
)

val_dataloader = dict(
    batch_size=val_bs,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
)

test_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt'),
)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(by_epoch=True, val_interval=10, max_epochs=150)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

randomness = dict(seed=0)

default_hooks = dict(
    checkpoint=dict(interval=250, max_keep_ckpts=1, save_best="multi-label/mAP", rule="greater"),
    logger=dict(interval=10),
)


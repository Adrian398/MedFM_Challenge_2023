_base_ = [
    'mmpretrain::_base_/models/convnext_v2/base.py',
    '../datasets/colon.py',
    'mmpretrain::_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 2.5e-3
train_bs = 8
val_bs = 32
dataset = 'colon'
model_name = 'convnext-v2-b'
exp_num = 1
nshot = 1

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
        num_classes=2,
        type='LinearClsHead'
    )
)

# dataset setting
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

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=lr),
    clip_grad=None,
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20)
]

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

# train, val, test setting
train_cfg = dict(by_epoch=True, val_interval=25, max_epochs=500)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

randomness = dict(seed=0)

default_hooks = dict(
    checkpoint=dict(interval=250, max_keep_ckpts=1, save_best="Accuracy", rule="greater"),
    logger=dict(interval=10),
)


_base_ = [
    '../datasets/endoscopy_pre_processed.py',
    '../schedules/adamw_inverted_cosine_lr.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 1e-6
train_bs = 8
val_bs = 32
dataset = 'endo'
model_name = 'swinv2'
exp_num = 1
nshot = 1

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/{dataset}/{nshot}-shot/{run_name}'

model = dict(
    _scope_='mmpretrain',
    backbone=dict(
        arch='base',
        drop_path_rate=0.2,
        img_size=384,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w24_in21k-pre_3rdparty_in1k-384px_20220803-44eb70f8.pth',
            prefix='backbone',
            type='Pretrained'),
        pretrained_window_sizes=[12, 12, 12, 6],
        type='SwinTransformerV2',
        window_size=[24, 24, 24, 12]),
    head=dict(
        in_channels=1024,
        num_classes=4,
        type='MultiLabelLinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')

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

train_cfg = dict(by_epoch=True, val_interval=15, max_epochs=500)

randomness = dict(seed=0)

default_hooks = dict(
    checkpoint=dict(interval=250, max_keep_ckpts=1, save_best="Aggregate", rule="greater"),
    logger=dict(interval=10),
)

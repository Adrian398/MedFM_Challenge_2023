_base_ = [
    '../datasets/colon.py',
    '../schedules/adamw_inverted_cosine_lr.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 1e-6
train_bs = 2
val_bs = 4
dataset = 'colon'
model_name = 'swinv2-l'
exp_num = 1
nshot = 1

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/{dataset}/{nshot}-shot/{run_name}'

model = dict(
    _scope_='mmpretrain',
    backbone=dict(
        arch='large',
        drop_path_rate=0.2,
        img_size=384,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-large-w24_in21k-pre_3rdparty_in1k-384px_20220803-3b36c165.pth',
            prefix='backbone',
            type='Pretrained'),
        pretrained_window_sizes=[12, 12, 12, 6],
        type='SwinTransformerV2',
        window_size=[24, 24, 24, 12]),
    head=dict(
        cal_acc=False,
        in_channels=1024,
        init_cfg=None,
        loss=dict(
            label_smooth_val=0.1,
            loss_weight=1.0,
            mode='original',
            type='LabelSmoothLoss'),
        num_classes=2,
        type='LinearClsHead'),
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

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=250, max_keep_ckpts=1, save_best="Aggregate", rule="greater"),
    logger=dict(interval=10),
)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(by_epoch=True, val_interval=25, max_epochs=500)
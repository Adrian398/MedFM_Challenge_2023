_base_ = [
    '../datasets/chest.py',
    '../schedules/adamw_inverted_cosine_lr.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]

lr = 5e-4
n = 1
vpl = 5
dataset = 'chest'
exp_num = 1
nshot = 1

run_name = f'in21k-swin-b_vpt-{vpl}_bs4_lr{lr}_{nshot}-shot_{dataset}_exp{exp_num}'
work_dir = f'work_dirs/chest/{nshot}-shot/{run_name}'

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
        num_classes=19,
        in_channels=1024,
    ))

train_dataloader = dict(
    batch_size=10,
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt'),
)

val_dataloader = dict(
    batch_size=16,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
)

test_dataloader = dict(
    batch_size=16,
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt'),
)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(by_epoch=True, val_interval=25, max_epochs=500)

randomness = dict(seed=0)

default_hooks = dict(
    checkpoint=dict(interval=250, max_keep_ckpts=1, save_best="AveragePrecision", rule="greater"),
    logger=dict(interval=10),
)
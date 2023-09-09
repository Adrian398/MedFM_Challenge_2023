# dataset settings
dataset_type = 'Endoscopy'
data_preprocessor = dict(
    num_classes=4,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
    to_onehot=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/scratch/medfm/medfm-challenge/data/MedFMC_train/endo/images',
        ann_file='data_anns/MedFMC/endo/train_20.txt',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/scratch/medfm/medfm-challenge/data/MedFMC_train/endo/images',
        ann_file='data_anns/MedFMC/endo/val_20.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/scratch/medfm/medfm-challenge/data/MedFMC_train/endo/images',
        ann_file='data_anns/MedFMC/endo/test_WithLabel.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

train_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='Aggregate'),
    dict(type='AUC')
]
val_evaluator = train_evaluator
test_evaluator = train_evaluator


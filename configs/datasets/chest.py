# dataset settings
dataset_type = 'Chest19'
data_preprocessor = dict(
    num_classes=19,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
    to_onehot=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='NumpyToPIL', to_rgb=True),
    dict(type='torchvision/RandomAffine', degrees=(-15, 15), translate=(0.05, 0.05), fill=128),
    dict(type='PILToNumpy', to_bgr=True),
    dict(type='RandomResizedCrop', scale=384, crop_ratio_range=(0.9, 1.0), backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
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
        data_prefix='data/MedFMC_train/chest/images',
        ann_file='data_anns/MedFMC/chest/train_20.txt',
        pipeline=train_pipeline,),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/MedFMC_train/chest/images',
        ann_file='data_anns/MedFMC/chest/val_20.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/MedFMC_train/chest/images',
        ann_file='/scratch/medfm/data_anns/MedFMC/chest/test_WithLabel.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

train_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='AUC', multilabel=True),
    dict(type='Aggregate', multilabel=True)]
val_evaluator = train_evaluator
test_evaluator = train_evaluator

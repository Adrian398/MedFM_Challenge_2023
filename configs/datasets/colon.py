# dataset settings
dataset_type = 'Colon'
data_preprocessor = dict(
    num_classes=2,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True, # convert image from BGR to RGB
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='ColorJitter', hue=0.3, brightness=0.4, contrast=0.4, saturation=0.4),
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
        data_prefix='/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
        ann_file='data_anns/MedFMC/colon/train_20.txt',
        pipeline=train_pipeline,),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
        ann_file='data_anns/MedFMC/colon/val_20.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/scratch/medfm/medfm-challenge/data/MedFMC_train/colon/images',
        ann_file='data_anns/MedFMC/colon/test_WithLabel.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

train_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='Accuracy', topk=(1,)),
    dict(type='SingleLabelMetric', items=['precision', 'recall']),
    dict(type='Aggregate'),
    dict(type='AUC')
]
val_evaluator = train_evaluator
test_evaluator = train_evaluator
# dataset settings

classes = ('Small Car', 'Van', 'Dump Truck', 'Cargo Truck',
           'Bus', 'Truck Tractor', 'Trailer', 'Excavator',
           'Tractor', 'Motorboat', 'Dry Cargo Ship',
           'Liquid Cargo Ship', 'Fishing Boat', 'Engineering Ship',
           'Passenger Ship', 'Warship', 'Tugboat', 'Boeing737', 
           'Boeing747', 'Boeing777', 'Boeing787', 'A220', 'A321',
            'A330', 'A350', 'C919', 'ARJ21')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(1333, 1024), keep_ratio=True),
    dict(type='Resize', 
         img_scale=[(1333, 640), (1333, 672), (1333, 704),
                    (1333, 736), (1333, 768), (1333, 800)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
data_root = 'data/osodd_data/osodd_fair1m/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train_all75.json',
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val_all.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_test_all.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline))

evaluation = dict(interval=5, metric='bbox')

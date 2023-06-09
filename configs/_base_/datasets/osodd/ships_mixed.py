# dataset settings

classes = ('Enterprise',
           'Nimitz',
           'Midway',
           'Ticonderoga',
           'Atago DD',
           'Arleigh Burke DD',
           'Hatsuyuki DD',
           'Hyuga DD',
           'Asagiri DD',
           'Perry FF',
           'Patrol',
           'YuTing LL',
           'YuDeng LL',
           'YuDao LL',
           'YuZhao LL',
           'Austin LL',
           'Osumi LL',
           'LSD 41 LL',
           'LHA LL',
           'Commander',
           'Medical Ship',
           'Test Ship',
           'Training Ship',
           'AOE',
           'Sanantonio AS',
           'EPF',
           'Container Ship',
           'Cargo',
           'Barge',
           'Tugboat',
           'Yacht',
           'Sailboat',
           'Fishing Vessel',
           'Oil Tanker',
           'Motorboat',
           )
classes_u = classes + ('UNKNOWN',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(1333, 930), keep_ratio=True),
    dict(type='Resize', 
         img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768),
                    (1333, 800), (1333, 832), (1333, 864), (1333, 896), (1333, 928)],
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
        img_scale=(1333, 930),
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
data_root = 'data/osodd_data/osodd_ships/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes_u,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes_u,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline),
    osodd_val=dict(
        type=dataset_type,
        classes=classes_u,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline),
    osodd_test=dict(
        type=dataset_type,
        classes=classes_u,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric='bbox')

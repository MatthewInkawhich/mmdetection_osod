# dataset settings

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 
           'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench',
           'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
           'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
           'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
           'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush'
           )
classes_u = classes + ('UNKNOWN',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', 
         img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768),
                    (1333, 800)],
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
data_root = 'data/coco/'
osodd_data_root = 'data/osodd_data/osodd_coco365/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations2017/annotations/instances_train2017.json',
        img_prefix=data_root + 'images2017/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations2017/annotations/instances_val2017.json',
        img_prefix=data_root + 'images2017/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations2017/annotations/instances_val2017.json',
        img_prefix=data_root + 'images2017/val2017/',
        pipeline=test_pipeline),
    osodd_val=dict(
        type=dataset_type,
        classes=classes_u,
        ann_file=osodd_data_root + 'annotations/instances_val.json',
        img_prefix=osodd_data_root + 'images/val/',
        pipeline=test_pipeline),
    osodd_test=dict(
        type=dataset_type,
        classes=classes_u,
        ann_file=osodd_data_root + 'annotations/instances_test.json',
        img_prefix=osodd_data_root + 'images/test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

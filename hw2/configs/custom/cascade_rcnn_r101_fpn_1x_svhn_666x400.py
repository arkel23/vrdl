_base_ = '../cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10)]))

load_from = 'checkpoints/cascade_rcnn_r101_fpn_1x_coco.pth'

# dataset settings
dataset_type = 'CocoDataset'
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(666, 400), keep_ratio=True),
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
        # img_scale=(1333, 800),
        img_scale=(666, 400),
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
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        filter_empty_gt=False),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'train/',
        pipeline=test_pipeline,
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        filter_empty_gt=False))
evaluation = dict(interval=1, metric='bbox')

runner = dict(type='EpochBasedRunner', max_epochs=20)

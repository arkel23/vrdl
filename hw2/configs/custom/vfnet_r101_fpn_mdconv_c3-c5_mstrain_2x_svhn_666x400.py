_base_ = '../vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    bbox_head=dict(
        num_classes=10))

load_from = 'checkpoints/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.pth'

# dataset settings
dataset_type = 'CocoDataset'
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
data_root = 'data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(666, 320), (666, 400)],
        multiscale_mode='range',
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
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'train.json',
            img_prefix=data_root + 'train/',
            filter_empty_gt=False,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'train/',
        filter_empty_gt=False,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'test/',
        filter_empty_gt=False,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

lr_config = dict(step=[5, 8])
runner = dict(type='EpochBasedRunner', max_epochs=10)

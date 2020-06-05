_base_ = [
    '../models/faster_rcnn_r50_fpn.py',
    #'../datasets/AN2_detection.py', #change
    '../schedules/schedule_2x.py', '../default_runtime.py'
]

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
        
# dataset settings
dataset_type = 'MyDataset'
data_root = '../../../AN-2'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=2000),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Collect', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
]

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/train/_annotations.coco.json',
        img_prefix=data_root + '/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/valid/_annotations.coco.json',
        img_prefix=data_root + '/valid/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/test/_annotations.coco.json',
        img_prefix=data_root + '/test/',
        pipeline=test_pipeline))

total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

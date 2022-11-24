import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

sequence_term='long_term'
sequence_mode = 'online' # Not use
seq_len = 3
gt_sample_mode='temporal'

# model settings
model = dict(
    type="MGTANet",
    seq_len = seq_len,
    sequence_mode = sequence_mode,
    pretrained='./pretrained_weight/centerpoint_mat_vfe_59.85.pth',
    reader=dict(
        type="SM_VFE",
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=21, ds_factor=8
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    alignment=dict(
        type="MGDA",
        input_channel_list=[256,128],
        encode_out_channel=64,
        target_channel=512,
        is_shared=False,
        mot_mode='default',
        is_down_sample=True,
        seq_len=seq_len,
        sequence_mode=sequence_mode,
        logger=logging.getLogger("MGDA"),
    ),
    aggregation=dict(
        type="STFA",
        src_in_channels=128,
        target_in_channels=512,
        feat_h=180,
        feat_w=180,
        seq_len = seq_len,
        with_pos_emb = False,
        encoder_cfg=dict(
            feedforward_channel=512,
            dropout=0.1,
            activation='relu',
            num_heads=8,
            enc_num_points=4,
            num_layers=3,
            num_levels=1,
            seq_len=seq_len
        ),
        logger=logging.getLogger("STFA"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
        share_conv_channel=64,
        dcn_head=False
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-54, -54],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.075, 0.075]
)

# dataset settings
dataset_type = "NuScenesDatasetVID"
nsweeps = 10
data_root = "/mnt/dataset/nuscenes/v1.0-trainval"

# No GT-AUG for last 10 epochs
db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path_list=[
                    "/mnt/dataset/nuscenes/v1.0-trainval/dbinfos_train_10sweeps_withvelo_gt_token.pkl",
                    "/mnt/dataset/nuscenes/v1.0-trainval/temporal_idx_t-1_dbinfos.pkl",
                    "/mnt/dataset/nuscenes/v1.0-trainval/temporal_idx_t+1_dbinfos.pkl"
                ],
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        dict(barrier=2),
        dict(motorcycle=6),
        dict(bicycle=6),
        dict(pedestrian=2),
        dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
    seq_len=seq_len,
    sequence_mode=sequence_mode
)

# train_preprocessor = dict(
#     mode="train",
#     shuffle_points=True,
#     global_rot_noise=[-0.78539816, 0.78539816],
#     global_scale_noise=[0.9, 1.1],
#     global_translate_std=0.5,
#     db_sampler=db_sampler,
#     class_names=class_names,
#     seq_len=seq_len,
#     sequence_mode=sequence_mode
# )

# No GT-AUG for last 10 epochs
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.9, 1.1],
    global_translate_std=0.5,
    db_sampler=None,
    class_names=class_names,
    seq_len=seq_len,
    sequence_mode=sequence_mode
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

voxel_generator = dict(
    range=[-54, -54, -5.0, 54, 54, 3.0],
    voxel_size=[0.075, 0.075, 0.2],
    max_points_in_voxel=10,
    max_voxel_num=[120000, 160000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, 
         seq_len = seq_len, sequence_mode = sequence_mode),
    dict(type="LoadPointCloudAnnotations", with_bbox=True, 
         seq_len = seq_len, sequence_mode=sequence_mode),
    dict(type="PreprocessVID", cfg=train_preprocessor, 
         seq_len = seq_len, sequence_mode=sequence_mode, gt_sample_mode=gt_sample_mode),
    dict(type="VoxelizationVID", cfg=voxel_generator, 
         seq_len = seq_len, sequence_mode=sequence_mode),
    dict(type="AssignLabelVID", cfg=train_cfg["assigner"], 
         seq_len = seq_len, sequence_mode=sequence_mode),
    dict(type="Reformat", seq_len = seq_len, sequence_mode=sequence_mode),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, 
         seq_len = seq_len, sequence_mode=sequence_mode),
    dict(type="LoadPointCloudAnnotations", with_bbox=True, 
         seq_len = seq_len, sequence_mode=sequence_mode),
    dict(type="PreprocessVID", cfg=val_preprocessor, 
         seq_len = seq_len, sequence_mode=sequence_mode),
    dict(type="VoxelizationVID", cfg=voxel_generator,
         seq_len = seq_len, sequence_mode=sequence_mode),
    dict(type="AssignLabelVID", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "/mnt/dataset/nuscenes/v1.0-trainval/infos_train_10sweeps_withvelo_filter_True_offline_{}.pkl".format(seq_len)
val_anno = "/mnt/dataset/nuscenes/v1.0-trainval/infos_val_10sweeps_withvelo_filter_True_offline_{}.pkl".format(seq_len)
test_anno = None

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.0002, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 40
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/mgtanet/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]

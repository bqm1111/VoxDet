# TartanAir Dataset Configuration for Occupancy Prediction
# Matches the format of voxdet-semantickitti-cam.py

data_root = "data/tartanair"  # Update with your TartanAir path
ann_file = "data/tartanair"  # Same as data_root if voxel labels are co-located
depth_root = "data/tartanair"  # Same as data_root

camera_used = ["front"]  # or 'lcam_front'

dataset_type = "TartanAirDataset"

# Point cloud range and voxel size
# TartanAir uses camera frame: X-right, Y-down, Z-forward
# Adjust based on your generate_voxels_robust.py settings
# point_cloud_range = [
#     -25.6,
#     -25.6,
#     0.0,
#     25.6,
#     6.4,
#     51.2,
# ]  # [x_min, y_min, z_min, x_max, y_max, z_max]
# occ_size = [256, 160, 256]  # [X, Y, Z] voxel grid dimensions
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]  # SemanticKITTI standard
occ_size = [256, 256, 32]  # [X, Y, Z] - SemanticKITTI standard

# TartanAir class frequencies (placeholder - compute from your dataset)
# These are from the LEARNING_MAP in generate_voxels_robust.py (34 classes)
tartanair_class_frequencies = [
    5.00e9,  # 0: unlabeled/empty
    1.00e7,  # 1: cabinet
    1.00e6,  # 2: cable
    1.00e6,  # 3: car
    1.00e6,  # 4: ceiling
    1.00e6,  # 5: cementcolumn
    1.00e6,  # 6: chair
    1.00e6,  # 7: chasis
    1.00e6,  # 8: cieling
    1.00e7,  # 9: door
    1.00e8,  # 10: floor
    1.00e5,  # 11: keyboard
    1.00e6,  # 12: lamp
    1.00e7,  # 13: light
    1.00e7,  # 14: metalcieling
    1.00e7,  # 15: metalfloor
    1.00e6,  # 16: metalhandrail
    1.00e7,  # 17: metalpanel
    1.00e7,  # 18: metalplatform
    1.00e6,  # 19: metalpole
    1.00e6,  # 20: metalramp
    1.00e6,  # 21: metalstair
    1.00e5,  # 22: monitor
    1.00e6,  # 23: pipecover
    1.00e6,  # 24: platform
    1.00e5,  # 25: plug
    1.00e6,  # 26: robotarm
    1.00e8,  # 27: sky
    1.00e6,  # 28: table
    1.00e6,  # 29: tireassembly
    1.00e6,  # 30: toolbox
    1.00e6,  # 31: ventpipe
    1.00e5,  # 32: ventpipeclamp
    1.00e8,  # 33: wall
]

# 34 classes for TartanAir (from LEARNING_MAP)
class_names = [
    "unlabeled",  # 0
    "cabinet",  # 1
    "pipecover",  # 2
    "metalpanel",  # 3
    "metalhandrail",  # 4
    "light",  # 5
    "cieling",  # 6
    "metalplatform",  # 7
    "chair",  # 8
    "cementcolumn",  # 9
    "plug",  # 10
    "ceiling",  # 11
    "metalpole",  # 12
    "ventpipe",  # 13
    "metalramp",  # 14
    "car",  # 15
    "metalfloor",  # 16
    "toolbox",  # 17
    "ventpipeclamp",  # 18
    "metalstair",  # 19
    "lamp",  # 20
    "tireassembly",  # 21
    "metalcieling",  # 22
    "platform",  # 23
    "monitor",  # 24
    "wall",  # 25
    "door",  # 26
    "sky",  # 27
    "cable",  # 28
    "chasis",  # 29
    "floor",  # 30
    "table",  # 31
    "robotarm",  # 32
    "keyboard",  # 33
]
num_class = len(class_names)

# BDA (Bird's-eye-view Data Augmentation) config
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0,
)

# Data config for image preprocessing
# TartanAir images are 640x640, adjust input_size as needed
data_config = {
    "input_size": (320, 448),  # (H, W) - adjust based on your model requirements
    "resize": (0.0, 0.0),  # No random resize for now
    "rot": (0.0, 0.0),  # No random rotation
    "flip": False,  # No random flip
    "crop_h": (0.0, 0.0),  # No random crop
    "resize_test": 0.00,
}

# Sequences configuration
# Define your train/val/test split here
# Format: list of (env_name, difficulty, trajectory_id) tuples
train_sequences = [
    ("CarWelding", "Data_easy", "P000"),
    # ("CarWelding", "Data_easy", "P001"),
    # ("CarWelding", "Data_easy", "P002"),
    # Add more sequences...
]

val_sequences = [
    ("CarWelding", "Data_easy", "P008"),
    # Add more sequences...
]

test_sequences = val_sequences  # Same as val for now

# Training pipeline
train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles_TartanAir",
        data_config=data_config,
        load_depth=True,
        is_train=True,
        color_jitter=(0.4, 0.4, 0.4),
    ),
    dict(type="CreateDepthFromTartanAir", data_root=data_root),
    dict(
        type="LoadTartanAirAnnotation",
        bda_aug_conf=bda_aug_conf,
        apply_bda=False,
        is_train=True,
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type="CollectData",
        keys=["img_inputs", "gt_occ"],
        meta_keys=[
            "pc_range",
            "occ_size",
            "raw_img",
            "stereo_depth",
            "focal_length",
            "baseline",
            "img_shape",
            "gt_depths",
        ],
    ),
]

# Training dataset config
trainset_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file,
    depth_root=depth_root,
    pipeline=train_pipeline,
    split="train",
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=False,
    sequences=train_sequences,
)

# Test pipeline
test_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles_TartanAir",
        data_config=data_config,
        load_depth=True,
        is_train=False,
        color_jitter=None,
    ),
    dict(type="CreateDepthFromTartanAir", data_root=data_root),
    dict(
        type="LoadTartanAirAnnotation",
        bda_aug_conf=bda_aug_conf,
        apply_bda=False,
        is_train=False,
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type="CollectData",
        keys=["img_inputs", "gt_occ"],
        meta_keys=[
            "pc_range",
            "occ_size",
            "sequence",
            "frame_id",
            "raw_img",
            "stereo_depth",
            "focal_length",
            "baseline",
            "img_shape",
            "gt_depths",
        ],
    ),
]

# Test dataset config
testset_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file,
    depth_root=depth_root,
    pipeline=test_pipeline,
    split="test",
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=True,
    sequences=test_sequences,
)

# Data config
data = dict(train=trainset_config, val=testset_config, test=testset_config)

# Dataloader config
train_dataloader_config = dict(batch_size=1, num_workers=4)

test_dataloader_config = dict(batch_size=1, num_workers=4)

# Model config
numC_Trans = 128
lss_downsample = [2, 2, 2]
voxel_out_channels = [128]
norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

grid_config = {
    "xbound": [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    "ybound": [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    "zbound": [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    "dbound": [0.5, 56.5, 0.5],  # Adjusted for TartanAir depth range
}

_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_levels_ = 1
_num_cams_ = 1  # TartanAir single camera
_dim_ = 128

model = dict(
    type="VoxDet",
    car_scale_filter_max=[30, 30, 30],
    use_gt_refine=True,
    img_backbone=dict(
        type="CustomResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        pretrained=True,
        track_running_stats=True,
    ),
    img_neck=dict(
        type="SECONDFPN",
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.5, 1, 2, 4],
        out_channels=[128, 128, 128, 128],
    ),
    depth_net=dict(
        type="GeometryDepth_Net",
        downsample=8,
        numC_input=512,
        numC_Trans=numC_Trans,
        cam_channels=33,
        grid_config=grid_config,
        loss_depth_type="kld",
        loss_depth_weight=0.0001,
    ),
    img_view_transformer=dict(
        type="LSSViewTransformer",
        downsample=8,
        grid_config=grid_config,
        data_config=data_config,
    ),
    proposal_layer=dict(
        type="VoxelProposalLayer",
        point_cloud_range=point_cloud_range,
        input_dimensions=[128, 128, 16],
        data_config=data_config,
        init_cfg=None,
    ),
    VoxFormer_head=dict(
        type="VoxFormerHeadCrossAttention",
        volume_h=128,
        volume_w=128,
        volume_z=16,
        data_config=data_config,
        point_cloud_range=point_cloud_range,
        embed_dims=_dim_,
        cross_transformer=dict(
            type="PerceptionTransformer_DFA3D",
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=_dim_,
            num_cams=_num_cams_,
            encoder=dict(
                type="VoxFormerEncoder_DFA3D",
                num_layers=_num_layers_cross_,
                pc_range=point_cloud_range,
                data_config=data_config,
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type="VoxFormerLayer",
                    attn_cfgs=[
                        dict(
                            type="DeformCrossAttention_DFA3D",
                            pc_range=point_cloud_range,
                            num_cams=_num_cams_,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D_DFA3D",
                                embed_dims=_dim_,
                                num_points=_num_points_cross_,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=_dim_,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    feedforward_channels=_dim_ * 2,
                    ffn_dropout=0.1,
                    operation_order=("cross_attn", "norm", "ffn", "norm"),
                ),
            ),
        ),
        mlp_prior=True,
    ),
    occ_encoder_backbone=dict(
        type="Ident",
        embed_dims=128,
        local_aggregator=dict(
            type="VoxelAggregatorDual",
            local_encoder_backbone=dict(
                type="CustomResNet3D",
                numC_input=128,
                num_layer=[2, 2, 2],
                num_channels=[128, 128, 128],
                stride=[1, 2, 2],
                norm_cfg=norm_cfg,
                drop_path_rate=0.3,
            ),
            local_encoder_neck=dict(
                type="SpatiallyDecoupledFPN",
                share_fpn=False,
                in_channels=[128, 128, 128],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=norm_cfg,
                conv_cfg=dict(type="Conv3d"),
                act_cfg=dict(type="ReLU", inplace=True),
                upsample_cfg=dict(mode="trilinear", align_corners=False),
            ),
        ),
    ),
    pts_bbox_head_aux=dict(
        type="OccHead",
        in_channels=[sum(voxel_out_channels)],
        out_channel=num_class,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=occ_size,
        loss_weight_cfg={
            "loss_voxel_ce_weight": 0.2,
            "loss_voxel_sem_scal_weight": 0.2,
            "loss_voxel_geo_scal_weight": 0.2,
        },
        conv_cfg=dict(type="Conv3d", bias=False),
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        balance_cls_weight=False,
        class_frequencies=tartanair_class_frequencies,
        num_classes=34,
    ),
    pts_bbox_head=dict(
        type="VoxDetHead",
        down_sampling_ratio=0.5,
        balance_reg_loss="none",
        box_down_sample="trilinear",
        align_corners=False,
        num_inst_layer=4,
        use_bias=False,
        isolation_scale=0,
        pred_six_directions=True,
        in_channels=[sum(voxel_out_channels)],
        out_channel=num_class,
        empty_idx=0,
        num_level=1,
        with_cp=False,
        occ_size=occ_size,
        loss_weight_cfg={
            "loss_voxel_ce_weight": 3.0,
            "loss_voxel_sem_scal_weight": 1.0,
            "loss_voxel_geo_scal_weight": 1.0,
            "loss_voxel_ctr_weight": 1.0,
        },
        conv_cfg=dict(type="Conv3d", bias=False),
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        balance_cls_weight=False,
        class_frequencies=tartanair_class_frequencies,
        num_classes=34
    ),
)

# Training params
learning_rate = 3e-4
training_steps = 25000

optimizer = dict(type="AdamW", lr=learning_rate, weight_decay=0.01)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy="cos",
    interval="step",
    frequency=1,
)

optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# Pretrained model path (update as needed)
load_from = "ckpts/preatrain_depth_model.ckpt"


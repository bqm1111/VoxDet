from mmdet3d_plugin import *
from torch.utils.data import DataLoader

# dataset config #
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0,
)

data_config = {
    "input_size": (384, 1280),
    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    # 'flip': True,
    "resize": (0.0, 0.0),
    "rot": (0.0, 0.0),
    "flip": (0.0, 0.0),
    "flip": False,
    "crop_h": (0.0, 0.0),
    "resize_test": 0.00,
}

point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
data_root = "data/kitti/dataset"
test_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        data_config=data_config,
        load_stereo_depth=True,
        is_train=False,
        color_jitter=None,
    ),
    dict(type="CreateDepthFromLiDAR", data_root=data_root, dataset="kitti"),
    dict(
        type="LoadAnnotationOcc",
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

train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        data_config=data_config,
        load_stereo_depth=True,
        is_train=True,
        color_jitter=(0.4, 0.4, 0.4),
    ),
    dict(
        type="CreateDepthFromLiDAR",
        data_root=data_root,
        dataset="kitti",
        load_seg=False,
    ),
    dict(
        type="LoadAnnotationOcc",
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
test_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        data_config=data_config,
        load_stereo_depth=True,
        is_train=False,
        color_jitter=None,
    ),
    dict(type="CreateDepthFromLiDAR", data_root=data_root, dataset="kitti"),
    dict(
        type="LoadAnnotationOcc",
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

dataset = SemanticKITTIDataset(
    data_root="data/kitti/dataset",
    ann_file="data/kitti/dataset/labels/",
    stereo_depth_root="data/kitti/dataset/depth/",
    camera_used=["left"],
    occ_size=[256, 256, 32],
    pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
    split="test",
    pipeline=test_pipeline,
    test_mode=True,
)

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)
import torch

for idx, data in enumerate(dataloader):
    if idx == 0:
        print(data.keys())
        print(torch.unique(data["gt_occ"]))
        print(data["gt_occ"].shape)
        for k in data.keys():
            if k != "gt_occ":
                print(f"data[{k}] = {data[k]}")

        break


from configs.voxdet_tartanair_cam import *
from torch.utils.data import DataLoader
from voxdet_models.datasets.tartanair import TartanAirDataset
import torch

dataset = TartanAirDataset(
    data_root=data_root,
    ann_file=ann_file,
    depth_root=depth_root,
    camera_used=["lcam_front"],
    occ_size=[256, 256, 32],
    pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
    split="test",
    pipeline=test_pipeline,
    test_mode=True,
)

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)

for idx, data in enumerate(dataloader):
    if idx == 0:
        print(data.keys())
        print(torch.unique(data["gt_occ"]))
        print(data["gt_occ"].shape)
        # for k in data.keys():
        #     if k != "gt_occ":
        #         print(f"data[{k}] = {data[k]}")

        break

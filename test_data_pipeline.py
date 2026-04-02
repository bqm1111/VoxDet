"""
Unit tests for the SemanticKITTI data pipeline.

Usage:
    python test_data_pipeline.py                  # run all tests on 1 sample
    python test_data_pipeline.py --full            # also test DataLoader batching
    python test_data_pipeline.py --index 5         # test a specific sample index

Tests each pipeline stage independently, then end-to-end with DataLoader.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

# Register all modules before anything else
from voxdet_core import Config, PIPELINES, DATASETS
from voxdet_core.data import Compose
from voxdet_models import *  # noqa: F401,F403 — triggers registry

CONFIG_PATH = "configs/voxdet-semantickitti-cam.py"
NUM_CLASSES = 20
OCC_SHAPE = (256, 256, 32)


def load_config():
    cfg = Config.fromfile(CONFIG_PATH)
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _ok(msg):
    print(f"  [PASS] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _assert(cond, msg):
    if cond:
        _ok(msg)
    else:
        _fail(msg)


# ---------------------------------------------------------------------------
# Test 1: Dataset loading & get_data_info
# ---------------------------------------------------------------------------

def test_dataset_loading(cfg, idx=0):
    _sep("Test 1: Dataset loading & get_data_info")

    dataset = DATASETS.build(cfg.data.test)
    _assert(len(dataset) > 0, f"Dataset has {len(dataset)} samples")

    info = dataset.get_data_info(idx)

    # Check all expected keys
    expected_keys = [
        "occ_size", "pc_range", "sequence", "frame_id",
        "img_filename", "lidar2img", "cam_intrinsic", "lidar2cam",
        "focal_length", "baseline", "stereo_depth_path", "gt_occ",
    ]
    for k in expected_keys:
        _assert(k in info, f"Key '{k}' present in info dict")

    # Check shapes
    n_cams = len(cfg.camera_used)
    _assert(len(info["img_filename"]) == n_cams,
            f"img_filename has {n_cams} entries")
    _assert(len(info["cam_intrinsic"]) == n_cams,
            f"cam_intrinsic has {n_cams} entries")
    _assert(len(info["lidar2cam"]) == n_cams,
            f"lidar2cam has {n_cams} entries")

    # Intrinsics and extrinsics are 4x4
    P = info["cam_intrinsic"][0]
    _assert(P.shape == (4, 4), f"cam_intrinsic shape is {P.shape} (expect 4x4)")
    T = info["lidar2cam"][0]
    _assert(T.shape == (4, 4), f"lidar2cam shape is {T.shape} (expect 4x4)")

    # File existence
    for img_path in info["img_filename"]:
        _assert(os.path.exists(img_path), f"Image exists: {os.path.basename(img_path)}")

    _assert(os.path.exists(info["stereo_depth_path"]),
            f"Stereo depth exists: {os.path.basename(info['stereo_depth_path'])}")

    # gt_occ
    if info["gt_occ"] is not None:
        _assert(info["gt_occ"].shape == OCC_SHAPE,
                f"gt_occ shape is {info['gt_occ'].shape}")
    else:
        _ok("gt_occ is None (test-submit split)")

    return dataset, info


# ---------------------------------------------------------------------------
# Test 2: LoadMultiViewImageFromFiles
# ---------------------------------------------------------------------------

def test_load_images(cfg, info):
    _sep("Test 2: LoadMultiViewImageFromFiles")

    test_pipe_cfg = cfg.data.test.pipeline[0]
    transform = PIPELINES.build(test_pipe_cfg)
    result = transform(dict(info))  # copy so we don't mutate

    _assert("img_inputs" in result, "img_inputs key present after LoadMultiViewImageFromFiles")

    img_inputs = result["img_inputs"]
    _assert(len(img_inputs) == 7,
            f"img_inputs is a tuple of {len(img_inputs)} tensors (expect 7)")

    imgs, rots, trans, intrins, post_rots, post_trans, cam2lidars = img_inputs
    n_cams = len(cfg.camera_used)
    fH, fW = cfg.data_config["input_size"]

    # imgs: (N_cams, 3, H, W)
    _assert(imgs.shape == (n_cams, 3, fH, fW),
            f"imgs shape {tuple(imgs.shape)} == ({n_cams}, 3, {fH}, {fW})")
    _assert(imgs.dtype == torch.float32, f"imgs dtype is {imgs.dtype}")

    # Normalized pixel values (not [0, 255])
    _assert(imgs.abs().max() < 20.0,
            f"imgs appear normalized (max abs = {imgs.abs().max():.2f})")

    # rots: (N_cams, 3, 3)
    _assert(rots.shape == (n_cams, 3, 3), f"rots shape {tuple(rots.shape)}")
    # trans: (N_cams, 3)
    _assert(trans.shape == (n_cams, 3), f"trans shape {tuple(trans.shape)}")
    # intrins: (N_cams, 4, 4)
    _assert(intrins.shape == (n_cams, 4, 4), f"intrins shape {tuple(intrins.shape)}")
    # post_rots: (N_cams, 3, 3)
    _assert(post_rots.shape == (n_cams, 3, 3), f"post_rots shape {tuple(post_rots.shape)}")
    # post_trans: (N_cams, 3)
    _assert(post_trans.shape == (n_cams, 3), f"post_trans shape {tuple(post_trans.shape)}")
    # cam2lidars: (N_cams, 4, 4)
    _assert(cam2lidars.shape == (n_cams, 4, 4), f"cam2lidars shape {tuple(cam2lidars.shape)}")

    # stereo_depth loaded
    if cfg.data.test.pipeline[0].get("load_stereo_depth", False):
        _assert("stereo_depth" in result, "stereo_depth key present")
        sd = result["stereo_depth"]
        _assert(sd.shape[-2:] == (fH, fW),
                f"stereo_depth spatial shape {tuple(sd.shape[-2:])} matches input_size")

    # raw_img
    _assert("raw_img" in result, "raw_img key present")
    _assert(len(result["raw_img"]) == n_cams,
            f"raw_img has {n_cams} entries")

    return result


# ---------------------------------------------------------------------------
# Test 3: CreateDepthFromLiDAR
# ---------------------------------------------------------------------------

def test_create_depth(cfg, result):
    _sep("Test 3: CreateDepthFromLiDAR")

    depth_pipe_cfg = cfg.data.test.pipeline[1]
    transform = PIPELINES.build(depth_pipe_cfg)
    result = transform(dict(result))

    _assert("gt_depths" in result, "gt_depths key present after CreateDepthFromLiDAR")

    gt_depths = result["gt_depths"]
    imgs = result["img_inputs"][0]
    n_cams, _, fH, fW = imgs.shape

    _assert(gt_depths.shape == (n_cams, fH, fW),
            f"gt_depths shape {tuple(gt_depths.shape)} == ({n_cams}, {fH}, {fW})")
    _assert(gt_depths.dtype == torch.float32,
            f"gt_depths dtype is {gt_depths.dtype}")
    _assert((gt_depths >= 0).all(),
            "gt_depths values are non-negative")

    nonzero = (gt_depths > 0).sum().item()
    total = gt_depths.numel()
    sparsity = 1.0 - nonzero / total
    _assert(nonzero > 0,
            f"gt_depths has {nonzero} nonzero pixels ({sparsity*100:.1f}% sparse)")
    _assert(sparsity > 0.9,
            f"gt_depths is sparse as expected (sparsity={sparsity*100:.1f}%)")

    # Depth range sanity (SemanticKITTI: typically 2-80m)
    valid_depths = gt_depths[gt_depths > 0]
    _assert(valid_depths.min() > 0.5,
            f"min valid depth = {valid_depths.min():.2f}m (> 0.5m)")
    _assert(valid_depths.max() < 120.0,
            f"max valid depth = {valid_depths.max():.2f}m (< 120m)")

    return result


# ---------------------------------------------------------------------------
# Test 4: LoadAnnotationOcc
# ---------------------------------------------------------------------------

def test_load_annotation(cfg, result):
    _sep("Test 4: LoadAnnotationOcc")

    occ_pipe_cfg = cfg.data.test.pipeline[2]
    transform = PIPELINES.build(occ_pipe_cfg)
    result = transform(dict(result))

    # img_inputs should now be 8-tuple (bda_rot added)
    _assert(len(result["img_inputs"]) == 8,
            f"img_inputs is now {len(result['img_inputs'])}-tuple (bda_rot added)")

    bda_rot = result["img_inputs"][6]
    _assert(bda_rot.shape == (4, 4), f"bda_rot shape is {tuple(bda_rot.shape)}")

    # gt_occ
    gt_occ = result["gt_occ"]
    _assert(isinstance(gt_occ, torch.Tensor), "gt_occ is a torch.Tensor")
    _assert(gt_occ.dtype == torch.long, f"gt_occ dtype is {gt_occ.dtype} (expect long)")
    _assert(gt_occ.shape == OCC_SHAPE,
            f"gt_occ shape {tuple(gt_occ.shape)} == {OCC_SHAPE}")
    _assert(gt_occ.min() >= 0, f"gt_occ min = {gt_occ.min()} (>= 0)")
    _assert(gt_occ.max() < NUM_CLASSES,
            f"gt_occ max = {gt_occ.max()} (< {NUM_CLASSES})")

    # img_shape
    _assert("img_shape" in result, "img_shape key present")
    fH, fW = cfg.data_config["input_size"]
    _assert(tuple(result["img_shape"]) == (fH, fW),
            f"img_shape = {result['img_shape']}")

    return result


# ---------------------------------------------------------------------------
# Test 5: CollectData
# ---------------------------------------------------------------------------

def test_collect_data(cfg, result):
    _sep("Test 5: CollectData")

    collect_pipe_cfg = cfg.data.test.pipeline[3]
    transform = PIPELINES.build(collect_pipe_cfg)
    data = transform(dict(result))

    # Final output should have img_inputs, gt_occ, img_metas
    _assert("img_inputs" in data, "img_inputs in final output")
    _assert("gt_occ" in data, "gt_occ in final output")
    _assert("img_metas" in data, "img_metas in final output")

    # Check meta_keys that should be present
    meta_keys_expected = collect_pipe_cfg["meta_keys"]
    for k in meta_keys_expected:
        if k in result:
            _assert(k in data["img_metas"],
                    f"meta key '{k}' present in img_metas")

    # No extra data keys beyond what's specified
    allowed = set(collect_pipe_cfg["keys"]) | {"img_metas"}
    _assert(set(data.keys()) == allowed,
            f"output keys = {set(data.keys())}")

    return data


# ---------------------------------------------------------------------------
# Test 6: Full pipeline via dataset[idx]
# ---------------------------------------------------------------------------

def test_full_pipeline(cfg, idx=0):
    _sep("Test 6: Full pipeline end-to-end via dataset[idx]")

    dataset = DATASETS.build(cfg.data.test)
    t0 = time.time()
    sample = dataset[idx]
    elapsed = time.time() - t0

    _assert(sample is not None, f"dataset[{idx}] returned a sample")
    _assert("img_inputs" in sample, "img_inputs in sample")
    _assert("gt_occ" in sample, "gt_occ in sample")
    _assert("img_metas" in sample, "img_metas in sample")

    # Verify shapes match expectations
    imgs = sample["img_inputs"][0]
    n_cams = len(cfg.camera_used)
    fH, fW = cfg.data_config["input_size"]
    _assert(imgs.shape == (n_cams, 3, fH, fW),
            f"end-to-end imgs shape {tuple(imgs.shape)}")
    _assert(sample["gt_occ"].shape == OCC_SHAPE,
            f"end-to-end gt_occ shape {tuple(sample['gt_occ'].shape)}")

    _ok(f"Pipeline took {elapsed:.2f}s for 1 sample")
    return dataset


# ---------------------------------------------------------------------------
# Test 7: DataLoader batch collation
# ---------------------------------------------------------------------------

def test_dataloader_batching(cfg, dataset):
    _sep("Test 7: DataLoader batch collation (batch_size=2)")

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    _assert("img_inputs" in batch, "img_inputs in batch")
    _assert("gt_occ" in batch, "gt_occ in batch")

    # img_inputs should each have batch dim prepended
    imgs = batch["img_inputs"][0]
    n_cams = len(cfg.camera_used)
    fH, fW = cfg.data_config["input_size"]
    _assert(imgs.shape == (2, n_cams, 3, fH, fW),
            f"batched imgs shape {tuple(imgs.shape)} == (2, {n_cams}, 3, {fH}, {fW})")

    gt_occ = batch["gt_occ"]
    _assert(gt_occ.shape == (2, *OCC_SHAPE),
            f"batched gt_occ shape {tuple(gt_occ.shape)} == (2, {', '.join(map(str, OCC_SHAPE))})")

    # All img_inputs tensors should have batch dim = 2
    for i, t in enumerate(batch["img_inputs"]):
        _assert(t.shape[0] == 2,
                f"img_inputs[{i}] batch dim = {t.shape[0]}")

    _ok("Batch collation works correctly")


# ---------------------------------------------------------------------------
# Test 8: Geometric sanity checks
# ---------------------------------------------------------------------------

def test_geometric_sanity(cfg, info):
    _sep("Test 8: Geometric sanity checks")

    # Build the first two transforms to get img_inputs with transforms applied
    t_img = PIPELINES.build(cfg.data.test.pipeline[0])
    result = t_img(dict(info))
    imgs, rots, trans, intrins, post_rots, post_trans, cam2lidars = result["img_inputs"]

    n_cams = len(cfg.camera_used)
    for cam_idx in range(n_cams):
        # cam2lidar is the inverse of lidar2cam
        cam2lidar = cam2lidars[cam_idx]  # (4, 4)
        lidar2cam_orig = torch.tensor(info["lidar2cam"][cam_idx], dtype=torch.float32)

        # Roundtrip: cam2lidar @ lidar2cam should be ~identity
        roundtrip = cam2lidar @ lidar2cam_orig
        identity = torch.eye(4)
        err = (roundtrip - identity).abs().max().item()
        _assert(err < 1e-4,
                f"cam[{cam_idx}] cam2lidar @ lidar2cam ~ I (max err={err:.2e})")

    # Verify rot/tran are consistent with cam2lidar
    for cam_idx in range(n_cams):
        c2l = cam2lidars[cam_idx]
        rot = rots[cam_idx]
        tran = trans[cam_idx]
        _assert(torch.allclose(c2l[:3, :3], rot, atol=1e-5),
                f"cam[{cam_idx}] rot == cam2lidar[:3,:3]")
        _assert(torch.allclose(c2l[:3, 3], tran, atol=1e-5),
                f"cam[{cam_idx}] tran == cam2lidar[:3, 3]")

    # LiDAR point projection sanity: load a few points, project, check bounds
    t_depth = PIPELINES.build(cfg.data.test.pipeline[1])
    result2 = t_depth(dict(result))
    gt_depths = result2["gt_depths"]
    fH, fW = imgs.shape[-2:]

    # Nonzero depth pixels should be within image bounds (they are by construction,
    # but verify the depth map dimensions match)
    _assert(gt_depths.shape[-2:] == (fH, fW),
            f"gt_depths spatial dims match image ({fH}x{fW})")

    # Check that projected depths land in the expected depth range from config
    dbound = cfg.grid_config["dbound"]
    valid = gt_depths[gt_depths > 0]
    in_range = ((valid >= dbound[0]) & (valid <= dbound[1])).float().mean().item()
    _assert(in_range > 0.8,
            f"{in_range*100:.1f}% of valid depths in dbound [{dbound[0]}, {dbound[1]}]m")

    _ok("Geometric consistency checks passed")


# ---------------------------------------------------------------------------
# Test 9: Train vs Test pipeline comparison
# ---------------------------------------------------------------------------

def test_train_vs_test(cfg, idx=0):
    _sep("Test 9: Train vs Test pipeline differences")

    # Build both datasets
    test_dataset = DATASETS.build(cfg.data.test)
    train_cfg = dict(cfg.data.train)
    train_cfg["split"] = "test"  # use same split so we get the same samples
    train_dataset = DATASETS.build(train_cfg)

    test_sample = test_dataset[idx]
    train_sample = train_dataset[idx]

    # Both should produce valid outputs
    _assert(test_sample is not None, "test pipeline produces output")
    _assert(train_sample is not None, "train pipeline produces output")

    # Same gt_occ (no BDA augmentation in this config)
    if not cfg.data.test.pipeline[2].get("apply_bda", False):
        _assert(torch.equal(test_sample["gt_occ"], train_sample["gt_occ"]),
                "gt_occ identical when apply_bda=False")

    # Train pipeline should have img_shape in img_metas
    _assert("img_shape" in train_sample["img_metas"],
            "train img_metas has img_shape")
    _assert("img_shape" in test_sample["img_metas"],
            "test img_metas has img_shape")

    # Test pipeline should have sequence and frame_id in meta
    _assert("sequence" in test_sample["img_metas"],
            "test img_metas has sequence")
    _assert("frame_id" in test_sample["img_metas"],
            "test img_metas has frame_id")

    _ok("Train/test pipeline comparison passed")


# ---------------------------------------------------------------------------
# Test 10: Dtype consistency
# ---------------------------------------------------------------------------

def test_dtypes(cfg, idx=0):
    _sep("Test 10: Dtype consistency")

    dataset = DATASETS.build(cfg.data.test)
    sample = dataset[idx]

    # img_inputs: all float32 except bda_rot
    float_names = ["imgs", "rots", "trans", "intrins",
                   "post_rots", "post_trans", "bda_rot", "cam2lidars"]
    for i, name in enumerate(float_names):
        t = sample["img_inputs"][i]
        _assert(t.dtype == torch.float32,
                f"img_inputs[{i}] ({name}) dtype = {t.dtype}")

    # gt_occ: long
    _assert(sample["gt_occ"].dtype == torch.long,
            f"gt_occ dtype = {sample['gt_occ'].dtype}")

    # Meta tensors
    metas = sample["img_metas"]
    for k in ["gt_depths", "stereo_depth"]:
        if k in metas and isinstance(metas[k], torch.Tensor):
            _assert(metas[k].dtype == torch.float32,
                    f"img_metas[{k}] dtype = {metas[k].dtype}")

    _ok("All dtypes consistent")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unit test the data pipeline")
    parser.add_argument("--index", type=int, default=0, help="Sample index to test")
    parser.add_argument("--full", action="store_true",
                        help="Run DataLoader batching test (slower)")
    args = parser.parse_args()

    cfg = load_config()
    idx = args.index

    # Stage-by-stage tests
    dataset, info = test_dataset_loading(cfg, idx)
    result = test_load_images(cfg, info)
    result = test_create_depth(cfg, result)
    result = test_load_annotation(cfg, result)
    test_collect_data(cfg, result)

    # End-to-end
    dataset = test_full_pipeline(cfg, idx)

    # Geometric sanity
    info = dataset.get_data_info(idx)
    test_geometric_sanity(cfg, info)

    # Train vs test
    test_train_vs_test(cfg, idx)

    # Dtype consistency
    test_dtypes(cfg, idx)

    # DataLoader batching (optional, slower)
    if args.full:
        test_dataloader_batching(cfg, dataset)

    _sep("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import re

def load_label_map(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Label map file not found: {p}")
    if p.suffix.lower() in [".yaml", ".yml"]:
        m = yaml.safe_load(p.read_text())
    elif p.suffix.lower() == ".json":
        m = json.loads(p.read_text())
    else:
        raise ValueError("Unsupported label map format (use YAML or JSON)")
    # normalize keys to int
    return {int(k): int(v) for k, v in m.items()}

def to_gray_intensity(rgb_img):
    arr = np.asarray(rgb_img).astype(np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return y.astype(np.float32)

def unproject_depth(depth, fx, fy, cx, cy):
    H, W = depth.shape
    ys, xs = np.indices((H, W))
    z = depth.astype(np.float32)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts

def compose_label_u32(sem_u16, inst_u16=None):
    if inst_u16 is None:
        inst_u16 = np.zeros_like(sem_u16, dtype=np.uint16)
    return (inst_u16.astype(np.uint32) << 16) | (sem_u16.astype(np.uint32) & 0xFFFF)

def apply_label_map(sem_ids, label_map, unknown_id=0):
    sem_ids = sem_ids.astype(np.int32)
    if label_map is None:
        return sem_ids.astype(np.uint16)
    out = np.full_like(sem_ids, unknown_id, dtype=np.uint16)
    unique = np.unique(sem_ids)
    for tid in unique:
        sk = label_map.get(int(tid), unknown_id)
        out[sem_ids == tid] = np.uint16(sk)
    return out

def write_velodyne(bin_path, pts_xyz, intensity):
    pts_xyz = pts_xyz.astype(np.float32)
    intensity = intensity.astype(np.float32).reshape(-1, 1)
    arr = np.concatenate([pts_xyz, intensity], axis=1)
    arr.tofile(bin_path)

def write_labels(label_path, labels_u32):
    labels_u32.astype(np.uint32).tofile(label_path)

def parse_tartanair_poses(pose_file):
    """
    Supports:
    - 7 values per line: tx ty tz qx qy qz qw
    - 12 values per line: 3x4 row-major
    - 16 values per line: 4x4 row-major
    Returns list of 4x4 world_T_cam matrices.
    """
    pose_file = Path(pose_file)
    if not pose_file.exists():
        return None

    def quat_to_rot(qx, qy, qz, qw):
        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        q = q / np.linalg.norm(q)
        x, y, z, w = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        return R

    mats = []
    with pose_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split()]
            n = len(vals)
            if n == 7:
                tx, ty, tz, qx, qy, qz, qw = vals
                R = quat_to_rot(qx, qy, qz, qw)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
            elif n == 12:
                M = np.array(vals, dtype=np.float64).reshape(3, 4)
                T = np.eye(4, dtype=np.float64)
                T[:3, :4] = M
            elif n == 16:
                T = np.array(vals, dtype=np.float64).reshape(4, 4)
            else:
                continue
            mats.append(T)
    if len(mats) == 0:
        return None
    return mats

def kitti_calib_text(lidar_T_cam):
    """
    Minimal KITTI-style calib.txt with identity-ish P0..P3 and R0_rect.
    Tr_velo_to_cam is taken from lidar_T_cam.
    """
    lines = []
    # Dummy projection matrices
    for i in range(4):
        lines.append(f"P{i}: 1 0 0 0  0 1 0 0  0 0 1 0")
    lines.append("R0_rect: 1 0 0  0 1 0  0 0 1")
    Tr = lidar_T_cam[:3, :].reshape(-1)
    lines.append("Tr_velo_to_cam: " + " ".join(f"{v:.6f}" for v in Tr))
    return "\n".join(lines) + "\n"

def numeric_key(p):
    m = re.search(r"(\d+)", p.name)
    return int(m.group(1)) if m else -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/media/minh/6eb5d567-e2e1-41bd-8b47-2b12cd4c247e/tartanair/carwelding/Easy/P001", type=str,
                        help="Path to TartanAir scene root (with depth_left, image_left, seg_left, pose_left.txt, etc.)")
    parser.add_argument("--dst", required=True, type=str,
                        help="Output root for fake SemanticKITTI dataset")
    parser.add_argument("--seq-id", type=str, default="99",
                        help="Sequence id under sequences/<seq-id>")
    parser.add_argument("--cam", type=str, default="left", choices=["left", "right"],
                        help="Camera side to use")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--label-map", type=str, default=None,
                        help="YAML/JSON mapping TartanAir seg IDs -> SemanticKITTI IDs")
    parser.add_argument("--pose-file", type=str, default=None,
                        help="Optional explicit pose file; default = pose_<cam>.txt under src")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Used to synthesize timestamps if none given")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-range", type=float, default=80.0)
    parser.add_argument("--intensity-constant", type=float, default=None,
                        help="If set, use this constant intensity instead of grayscale")
    parser.add_argument("--lidar-from-cam-extrinsic", type=str, default=None,
                        help="Row-major 3x4 or 4x4 lidar_T_cam; default identity")
    args = parser.parse_args()

    src = Path(args.src)
    seq_dir = Path(args.dst) / "sequences" / args.seq_id
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    seq_dir.mkdir(parents=True, exist_ok=True)
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img_dir = src / f"image_{args.cam}"
    depth_dir = src / f"depth_{args.cam}"
    seg_dir = src / f"seg_{args.cam}"

    if not img_dir.exists() or not depth_dir.exists() or not seg_dir.exists():
        raise RuntimeError("Expected image_*, depth_*, seg_* directories under src")

    img_files = sorted(img_dir.glob("*.png"), key=numeric_key)
    depth_files = sorted(depth_dir.glob("*.npy"), key=numeric_key)
    seg_files = sorted(seg_dir.glob("*.npy"), key=numeric_key)

    n = min(len(img_files), len(depth_files), len(seg_files))
    if n == 0:
        raise RuntimeError("No frames found. Check that image/depth/seg npy/png exist with matching indices.")

    img_files = img_files[:n]
    depth_files = depth_files[:n]
    seg_files = seg_files[:n]

    # intrinsics
    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    # lidar from cam extrinsic
    if args.lidar_from_cam_extrinsic is None:
        lidar_T_cam = np.eye(4, dtype=np.float64)
    else:
        vals = [float(v) for v in args.lidar_from_cam_extrinsic.strip().split()]
        if len(vals) == 12:
            M = np.array(vals, dtype=np.float64).reshape(3, 4)
            lidar_T_cam = np.eye(4, dtype=np.float64)
            lidar_T_cam[:3, :4] = M
        elif len(vals) == 16:
            lidar_T_cam = np.array(vals, dtype=np.float64).reshape(4, 4)
        else:
            raise ValueError("lidar-from-cam-extrinsic must have 12 or 16 values")

    cam_T_lidar = np.linalg.inv(lidar_T_cam)

    # label map
    label_map = load_label_map(args.label_map)

    # timestamps
    times = [i / args.fps for i in range(n)]
    with (seq_dir / "times.txt").open("w") as f:
        for t in times:
            f.write(f"{t:.6f}\n")

    # calib
    (seq_dir / "calib.txt").write_text(kitti_calib_text(lidar_T_cam))

    # poses
    if args.pose_file is not None:
        pose_file = Path(args.pose_file)
    else:
        pose_file = src / f"pose_{args.cam}.txt"

    lidar_poses = None
    if pose_file.exists():
        cam_poses = parse_tartanair_poses(pose_file)
        if cam_poses is not None:
            if len(cam_poses) < n:
                # pad with identity
                for _ in range(n - len(cam_poses)):
                    cam_poses.append(np.eye(4, dtype=np.float64))
            lidar_poses = []
            for i in range(n):
                world_T_cam = cam_poses[i]
                world_T_lidar = world_T_cam @ cam_T_lidar
                lidar_poses.append(world_T_lidar)

            with (seq_dir / "poses.txt").open("w") as f:
                for T in lidar_poses:
                    M = T[:3, :4].reshape(-1)
                    f.write(" ".join(f"{v:.6f}" for v in M) + "\n")

    # process frames
    for idx in range(n):
        img_path = img_files[idx]
        depth_path = depth_files[idx]
        seg_path = seg_files[idx]

        depth = np.load(depth_path).astype(np.float32)  # assume meters
        seg = np.load(seg_path)                         # integer IDs

        if depth.shape != seg.shape:
            raise RuntimeError(f"Shape mismatch depth vs seg at index {idx}: {depth.shape} vs {seg.shape}")

        # validity mask
        valid = np.isfinite(depth)
        valid &= depth > args.min_depth
        if np.isfinite(args.max_range):
            valid &= depth <= args.max_range

        pts_cam = unproject_depth(depth, fx, fy, cx, cy)
        pts_cam = pts_cam[valid.reshape(-1)]

        # transform into lidar frame
        R = lidar_T_cam[:3, :3]
        t = lidar_T_cam[:3, 3]
        pts_lidar = (R @ pts_cam.T).T + t

        # intensity
        if args.intensity_constant is not None:
            intensity = np.full((pts_lidar.shape[0],), float(args.intensity_constant), dtype=np.float32)
        else:
            rgb = Image.open(img_path).convert("RGB")
            gray = to_gray_intensity(rgb)
            intensity = gray.reshape(-1)[valid.reshape(-1)]

        # labels
        sem_flat = seg.reshape(-1)[valid.reshape(-1)]
        sem_mapped = apply_label_map(sem_flat, label_map, unknown_id=0)
        lbl_u32 = compose_label_u32(sem_mapped, None)

        out_idx = f"{idx:06d}"
        write_velodyne(velodyne_dir / f"{out_idx}.bin", pts_lidar, intensity)
        write_labels(labels_dir / f"{out_idx}.label", lbl_u32)

    print(f"Done. Wrote {n} frames to {seq_dir}")

if __name__ == "__main__":
    main()

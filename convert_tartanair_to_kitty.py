#!/usr/bin/env python3
"""
convert_tartanair_to_semantickitti.py

Turn a TartanAir RGB-D + semantic sequence into a SemanticKITTI-like dataset.

What it does:
- Reads color, depth, and semantic segmentation frames from a TartanAir trajectory.
- Back-projects depth into a 3D point cloud using camera intrinsics (pinhole model).
- Optionally filters points by max range and depth validity.
- Writes KITTI/SemanticKITTI-style files:
    sequences/<seq_id>/velodyne/<idx>.bin            (float32 x,y,z,intensity)
    sequences/<seq_id>/labels/<idx>.label            (uint32, upper16=instance, lower16=semantic)
    sequences/<seq_id>/times.txt                     (sec since start; one per frame)
    sequences/<seq_id>/poses.txt                     (T_w_lidar 4x4 row-major per line; optional if --poses provided)
    sequences/<seq_id>/calib.txt                     (KITTI calib file; R|T from lidar to cam if provided/identity)

Assumptions / Notes:
- TartanAir doesn't ship LiDAR; we synthesize a "LiDAR-like" cloud by unprojecting depth.
- Intensity is derived from grayscale luminance unless --intensity-constant is set.
- Instance IDs: not provided by TartanAir; set to 0. You can inject later if you have panoptic instances.
- Semantic label mapping must be provided via a YAML or JSON mapping of {tartanair_id: semkitti_id}. See --label-map.
- Intrinsics must be known. If not supplied, script tries to read common TartanAir intrinsics files; else require CLI.
- Poses are optional. If available, provide a per-frame 4x4 or 3x4 camera pose (world_T_cam). We'll convert to world_T_lidar
  using the extrinsics (lidar_T_cam). If no poses, we still generate per-frame clouds with identity pose sequence.

Usage example:
python convert_tartanair_to_semantickitti.py \
  --src /path/to/TartanAir/abandonedfactory/Easy/P000/ \
  --dst /out/semkitti/ --seq-id 00 \
  --cam left \
  --color-pattern "image_{cam}/*_{cam}.png" \
  --depth-pattern "depth_{cam}/*_{cam}.npy" \
  --seg-pattern   "seg_{cam}/*_{cam}.png" \
  --fx 320.0 --fy 320.0 --cx 320.0 --cy 240.0 \
  --label-map /path/to/tartanair_to_semkitti_labels.yaml \
  --lidar-from-cam-extrinsic "1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1" \
  --max-range 80.0

Output matches SemanticKITTI file layout but is "generated LiDAR" not rotating 64-beam scans.
"""

import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import re
import sys

def load_label_map(p):
    if p is None:
        return None
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"Label map file not found: {p}")
    if p.suffix.lower() in [".yaml", ".yml"]:
        return yaml.safe_load(p.read_text())
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    raise ValueError("Unsupported label map format. Use YAML or JSON.")

def parse_intrinsics(args, src):
    # Try explicit CLI
    if all(v is not None for v in [args.fx, args.fy, args.cx, args.cy]):
        K = np.array([[args.fx, 0, args.cx],[0, args.fy, args.cy],[0,0,1]], dtype=np.float64)
        return K
    # Try common TartanAir intrinsics files
    candidates = [
        src / "intrinsics.txt",
        src / "K.txt",
        src / "camera_intrinsics.txt",
        src / "CamInfo_left.txt" if args.cam=="left" else src / "CamInfo_right.txt"
    ]
    for c in candidates:
        if c.exists():
            try:
                data = np.loadtxt(c, dtype=np.float64)
                if data.size == 9:
                    K = data.reshape(3,3)
                    return K
                # Try fx, fy, cx, cy in a line
                if data.size >= 4:
                    fx, fy, cx, cy = data.ravel()[:4]
                    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
            except Exception:
                pass
    raise ValueError("Camera intrinsics not found. Provide --fx --fy --cx --cy or an intrinsics file.")

def parse_extrinsic(extr_str):
    if extr_str is None:
        return np.eye(4, dtype=np.float64) # lidar_T_cam = I
    vals = [float(v) for v in extr_str.strip().split()]
    if len(vals) == 12:
        M = np.array(vals, dtype=np.float64).reshape(3,4)
        T = np.eye(4, dtype=np.float64); T[:3,:4] = M
        return T
    if len(vals) == 16:
        T = np.array(vals, dtype=np.float64).reshape(4,4)
        return T
    raise ValueError("Extrinsic must be 3x4 (12 numbers) or 4x4 (16 numbers) row-major.")

def globs_from_pattern(src, pattern, cam_token):
    pat = pattern.replace("{cam}", cam_token)
    files = sorted((src).glob(pat))
    return files

def to_gray_intensity(rgb):
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    # simple luminance
    y = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    return y

def unproject_depth(depth, K):
    H, W = depth.shape
    ys, xs = np.indices((H, W))
    z = depth.astype(np.float32)
    x = (xs - K[0,2]) * z / K[0,0]
    y = (ys - K[1,2]) * z / K[1,1]
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts

def load_depth(path):
    p = Path(path)
    if p.suffix.lower() in [".png", ".jpg"]:
        # assume depth in meters encoded as 16-bit PNG or similar; try 16-bit first
        im = Image.open(p)
        arr = np.array(im)
        if arr.dtype == np.uint16:
            # common: depth_mm
            depth = arr.astype(np.float32) / 1000.0
        else:
            # fallback assume already in meters 8-bit (unlikely)
            depth = arr.astype(np.float32)
        return depth
    if p.suffix.lower() == ".npy":
        return np.load(p).astype(np.float32)
    if p.suffix.lower() == ".exr":
        try:
            import openexr  # optional
            raise NotImplementedError("EXR reading not implemented in this script.")
        except Exception:
            raise ValueError("Install OpenEXR or convert EXR to PNG/NPY.")
    raise ValueError(f"Unsupported depth format: {p.suffix}")

def load_semantic(path):
    # Expect per-pixel class id image (uint8/uint16). No palette decoding.
    im = Image.open(path)
    return np.array(im)

def apply_label_map(sem, label_map, unknown_id=0):
    if label_map is None:
        return sem.astype(np.uint16)
    # label_map: dict of tartanair_id -> semkitti_id
    out = np.full_like(sem, fill_value=unknown_id, dtype=np.uint16)
    # fast vectorized remap for limited IDs
    unique = np.unique(sem)
    for tid in unique:
        sk = label_map.get(int(tid), unknown_id)
        out[sem == tid] = np.uint16(sk)
    return out

def compose_label_uint32(semantic_u16, instance_u16=None):
    if instance_u16 is None:
        instance_u16 = np.zeros_like(semantic_u16, dtype=np.uint16)
    return (instance_u16.astype(np.uint32) << 16) | (semantic_u16.astype(np.uint32) & 0xFFFF)

def write_bin(bin_path, points, intensity):
    assert points.shape[0] == intensity.shape[0]
    arr = np.concatenate([points.astype(np.float32), intensity.reshape(-1,1).astype(np.float32)], axis=1)
    arr.astype(np.float32).tofile(bin_path)

def write_label(label_path, labels_u32):
    labels_u32.astype(np.uint32).tofile(label_path)

def load_pose_file(pose_path):
    # Flexible: lines with 12 or 16 numbers, or 4x4 per line flattened. Return list of 4x4.
    mats = []
    with open(pose_path, "r") as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            if len(vals) == 12:
                M = np.array(vals, dtype=np.float64).reshape(3,4)
                T = np.eye(4); T[:3,:4] = M
            elif len(vals) == 16:
                T = np.array(vals, dtype=np.float64).reshape(4,4)
            else:
                continue
            mats.append(T)
    return mats

def kitti_calib_text(lidar_T_cam):
    # KITTI expects several entries. We'll provide a minimal calib with Tr and P0..P3 stubs.
    # Tr: velocitiy (lidar) to camera_0 (KITTI: Tr_velo_to_cam)
    Tr = lidar_T_cam[:3,:].reshape(-1)
    # Default pinhole projection unknown; write identity-ish P0..P3 to satisfy loaders.
    lines = []
    for i in range(4):
        lines.append(f"P{i}: 1 0 0 0  0 1 0 0  0 0 1 0")
    lines.append("R0_rect: 1 0 0  0 1 0  0 0 1")
    lines.append("Tr_velo_to_cam: " + " ".join([f"{v:.6f}" for v in Tr]))
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="TartanAir sequence root (folder containing image/depth/seg).")
    ap.add_argument("--dst", type=str, required=True, help="Output root for SemanticKITTI-like dataset.")
    ap.add_argument("--seq-id", type=str, default="00", help="Sequence id to write under sequences/<seq-id>.")
    ap.add_argument("--cam", type=str, default="left", choices=["left","right"], help="Camera side token for patterns.")
    ap.add_argument("--color-pattern", type=str, default="image_{cam}/*_{cam}.png", help="Glob relative to --src for color.")
    ap.add_argument("--depth-pattern", type=str, default="depth_{cam}/*_{cam}.npy", help="Glob relative to --src for depth.")
    ap.add_argument("--seg-pattern",   type=str, default="seg_{cam}/*_{cam}.png",   help="Glob relative to --src for semantic IDs.")
    ap.add_argument("--pose-file", type=str, default=None, help="Optional pose file (world_T_cam per frame).")
    ap.add_argument("--fx", type=float, default=None); ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None); ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--label-map", type=str, default=None, help="YAML/JSON: tartanair_id -> semkitti_id.")
    ap.add_argument("--lidar-from-cam-extrinsic", type=str, default=None, help="Row-major 3x4 or 4x4 lidar_T_cam.")
    ap.add_argument("--max-range", type=float, default=120.0)
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--intensity-constant", type=float, default=None, help="If set, use this constant intensity for all points.")
    ap.add_argument("--stride", type=int, default=1, help="Sample every Nth pixel to limit point count (>=1).")
    ap.add_argument("--downsample", type=int, default=1, help="Keep 1 every N points after unprojection (>=1).")
    ap.add_argument("--timestamps", type=str, default=None, help="Optional timestamps file; else compute 0, dt, ...")
    ap.add_argument("--fps", type=float, default=10.0, help="Used if no timestamps provided.")
    args = ap.parse_args()
    
    src = Path(args.src)
    dst = Path(args.dst) / "sequences" / args.seq_id
    velodyne_dir = dst / "velodyne"; labels_dir = dst / "labels"
    dst.mkdir(parents=True, exist_ok=True); velodyne_dir.mkdir(parents=True, exist_ok=True); labels_dir.mkdir(parents=True, exist_ok=True)

    K = parse_intrinsics(args, src)
    lidar_T_cam = parse_extrinsic(args.lidar_from_cam_extrinsic)
    cam_T_lidar = np.linalg.inv(lidar_T_cam)

    color_files = globs_from_pattern(src, args.color_pattern, args.cam)
    depth_files = globs_from_pattern(src, args.depth_pattern, args.cam)
    seg_files   = globs_from_pattern(src, args.seg_pattern,   args.cam)

    n = min(len(color_files), len(depth_files), len(seg_files))
    if n == 0:
        print("No frames found. Check your --*pattern arguments.", file=sys.stderr)
        sys.exit(2)
    if n == 0:
        print("No frames found. Check your --*pattern arguments.", file=sys.stderr)
    # 
    # Enforce aligned filenames by sorting and matching numeric indices if possible
    def index_key(p):
        m = re.search(r"(\d+)", p.name)
        return int(m.group(1)) if m else -1
    color_files.sort(key=index_key); depth_files.sort(key=index_key); seg_files.sort(key=index_key)
    
    label_map = load_label_map(args.label_map)

    # timestamps
    if args.timestamps:
        ts = [float(x.strip()) for x in Path(args.timestamps).read_text().splitlines() if x.strip()]
        if len(ts) < n:
            ts = ts + [ts[-1] + (1.0/args.fps)*(i+1) for i in range(n-len(ts))]
    else:
        ts = [i / args.fps for i in range(n)]
    times_txt = dst / "times.txt"
    times_txt.write_text("\n".join([f"{t:.6f}" for t in ts]) + "\n")

    # calib
    (dst / "calib.txt").write_text(kitti_calib_text(lidar_T_cam))

    # poses
    poses_path = None
    if args.pose_file:
        cam_poses = load_pose_file(args.pose_file)  # world_T_cam
        if len(cam_poses) < n:
            print("Warning: pose_file has fewer poses than frames; remaining set to identity.", file=sys.stderr)
            for _ in range(n - len(cam_poses)):
                cam_poses.append(np.eye(4))
        lidar_poses = [ T @ cam_T_lidar for T in cam_poses ]  # world_T_lidar = world_T_cam * cam_T_lidar
        poses_path = dst / "poses.txt"
        with open(poses_path, "w") as f:
            for T in lidar_poses[:n]:
                f.write(" ".join([f"{v:.6f}" for v in T.reshape(-1)]) + "\n")

    # Process frames
    for i in range(0, n):
        color_path = color_files[i]
        depth_path = depth_files[i]
        seg_path   = seg_files[i]

        depth = load_depth(depth_path)
        H, W = depth.shape

        # stride sampling to control density
        depth = depth[::args.stride, ::args.stride]

        # mask validity
        valid = (depth > args.min_depth) & np.isfinite(depth)
        if args.max_range is not None and np.isfinite(args.max_range):
            valid &= (depth <= args.max_range)

        pts_cam = unproject_depth(depth, np.array([[K[0,0],0,K[0,2]],[0,K[1,1],K[1,2]],[0,0,1]], dtype=np.float64))
        pts_cam = pts_cam[valid.reshape(-1)]

        # transform to lidar frame
        # cam_T_world? No, per-frame clouds are in camera frame; for velodyne clouds, convention is lidar frame per frame.
        # We output in lidar frame by applying lidar_T_cam to points in cam frame.
        R = lidar_T_cam[:3,:3]; t = lidar_T_cam[:3,3]
        pts_lidar = (R @ pts_cam.T).T + t

        # intensity
        if args.intensity_constant is not None:
            intens = np.full((pts_lidar.shape[0],), float(args.intensity_constant), dtype=np.float32)
        else:
            rgb = Image.open(color_path).convert("RGB")
            gray = to_gray_intensity(rgb)[::args.stride, ::args.stride]
            intens = gray.reshape(-1)[valid.reshape(-1)].astype(np.float32)

        # labels
        sem = load_semantic(seg_path)[::args.stride, ::args.stride]
        sem = sem.reshape(-1)[valid.reshape(-1)]
        sem_mapped = apply_label_map(sem, label_map, unknown_id=0)
        lbl_u32 = compose_label_uint32(sem_mapped, None)

        # optional downsample AFTER validity masking to keep sync
        if args.downsample > 1:
            idx = np.arange(pts_lidar.shape[0])[::args.downsample]
            pts_lidar = pts_lidar[idx]
            intens    = intens[idx]
            lbl_u32   = lbl_u32[idx]

        out_idx = f"{i:06d}"
        write_bin(velodyne_dir / f"{out_idx}.bin", pts_lidar, intens)
        write_label(labels_dir / f"{out_idx}.label", lbl_u32)

    print(f"Done. Wrote {n} frames to {dst}")

if __name__ == "__main__":
    main()
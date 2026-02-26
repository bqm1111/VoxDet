#!/usr/bin/env python3
"""
Generate voxel ground truth with FIXED output shape for SemanticKITTI model compatibility.

OUTPUT COORDINATE FRAME: LiDAR (SemanticKITTI convention)
- X: forward (index 0 of voxel grid)
- Y: left    (index 1 of voxel grid)  
- Z: up      (index 2 of voxel grid)

Transformation from TartanAir Camera frame:
- Camera: X-right, Y-down, Z-forward
- LiDAR:  X-forward, Y-left, Z-up
- X_lidar = Z_cam, Y_lidar = -X_cam, Z_lidar = -Y_cam

Key features:
- Always outputs fixed shape (256, 256, 32) = (X, Y, Z) in LiDAR frame
- Uses fixed pc_range [0, -25.6, -2, 51.2, 25.6, 4.4] (SemanticKITTI standard)
- Points outside the fixed range are ignored (not causing shape changes)

SemanticKITTI model expects: (batch, 256, 256, 32) = (B, X, Y, Z)
"""

import os
import sys
import argparse
import glob
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
# Try to import open3d, but make it optional
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not available, visualization will be limited")

# =============================================================================
# FIXED Configuration for SemanticKITTI Model Compatibility
# =============================================================================

# Target output shape: (256, 256, 32) = (X, Y, Z) - SemanticKITTI standard
FIXED_OCC_SIZE = (256, 256, 32)

# Voxel size in meters
VOXEL_SIZE = 0.2

# SemanticKITTI standard pc_range
# X: 256 * 0.2 = 51.2m (forward in LiDAR frame)
# Y: 256 * 0.2 = 51.2m (left-right, centered at 0)
# Z: 32 * 0.2 = 6.4m (height)
#
# For LiDAR frame (X-forward, Y-left, Z-up):
# pc_range = [x_min, y_min, z_min, x_max, y_max, z_max]
FIXED_PC_RANGE = (
    0.0,    # x_min: start at camera/sensor position
    -25.6,  # y_min: 25.6m to the right
    -2.0,   # z_min: 2m below sensor
    51.2,   # x_max: 51.2m forward
    25.6,   # y_max: 25.6m to the left
    4.4,    # z_max: 4.4m above ground (6.4m total height)
)
# Verification: occ_size * voxel_size = range
# X: 256 * 0.2 = 51.2 ✓ (51.2 - 0.0)
# Y: 256 * 0.2 = 51.2 ✓ (25.6 - (-25.6))
# Z: 32 * 0.2 = 6.4 ✓ (4.4 - (-2.0))

# Camera intrinsics (TartanAir default)
FOCAL_LENGTH = 320.0
PRINCIPAL_POINT = (320.0, 320.0)

# Transformation: Camera (X-right, Y-down, Z-forward) -> LiDAR (X-forward, Y-left, Z-up)
CAM_TO_LIDAR = np.array([
    [0,  0,  1],   # X_lidar = Z_cam
    [-1, 0,  0],   # Y_lidar = -X_cam  
    [0, -1,  0],   # Z_lidar = -Y_cam
], dtype=np.float64)

# Label mapping (TartanAir -> SemanticKITTI training classes)
LEARNING_MAP = {
    0: 0, 6: 1, 7: 2, 8: 3, 9: 4, 23: 5, 27: 6, 28: 7, 36: 8, 59: 9,
    60: 10, 64: 11, 65: 12, 69: 13, 70: 14, 72: 15, 116: 16, 123: 17,
    132: 18, 143: 19, 146: 20, 157: 21, 160: 22, 161: 23, 171: 24,
    175: 25, 180: 26, 188: 27, 191: 28, 195: 29, 199: 30, 205: 31,
    208: 32, 239: 33,
}


# =============================================================================
# Core Functions
# =============================================================================

def read_depth(depth_path):
    """Read TartanAir depth image (stored as RGBA with float32)."""
    depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_rgba is None:
        raise ValueError(f"Could not load depth: {depth_path}")
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)


def read_seg(seg_path):
    """Read segmentation mask."""
    seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        raise ValueError(f"Could not load segmentation: {seg_path}")
    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    return seg.astype(np.uint8)


def map_labels(labels):
    """Map original TartanAir labels to training class IDs."""
    mapped = np.zeros_like(labels)
    for orig_id, learn_id in LEARNING_MAP.items():
        mapped[labels == orig_id] = learn_id
    return mapped


def depth_to_pointcloud(depth, fx=FOCAL_LENGTH, fy=FOCAL_LENGTH, 
                        cx=PRINCIPAL_POINT[0], cy=PRINCIPAL_POINT[1]):
    """Convert depth image to 3D point cloud in camera frame."""
    height, width = depth.shape
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.stack([x, y, z], axis=-1)


def transform_cam_to_lidar(points):
    """Transform points from camera frame to LiDAR frame."""
    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    points_lidar = (CAM_TO_LIDAR @ points_flat.T).T
    return points_lidar.reshape(original_shape)


def generate_fixed_voxel(
    depth_path,
    seg_path,
    output_path=None,
    occ_size=FIXED_OCC_SIZE,
    pc_range=FIXED_PC_RANGE,
    min_depth=0.5,
    max_depth=80.0,
    apply_mapping=True,
    verbose=False,
):
    """
    Generate voxel ground truth with FIXED output shape.
    
    Args:
        depth_path: Path to depth image
        seg_path: Path to segmentation image
        output_path: Where to save .npy file (optional)
        occ_size: Fixed output shape (X, Y, Z) - default (256, 160, 256)
        pc_range: Fixed point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        apply_mapping: Apply label mapping
        verbose: Print debug info
        
    Returns:
        voxel_grid: np.ndarray of shape occ_size with dtype uint8
    """
    # Read inputs
    depth = read_depth(depth_path)
    seg = read_seg(seg_path)
    
    # Convert to point cloud (camera frame)
    points_cam = depth_to_pointcloud(depth).reshape(-1, 3)
    labels = seg.reshape(-1)
    depth_flat = depth.reshape(-1)
    
    # Apply label mapping
    if apply_mapping:
        labels = map_labels(labels)
    
    # Filter valid depth
    valid_mask = (depth_flat > min_depth) & (depth_flat < max_depth)
    points_cam = points_cam[valid_mask]
    labels = labels[valid_mask]
    
    if verbose:
        print(f"Valid points after depth filter: {len(points_cam):,}")
    
    if len(points_cam) == 0:
        if verbose:
            print("Warning: No valid points, returning empty voxel grid")
        voxel_grid = np.zeros(occ_size, dtype=np.uint8)
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            np.save(output_path, voxel_grid)
        return voxel_grid
    
    # Transform to LiDAR frame (X-forward, Y-left, Z-up)
    points = transform_cam_to_lidar(points_cam)
    
    if verbose:
        print(f"Coordinate frame: LiDAR (X-forward, Y-left, Z-up)")
        print(f"Point cloud bounds (LiDAR frame):")
        print(f"  X (forward): [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Y (left):    [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z (up):      [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        print(f"Fixed pc_range: {pc_range}")
        print(f"  X: [{pc_range[0]}, {pc_range[3]}]")
        print(f"  Y: [{pc_range[1]}, {pc_range[4]}]")
        print(f"  Z: [{pc_range[2]}, {pc_range[5]}]")
        print(f"Fixed occ_size: {occ_size} (X, Y, Z)")
    
    # Convert to numpy arrays
    occ_size = np.array(occ_size)
    pc_range = np.array(pc_range)
    
    # Filter points within fixed pc_range
    in_range = (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
        (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
        (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
    )
    
    points_in = points[in_range]
    labels_in = labels[in_range]
    
    if verbose:
        pct = 100.0 * len(points_in) / max(1, len(points))
        print(f"Points in fixed range: {len(points_in):,} / {len(points):,} ({pct:.1f}%)")
    
    # Create voxel grid with FIXED size
    voxel_grid = np.zeros(occ_size, dtype=np.uint8)
    
    if len(points_in) == 0:
        if verbose:
            print("Warning: No points in range, returning empty voxel grid")
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            np.save(output_path, voxel_grid)
        return voxel_grid
    
    # Compute voxel indices
    voxel_size = (pc_range[3:6] - pc_range[0:3]) / occ_size
    voxel_indices = ((points_in - pc_range[0:3]) / voxel_size).astype(np.int32)
    voxel_indices = np.clip(voxel_indices, 0, occ_size - 1)
    
    # Majority voting for overlapping points
    voxel_labels = {}
    for idx, label in zip(voxel_indices, labels_in):
        key = (idx[0], idx[1], idx[2])
        if key not in voxel_labels:
            voxel_labels[key] = []
        voxel_labels[key].append(label)
    
    for key, label_list in voxel_labels.items():
        label_array = np.array(label_list)
        non_zero = label_array[label_array > 0]
        if len(non_zero) > 0:
            unique, counts = np.unique(non_zero, return_counts=True)
            voxel_grid[key] = unique[np.argmax(counts)]
    
    if verbose:
        print(f"Occupied voxels: {np.sum(voxel_grid > 0):,}")
        print(f"Output shape: {voxel_grid.shape} (X, Y, Z) in LiDAR frame")
    
    # Verify output shape matches expected
    expected_shape = tuple(occ_size) if isinstance(occ_size, np.ndarray) else occ_size
    assert voxel_grid.shape == expected_shape, \
        f"Shape mismatch: got {voxel_grid.shape}, expected {expected_shape}"
    
    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.save(output_path, voxel_grid)
        if verbose:
            print(f"Saved: {output_path}")
    
    return voxel_grid


def _worker(args):
    """Worker function for multiprocessing."""
    (depth_path, seg_path, output_path, occ_size, pc_range, 
     min_depth, max_depth, apply_mapping) = args
    
    try:
        generate_fixed_voxel(
            depth_path, seg_path, output_path,
            occ_size=occ_size,
            pc_range=pc_range,
            min_depth=min_depth,
            max_depth=max_depth,
            apply_mapping=apply_mapping,
            verbose=False
        )
        return True, depth_path, None
    except Exception as e:
        import traceback
        return False, depth_path, str(e) + "\n" + traceback.format_exc()


def process_dataset(
    data_root,
    output_dir=None,
    occ_size=FIXED_OCC_SIZE,
    pc_range=FIXED_PC_RANGE,
    camera="lcam_front",
    num_workers=1,
    min_depth=0.5,
    max_depth=80.0,
    apply_mapping=True,
):
    """
    Process entire dataset to generate fixed-size voxel labels.
    
    Args:
        data_root: Root directory of TartanAir dataset
        output_dir: Output directory (optional, defaults to inside data_root)
        occ_size: Fixed voxel grid size (default: 256, 160, 256)
        pc_range: Fixed pc_range for voxelization
        camera: Camera name
        num_workers: Parallel workers
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        apply_mapping: Apply label mapping
    """
    # Find depth files
    depth_pattern = os.path.join(data_root, "P*", f"depth_{camera}", "*.png")
    depth_files = sorted(glob.glob(depth_pattern))
    
    if len(depth_files) == 0:
        # Try alternative pattern
        depth_files = sorted(glob.glob(os.path.join(data_root, f"*_{camera}_depth.png")))
    
    print(f"Found {len(depth_files)} depth files")
    print(f"Fixed output shape: {occ_size}")
    print(f"Fixed pc_range: {pc_range}")
    
    # Build tasks
    tasks = []
    for depth_path in depth_files:
        depth_dir = os.path.dirname(depth_path)
        traj_dir = os.path.dirname(depth_dir)
        frame_id = os.path.basename(depth_path).replace(".png", "").split("_")[0]
        seg_path = os.path.join(traj_dir, f"seg_{camera}", f"{frame_id}_{camera}_seg.png")
        
        if not os.path.exists(seg_path):
            continue
        
        if output_dir:
            out_subdir = os.path.basename(traj_dir)
            output_path = os.path.join(output_dir, out_subdir, f"voxel_label_{camera}", f"{frame_id}_voxel_label.npy")
        else:
            output_path = os.path.join(traj_dir, f"voxel_label_{camera}", f"{frame_id}_voxel_label.npy")
        
        tasks.append((
            depth_path, seg_path, output_path,
            tuple(occ_size), tuple(pc_range),
            min_depth, max_depth, apply_mapping
        ))
    
    print(f"Processing {len(tasks)} frames...")
    
    successful = 0
    failed = 0
    errors = []
    
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_worker, task): task[0] for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures)):
                success, path, error = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                    errors.append((path, error))
    else:
        for task in tqdm(tasks):
            success, path, error = _worker(task)
            if success:
                successful += 1
            else:
                failed += 1
                errors.append((path, error))
    
    print(f"\nCompleted: {successful} successful, {failed} failed")
    
    if errors and len(errors) <= 5:
        print("\nErrors:")
        for path, error in errors:
            print(f"  {path}: {error}")
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description='Generate FIXED-SIZE voxel labels for SemanticKITTI model compatibility'
    )
    parser.add_argument('--depth_file', default="data/tartanair/CarWelding/Data_easy/P000/depth_lcam_front/000000_lcam_front_depth.png", type=str, help='Single depth file')
    parser.add_argument('--seg_file', default ="data/tartanair/CarWelding/Data_easy/P000/seg_lcam_front/000000_lcam_front_seg.png", type=str, help='Single segmentation file')
    parser.add_argument('--data_root', type=str, help='Dataset root for batch')
    parser.add_argument('--output', type=str, help='Output file/directory')
    parser.add_argument('--camera', type=str, default='lcam_front')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--min_depth', type=float, default=0.5)
    parser.add_argument('--max_depth', type=float, default=80.0)
    parser.add_argument('--verbose', '-v', action='store_true')
    
    # Allow custom occ_size (defaults match SemanticKITTI: 256, 256, 32)
    parser.add_argument('--occ_x', type=int, default=256)
    parser.add_argument('--occ_y', type=int, default=256)
    parser.add_argument('--occ_z', type=int, default=32)
    
    args = parser.parse_args()
    
    occ_size = (args.occ_x, args.occ_y, args.occ_z)
    
    # Recompute pc_range based on occ_size if custom size provided
    if occ_size != FIXED_OCC_SIZE:
        # Compute pc_range to maintain 0.2m voxel size
        # Following SemanticKITTI convention: Y centered, Z starts at -2
        pc_range = (
            0.0,                        # x_min
            -args.occ_y * 0.2 / 2,      # y_min (centered)
            -2.0,                       # z_min (fixed)
            args.occ_x * 0.2,           # x_max
            args.occ_y * 0.2 / 2,       # y_max (centered)
            -2.0 + args.occ_z * 0.2,    # z_max
        )
        print(f"Custom occ_size {occ_size}, computed pc_range: {pc_range}")
    else:
        pc_range = FIXED_PC_RANGE

    if args.depth_file and args.seg_file:
        # Single file
        voxel = generate_fixed_voxel(
            args.depth_file, args.seg_file,
            output_path=args.output,
            occ_size=occ_size,
            pc_range=pc_range,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            verbose=args.verbose or True,
        )
        print(f"Generated voxel shape: {voxel.shape}")
        print(f"Occupied voxels: {np.sum(voxel > 0):,}")

    elif args.data_root:
        # Batch processing
        process_dataset(
            args.data_root,
            output_dir=args.output,
            occ_size=occ_size,
            pc_range=pc_range,
            camera=args.camera,
            num_workers=args.num_workers,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
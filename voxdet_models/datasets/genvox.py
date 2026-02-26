import os
import sys
import argparse
import glob
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from scipy.spatial.transform import Rotation

"""
Coordinate Systems:
-------------------
Camera Frame (OpenCV convention):
- X: right
- Y: down
- Z: forward (depth)

The pc_range is automatically computed from the point cloud to ensure
all points are captured, regardless of the scene.
"""

FOCAL_LENGTH = 320.0
PRINCIPAL_POINT = (320.0, 320.0)
IMAGE_SIZE = (640, 640)
DEFAULT_OCC_SIZE = (256, 32, 256)

# Default pc_range - used only as fallback or for consistent grid sizing
# For actual voxelization, use auto_range=True or compute_pc_range_from_points()
DEFAULT_PC_RANGE = (-25.6, -25.6, 0.0, 25.6, 6.4, 51.2)

LEARNING_MAP = {
    0: 0, 6: 1, 7: 2, 8: 3, 9: 4, 23: 5, 27: 6, 28: 7, 36: 8, 59: 9,
    60: 10, 64: 11, 65: 12, 69: 13, 70: 14, 72: 15, 116: 16, 123: 17,
    132: 18, 143: 19, 146: 20, 157: 21, 160: 22, 161: 23, 171: 24,
    175: 25, 180: 26, 188: 27, 191: 28, 195: 29, 199: 30, 205: 31,
    208: 32, 239: 33,
}

CLASS_NAMES = {
    "cabinet": 6, "cable": 191, "car": 72, "ceiling": 64, "cementcolumn": 59,
    "chair": 36, "chasis": 195, "cieling": 27, "door": 180, "floor": 199,
    "keyboard": 239, "lamp": 146, "light": 23, "metalcieling": 160,
    "metalfloor": 116, "metalhandrail": 9, "metalpanel": 8, "metalplatform": 28,
    "metalpole": 65, "metalramp": 70, "metalstair": 143, "monitor": 171,
    "pipecover": 7, "platform": 161, "plug": 60, "robotarm": 208, "sky": 188,
    "table": 205, "tireassembly": 157, "toolbox": 123, "ventpipe": 69,
    "ventpipeclamp": 132, "wall": 175,
}

COLOR_MAP = [
    [0, 0, 0], [153, 108, 6], [112, 105, 191], [89, 121, 72], [190, 225, 64],
    [206, 190, 59], [81, 13, 36], [115, 176, 195], [161, 171, 27], [135, 169, 180],
    [29, 26, 199], [102, 16, 239], [242, 107, 146], [156, 198, 23], [49, 89, 160],
    [68, 218, 116], [11, 236, 9], [196, 30, 8], [121, 67, 28], [0, 53, 65],
    [146, 52, 70], [226, 149, 143], [151, 126, 171], [194, 39, 7], [205, 120, 161],
    [212, 51, 60], [211, 80, 208], [189, 135, 188], [54, 72, 205], [103, 252, 157],
    [124, 21, 123], [19, 132, 69], [195, 237, 132], [94, 253, 175],
]

CAM_TO_NED = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
NED_TO_CAM = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)


# =============================================================================
# Utility Functions
# =============================================================================

def map_labels(labels):
    """Map original TartanAir labels to training class IDs."""
    mapped = np.zeros_like(labels)
    for orig_id, learn_id in LEARNING_MAP.items():
        mapped[labels == orig_id] = learn_id
    return mapped


def read_depth(depth_path):
    """Read TartanAir depth image."""
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


def depth_to_pointcloud(
    depth,
    fx=FOCAL_LENGTH,
    fy=FOCAL_LENGTH,
    cx=PRINCIPAL_POINT[0],
    cy=PRINCIPAL_POINT[1],
):
    """Convert depth image to 3D point cloud in camera frame."""
    height, width = depth.shape

    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)

    return points


# =============================================================================
# PC Range Computation
# =============================================================================

def compute_pc_range_from_points(points, padding=2.0, align_to_grid=True, 
                                  target_voxel_size=0.2):
    """
    Compute pc_range that encompasses all points.
    
    Args:
        points: (N, 3) point cloud
        padding: Extra padding around bounds
        align_to_grid: If True, align range to nice voxel boundaries
        target_voxel_size: Target voxel size for alignment
        
    Returns:
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    if len(points) == 0:
        return DEFAULT_PC_RANGE
    
    min_bounds = points.min(axis=0) - padding
    max_bounds = points.max(axis=0) + padding
    
    if align_to_grid:
        # Align to voxel grid for consistent sizing
        min_bounds = np.floor(min_bounds / target_voxel_size) * target_voxel_size
        max_bounds = np.ceil(max_bounds / target_voxel_size) * target_voxel_size
    
    return tuple(min_bounds.tolist() + max_bounds.tolist())


def compute_pc_range_from_depth(depth, min_depth=0.1, max_depth=100.0, 
                                 padding=2.0):
    """
    Compute pc_range from a depth image.
    
    Args:
        depth: Depth image (H, W)
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        padding: Extra padding
        
    Returns:
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    points = depth_to_pointcloud(depth).reshape(-1, 3)
    depth_flat = depth.reshape(-1)
    
    valid = (depth_flat > min_depth) & (depth_flat < max_depth)
    points_valid = points[valid]
    
    return compute_pc_range_from_points(points_valid, padding)


def compute_occ_size_from_range(pc_range, voxel_size=0.2):
    """
    Compute voxel grid dimensions from pc_range and voxel_size.
    
    Args:
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: Size of each voxel in meters
        
    Returns:
        occ_size: (X, Y, Z) grid dimensions
    """
    pc_range = np.array(pc_range)
    extents = pc_range[3:6] - pc_range[0:3]
    occ_size = np.ceil(extents / voxel_size).astype(int)
    return tuple(occ_size.tolist())


# =============================================================================
# Voxelization
# =============================================================================

def voxelize_depth_seg(
    depth, 
    seg, 
    occ_size=None,
    pc_range=None, 
    apply_mapping=True,
    auto_range=True,
    min_depth=0.1,
    max_depth=100.0,
    voxel_size=0.2,
    padding=2.0,
    verbose=False
):
    """
    Voxelize depth map with semantic labels.
    
    Args:
        depth: Depth image (H, W)
        seg: Segmentation mask (H, W)
        occ_size: Voxel grid dimensions. If None, computed from pc_range and voxel_size
        pc_range: Point cloud range. If None and auto_range=True, computed from points
        apply_mapping: Whether to map labels using LEARNING_MAP
        auto_range: Automatically compute pc_range from point cloud
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        voxel_size: Voxel size in meters (used when occ_size is None)
        padding: Padding for auto_range
        verbose: Print debug information
        
    Returns:
        voxel_grid: 3D numpy array with semantic labels
        pc_range: The pc_range used (useful when auto_range=True)
    """
    # Convert depth to point cloud
    points = depth_to_pointcloud(depth)
    points_flat = points.reshape(-1, 3)
    labels_flat = seg.reshape(-1)

    # Apply label mapping
    if apply_mapping:
        labels_flat = map_labels(labels_flat)

    # Valid depth mask
    depth_flat = depth.reshape(-1)
    valid_mask = (depth_flat > min_depth) & (depth_flat < max_depth)

    points_valid = points_flat[valid_mask]
    labels_valid = labels_flat[valid_mask]
    
    if verbose:
        print(f"Valid points: {len(points_valid):,}")
        print(f"Point cloud bounds:")
        print(f"  X: [{points_valid[:, 0].min():.2f}, {points_valid[:, 0].max():.2f}]")
        print(f"  Y: [{points_valid[:, 1].min():.2f}, {points_valid[:, 1].max():.2f}]")
        print(f"  Z: [{points_valid[:, 2].min():.2f}, {points_valid[:, 2].max():.2f}]")

    # Compute pc_range if needed
    if pc_range is None:
        if auto_range:
            pc_range = compute_pc_range_from_points(points_valid, padding=padding)
            if verbose:
                print(f"Auto-computed pc_range: {pc_range}")
        else:
            pc_range = DEFAULT_PC_RANGE
            if verbose:
                print(f"Using DEFAULT_PC_RANGE: {pc_range}")
    
    pc_range = np.array(pc_range)
    
    # Compute occ_size if needed
    if occ_size is None:
        occ_size = compute_occ_size_from_range(pc_range, voxel_size)
        if verbose:
            print(f"Computed occ_size: {occ_size}")
    
    occ_size = np.array(occ_size)

    # Filter points within pc_range
    in_range = (
        (points_valid[:, 0] >= pc_range[0])
        & (points_valid[:, 0] < pc_range[3])
        & (points_valid[:, 1] >= pc_range[1])
        & (points_valid[:, 1] < pc_range[4])
        & (points_valid[:, 2] >= pc_range[2])
        & (points_valid[:, 2] < pc_range[5])
    )

    points_in = points_valid[in_range]
    labels_in = labels_valid[in_range]
    
    if verbose:
        pct = 100.0 * len(points_in) / max(1, len(points_valid))
        print(f"Points in range: {len(points_in):,} / {len(points_valid):,} ({pct:.1f}%)")
        if pct < 90:
            print("  WARNING: Many points outside pc_range. Consider using auto_range=True")

    # Create voxel grid
    voxel_grid = np.zeros(occ_size, dtype=np.uint8)

    if len(points_in) == 0:
        return voxel_grid, tuple(pc_range.tolist())

    # Compute voxel indices
    voxel_size_arr = (pc_range[3:6] - pc_range[0:3]) / occ_size
    voxel_indices = ((points_in - pc_range[0:3]) / voxel_size_arr).astype(np.int32)
    voxel_indices = np.clip(voxel_indices, 0, occ_size - 1)

    # Majority voting for semantic labels
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

    return voxel_grid, tuple(pc_range.tolist())


def voxelize_with_fixed_grid(
    depth, 
    seg, 
    occ_size=DEFAULT_OCC_SIZE,
    voxel_size=0.2,
    apply_mapping=True,
    min_depth=0.1,
    max_depth=100.0,
    verbose=False
):
    """
    Voxelize with fixed grid size, auto-computing pc_range to fit points.
    
    This ensures consistent grid dimensions while adapting to any scene.
    The grid is centered on the point cloud and sized to capture most points.
    
    Args:
        depth: Depth image
        seg: Segmentation mask
        occ_size: Fixed voxel grid dimensions (X, Y, Z)
        voxel_size: Voxel size in meters
        apply_mapping: Map labels
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        verbose: Print debug info
        
    Returns:
        voxel_grid: 3D voxel grid
        pc_range: The computed pc_range
    """
    occ_size = np.array(occ_size)
    
    # Get valid points
    points = depth_to_pointcloud(depth).reshape(-1, 3)
    depth_flat = depth.reshape(-1)
    valid = (depth_flat > min_depth) & (depth_flat < max_depth)
    points_valid = points[valid]
    
    if len(points_valid) == 0:
        return np.zeros(occ_size, dtype=np.uint8), DEFAULT_PC_RANGE
    
    # Compute extents from occ_size and voxel_size
    extents = occ_size * voxel_size
    
    # Center the grid to capture the bulk of points
    # Use percentile to ignore outliers
    p_low = np.percentile(points_valid, 5, axis=0)
    p_high = np.percentile(points_valid, 95, axis=0)
    center = (p_low + p_high) / 2
    
    # Compute range centered on the point cloud
    min_bounds = center - extents / 2
    max_bounds = center + extents / 2
    
    # Adjust Z to start near camera (Z >= 0 for forward-facing)
    # Shift to capture from minimum Z
    z_shift = max(0, points_valid[:, 2].min() - 1.0) - min_bounds[2]
    min_bounds[2] += z_shift
    max_bounds[2] += z_shift
    
    pc_range = tuple(min_bounds.tolist() + max_bounds.tolist())
    
    if verbose:
        print(f"Fixed occ_size: {tuple(occ_size.astype(int))}")
        print(f"Voxel size: {voxel_size}m")
        print(f"Grid extents: X={extents[0]:.1f}m, Y={extents[1]:.1f}m, Z={extents[2]:.1f}m")
        print(f"Computed pc_range: [{min_bounds[0]:.1f}, {min_bounds[1]:.1f}, {min_bounds[2]:.1f}] to [{max_bounds[0]:.1f}, {max_bounds[1]:.1f}, {max_bounds[2]:.1f}]")
    
    return voxelize_depth_seg(
        depth, seg,
        occ_size=tuple(occ_size.astype(int)),
        pc_range=pc_range,
        apply_mapping=apply_mapping,
        auto_range=False,
        min_depth=min_depth,
        max_depth=max_depth,
        verbose=verbose
    )


# =============================================================================
# Visualization
# =============================================================================

def create_coordinate_frame(size=5.0, origin=[0, 0, 0]):
    """Create coordinate frame as LineSet."""
    origin = np.array(origin)
    points = np.array([
        origin,
        origin + [size, 0, 0],
        origin + [0, size, 0],
        origin + [0, 0, size],
    ])
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    frame = o3d.geometry.LineSet()
    frame.points = o3d.utility.Vector3dVector(points)
    frame.lines = o3d.utility.Vector2iVector(lines)
    frame.colors = o3d.utility.Vector3dVector(colors)
    return frame


def visualize_voxel_grid(voxel_grid, pc_range, show_frame=True):
    """Visualize voxel grid with Open3D."""
    voxel_grid = np.array(voxel_grid)
    occ_size = np.array(voxel_grid.shape)
    pc_range = np.array(pc_range)

    grid_min = pc_range[0:3]
    grid_max = pc_range[3:6]
    voxel_size = (grid_max - grid_min) / occ_size

    xs, ys, zs = np.nonzero(voxel_grid)
    
    if len(xs) == 0:
        print("Warning: Voxel grid is empty!")
        return
    
    labels = voxel_grid[xs, ys, zs]
    pts = grid_min + np.stack([xs, ys, zs], axis=1) * voxel_size + voxel_size * 0.5
    
    cmap = np.array(COLOR_MAP)[:, ::-1] / 255.0  # BGR to RGB
    colors = cmap[np.clip(labels, 0, len(cmap)-1)]
    
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pc]
    
    if show_frame:
        frame = create_coordinate_frame(size=5.0)
        geometries.append(frame)

    o3d.visualization.draw_geometries(geometries)


def visualize_voxel_semantic(voxel_grid, pc_range=None, show_legend=True):
    """Visualize voxel grid with colors."""
    if pc_range is None:
        pc_range = DEFAULT_PC_RANGE
    
    occupied = np.argwhere(voxel_grid > 0)
    labels = voxel_grid[voxel_grid > 0]
    
    print(f"Visualizing {len(occupied):,} occupied voxels")
    visualize_voxel_grid(voxel_grid, pc_range)
    
    return occupied, labels


# =============================================================================
# Main Functions
# =============================================================================

def generate_voxel_from_depth(
    depth_path,
    seg_path,
    output_path=None,
    occ_size=None,
    pc_range=None,
    auto_range=True,
    voxel_size=0.2,
    save=True,
    visualize=True,
    verbose=True,
):
    """
    Generate voxel grid from depth and segmentation.
    
    Args:
        depth_path: Path to depth image
        seg_path: Path to segmentation image
        output_path: Path to save voxel grid (None to skip saving)
        occ_size: Voxel grid dimensions (None to auto-compute)
        pc_range: Point cloud range (None to auto-compute)
        auto_range: Auto-compute pc_range from point cloud
        voxel_size: Voxel size in meters
        save: Whether to save the voxel grid
        visualize: Whether to visualize
        verbose: Print debug info
        
    Returns:
        voxel_grid: The generated voxel grid
        pc_range: The pc_range used
    """
    if verbose:
        print(f"Loading depth: {depth_path}")
        print(f"Loading seg: {seg_path}")
    
    depth = read_depth(depth_path)
    seg = read_seg(seg_path)

    voxel_grid, used_pc_range = voxelize_depth_seg(
        depth, seg,
        occ_size=occ_size,
        pc_range=pc_range,
        auto_range=auto_range,
        voxel_size=voxel_size,
        verbose=verbose
    )
    
    if save and output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        np.save(output_path, voxel_grid)
        # Also save the pc_range for later visualization
        meta_path = output_path.replace('.npy', '_meta.npz')
        np.savez(meta_path, pc_range=used_pc_range, occ_size=voxel_grid.shape)
        if verbose:
            print(f"Saved: {output_path}")
            print(f"Saved: {meta_path}")

    if visualize:
        visualize_voxel_semantic(voxel_grid, used_pc_range, show_legend=False)

    return voxel_grid, used_pc_range

def _worker_generate_voxel(args):
    """
    Worker function for multiprocessing.
    
    Takes a tuple of arguments to avoid pickle issues with complex types.
    All arguments should be plain Python types (str, tuple of floats, etc.)
    """
    (depth_path, seg_path, output_path, occ_size, pc_range, 
     auto_range, voxel_size, apply_mapping, min_depth, max_depth) = args
    
    try:
        # Convert tuples back to proper types inside worker
        if occ_size is not None:
            occ_size = tuple(int(x) for x in occ_size)
        if pc_range is not None:
            pc_range = tuple(float(x) for x in pc_range)
        
        # Read data inside worker (don't pass numpy arrays between processes)
        depth = read_depth(depth_path)
        seg = read_seg(seg_path)
        
        # Process
        voxel_grid, used_pc_range = voxelize_depth_seg(
            depth, seg,
            occ_size=occ_size,
            pc_range=pc_range,
            apply_mapping=apply_mapping,
            auto_range=auto_range,
            min_depth=min_depth,
            max_depth=max_depth,
            voxel_size=voxel_size,
            verbose=False
        )
        
        # Save
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            np.save(output_path, voxel_grid)
        
        return True, depth_path, None
        
    except Exception as e:
        import traceback
        return False, depth_path, str(e) + "\n" + traceback.format_exc()

def process_dataset(
    data_root,
    output_dir=None,
    occ_size=DEFAULT_OCC_SIZE,
    pc_range=None,
    auto_range=True,
    voxel_size=0.2,
    num_workers=1,
    camera="lcam_front",
    save=True,
    apply_mapping=True,
    min_depth=0.1,
    max_depth=100.0,
):
    """
    Process entire dataset to generate voxel labels.
    
    Args:
        data_root: Root directory of dataset
        output_dir: Output directory (optional)
        occ_size: Voxel grid dimensions (tuple of 3 ints)
        pc_range: Point cloud range (tuple of 6 floats), or None for auto
        auto_range: Automatically compute pc_range from point cloud
        voxel_size: Voxel size in meters
        num_workers: Number of parallel workers (1 for sequential)
        camera: Camera name
        save: Whether to save voxel grids
        apply_mapping: Whether to apply label mapping
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    """
    depth_pattern = os.path.join(data_root, "P*", f"depth_{camera}", "*.png")
    print("depth_pattern =", depth_pattern)
    depth_files = sorted(glob.glob(depth_pattern))

    if len(depth_files) == 0:
        depth_files = sorted(glob.glob(os.path.join(data_root, f"*_{camera}_depth.png")))

    print(f"Found {len(depth_files)} depth files")

    # Prepare tasks - use only plain Python types for multiprocessing safety
    tasks = []
    
    # Convert occ_size and pc_range to plain Python tuples
    if occ_size is not None:
        occ_size = tuple(int(x) for x in occ_size)
    if pc_range is not None:
        pc_range = tuple(float(x) for x in pc_range)
    
    for depth_path in depth_files:
        depth_dir = os.path.dirname(depth_path)
        traj_dir = os.path.dirname(depth_dir)
        frame_id = os.path.basename(depth_path).replace(".png", "").split("_")[0]
        seg_path = os.path.join(traj_dir, f"seg_{camera}", f"{frame_id}_{camera}_seg.png")

        if not os.path.exists(seg_path):
            continue
        
        output_path = None
        if save:
            output_path = os.path.join(traj_dir, f"voxel_label_{camera}", f"{frame_id}_voxel_label.npy")
        
        # Pack all arguments as plain Python types
        task_args = (
            str(depth_path),      # Ensure string
            str(seg_path),        # Ensure string
            str(output_path) if output_path else None,
            occ_size if not auto_range else None,  # Already tuple of ints or None
            pc_range,             # Already tuple of floats or None
            bool(auto_range),
            float(voxel_size),
            bool(apply_mapping),
            float(min_depth),
            float(max_depth),
        )
        tasks.append(task_args)

    print(f"Processing {len(tasks)} frames with {num_workers} workers...")

    successful = 0
    failed = 0
    errors = []

    if num_workers > 1:
        # Multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_worker_generate_voxel, task): task[0] for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    success, path, error = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        errors.append((path, error))
                except Exception as e:
                    failed += 1
                    errors.append((futures[future], str(e)))
    else:
        # Sequential processing
        for task_args in tqdm(tasks):
            success, path, error = _worker_generate_voxel(task_args)
            if success:
                successful += 1
            else:
                failed += 1
                errors.append((path, error))

    print(f"\nCompleted: {successful} successful, {failed} failed")
    
    if errors and len(errors) <= 10:
        print("\nErrors:")
        for path, error in errors:
            print(f"  {path}: {error}")
    elif errors:
        print(f"\nFirst 5 errors (of {len(errors)}):")
        for path, error in errors[:5]:
            print(f"  {path}: {error}")
    
    return successful, failed


# def process_dataset(
#     data_root,
#     output_dir=None,
#     occ_size=DEFAULT_OCC_SIZE,
#     auto_range=True,
#     voxel_size=0.2,
#     num_workers=1,
#     camera="lcam_front",
#     save=True,
# ):
#     """Process entire dataset to generate voxel labels."""
#     depth_pattern = os.path.join(data_root, "P*", f"depth_{camera}", "*.png")
#     print("depth_pattern =", depth_pattern)
#     depth_files = sorted(glob.glob(depth_pattern))

#     if len(depth_files) == 0:
#         depth_files = sorted(glob.glob(os.path.join(data_root, f"*_{camera}_depth.png")))

#     print(f"Found {len(depth_files)} depth files")

#     tasks = []
#     for depth_path in depth_files:
#         depth_dir = os.path.dirname(depth_path)
#         traj_dir = os.path.dirname(depth_dir)
#         frame_id = os.path.basename(depth_path).replace(".png", "").split("_")[0]
#         seg_path = os.path.join(traj_dir, f"seg_{camera}", f"{frame_id}_{camera}_seg.png")

#         if not os.path.exists(seg_path):
#             continue
            
#         output_path = os.path.join(traj_dir, f"voxel_label_{camera}", f"{frame_id}_voxel_label.npy")
#         tasks.append((depth_path, seg_path, output_path))

#     print(f"Processing {len(tasks)} frames...")

#     successful = 0
#     failed = 0
#     if num_workers > 1:
#         with ProcessPoolExecutor(max_workers=num_workers) as executor:
#             futures = {
#                 executor.submit(
#                     generate_voxel_from_depth,
#                     depth_path,
#                     seg_path,
#                     output_path,
#                     occ_size,
#                     auto_range,
#                     voxel_size,
#                     save,
#                     False,
#                     False,
#                     False
#                     ): (depth_path, output_path)
#                 for depth_path, seg_path, output_path in tasks
#             }

#             for future in tqdm(as_completed(futures), total=len(futures)):
#                 if future.result():
#                     successful += 1
#                 else:
#                     failed += 1
#     else:
#         for depth_path, seg_path, output_path in tqdm(tasks):
#             try:
#                 generate_voxel_from_depth(
#                     depth_path, seg_path, output_path,
#                     occ_size=occ_size if not auto_range else None,
#                     auto_range=auto_range,
#                     voxel_size=voxel_size,
#                     save=save,
#                     visualize=False,
#                     verbose=False,
#                 )
#                 successful += 1
#             except Exception as e:
#                 print(f"Error processing {depth_path}: {e}")
#                 failed += 1

#     print(f"\nCompleted: {successful} successful, {failed} failed")
#     return successful, failed


def main():
    parser = argparse.ArgumentParser(description='Generate voxel labels from depth')
    parser.add_argument('--depth_file', type=str, help='Single depth file')
    parser.add_argument('--seg_file', type=str, help='Single segmentation file')
    parser.add_argument('--data_root', type=str, help='Dataset root for batch processing')
    parser.add_argument('--camera', type=str, default='lcam_front', help='Camera name')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no_auto_range', action='store_true', help='Disable auto range')
    parser.add_argument('--voxel_size', type=float, default=0.2, help='Voxel size in meters')
    parser.add_argument('--no_visualize', action='store_true', help='Skip visualization')
    parser.add_argument('--no_save', action='store_true', help='Skip saving')

    args = parser.parse_args()
    
    if args.depth_file and args.seg_file:
        # Single file processing
        generate_voxel_from_depth(
            args.depth_file,
            args.seg_file,
            output_path=args.output,
            auto_range=not args.no_auto_range,
            voxel_size=args.voxel_size,
            save=not args.no_save,
            visualize=not args.no_visualize,
            verbose=True,
        )
        # 
    elif args.data_root:
        # Batch processing
        process_dataset(
            args.data_root,
            auto_range=not args.no_auto_range,
            voxel_size=args.voxel_size,
            camera=args.camera,
            save=not args.no_save,
            num_workers=32
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    # Example usage - modify paths as needed
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        # Default test
        id = 0
        depth_path = f"data/tartanair/CarWelding/Data_easy/P000/depth_lcam_front/{str(id).zfill(6)}_lcam_front_depth.png"
        seg_path = f"data/tartanair/CarWelding/Data_easy/P000/seg_lcam_front/{str(id).zfill(6)}_lcam_front_seg.png"
        
        if os.path.exists(depth_path) and os.path.exists(seg_path):
            generate_voxel_from_depth(
                depth_path, seg_path, 
                output_path=None,
                auto_range=True,  # KEY: auto-compute range for any scene
                visualize=True,
                verbose=True
            )
        else:
            print("Usage:")
            print("  python script.py --depth_file depth.png --seg_file seg.png")
            print("  python script.py --data_root /path/to/dataset --camera lcam_front")
    
    
#!/usr/bin/env python3
"""
Visualize voxel predictions for TartanAir dataset.
Adapted from SemanticKITTI visualization code.

Usage:
    python visualize_tartanair.py --pred_root /path/to/predictions --data_root /path/to/tartanair
    python visualize_tartanair.py --voxel_file prediction.npy --output_dir ./vis_output
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import argparse
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from mayavi import mlab
mlab.options.offscreen = True

# =============================================================================
# TartanAir Configuration
# =============================================================================

# 34 classes for TartanAir (matching LEARNING_MAP in generate_voxels_fixed.py)
CLASS_NAMES = [
    'unlabeled',      # 0
    'cabinet',        # 1
    'pipecover',      # 2
    'metalpanel',     # 3
    'metalhandrail',  # 4
    'light',          # 5
    'cieling',        # 6
    'metalplatform',  # 7
    'chair',          # 8
    'cementcolumn',   # 9
    'plug',           # 10
    'ceiling',        # 11
    'metalpole',      # 12
    'ventpipe',       # 13
    'metalramp',      # 14
    'car',            # 15
    'metalfloor',     # 16
    'toolbox',        # 17
    'ventpipeclamp',  # 18
    'metalstair',     # 19
    'lamp',           # 20
    'tireassembly',   # 21
    'metalcieling',   # 22
    'platform',       # 23
    'monitor',        # 24
    'wall',           # 25
    'door',           # 26
    'sky',            # 27
    'cable',          # 28
    'chasis',         # 29
    'floor',          # 30
    'table',          # 31
    'robotarm',       # 32
    'keyboard',       # 33
]

# Colors for TartanAir classes (34 classes)
# Format: [R, G, B, A]
COLORS = np.array([
    [0, 0, 0, 255],         # 0: unlabeled - black
    [153, 108, 6, 255],     # 1: cabinet - brown
    [112, 105, 191, 255],   # 2: pipecover - purple
    [89, 121, 72, 255],     # 3: metalpanel - olive
    [190, 225, 64, 255],    # 4: metalhandrail - lime
    [206, 190, 59, 255],    # 5: light - yellow
    [81, 13, 36, 255],      # 6: cieling - dark red
    [115, 176, 195, 255],   # 7: metalplatform - cyan
    [161, 171, 27, 255],    # 8: chair - olive green
    [135, 169, 180, 255],   # 9: cementcolumn - gray blue
    [29, 26, 199, 255],     # 10: plug - blue
    [102, 16, 239, 255],    # 11: ceiling - violet
    [242, 107, 146, 255],   # 12: metalpole - pink
    [156, 198, 23, 255],    # 13: ventpipe - yellow green
    [49, 89, 160, 255],     # 14: metalramp - steel blue
    [68, 218, 116, 255],    # 15: car - green
    [11, 236, 9, 255],      # 16: metalfloor - bright green
    [196, 30, 8, 255],      # 17: toolbox - red
    [121, 67, 28, 255],     # 18: ventpipeclamp - brown
    [0, 53, 65, 255],       # 19: metalstair - dark teal
    [146, 52, 70, 255],     # 20: lamp - maroon
    [226, 149, 143, 255],   # 21: tireassembly - salmon
    [151, 126, 171, 255],   # 22: metalcieling - lavender
    [194, 39, 7, 255],      # 23: platform - orange red
    [205, 120, 161, 255],   # 24: monitor - pink
    [212, 51, 60, 255],     # 25: wall - red
    [211, 80, 208, 255],    # 26: door - magenta
    [189, 135, 188, 255],   # 27: sky - light purple
    [54, 72, 205, 255],     # 28: cable - blue
    [103, 252, 157, 255],   # 29: chasis - mint
    [124, 21, 123, 255],    # 30: floor - purple
    [19, 132, 69, 255],     # 31: table - green
    [195, 237, 132, 255],   # 32: robotarm - light green
    [94, 253, 175, 255],    # 33: keyboard - aqua
]).astype(np.uint8)

# TartanAir camera intrinsics
FOCAL_LENGTH = 320.0
PRINCIPAL_POINT = (320.0, 320.0)
IMAGE_SIZE = (640, 640)  # (width, height)

# Voxel grid configuration (SemanticKITTI standard)
VOX_ORIGIN = np.array([0, -25.6, -2])
VOX_SIZE = 0.2
GRID_DIMS = [256, 256, 32]

# Camera to LiDAR transformation
# Camera: X-right, Y-down, Z-forward
# LiDAR: X-forward, Y-left, Z-up
CAM_TO_LIDAR = np.array([
    [0,  0,  1, 0],
    [-1, 0,  0, 0],
    [0, -1,  0, 0],
    [0,  0,  0, 1],
], dtype=np.float64)

LIDAR_TO_CAM = np.array([
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1],
], dtype=np.float64)


# =============================================================================
# Utility Functions
# =============================================================================

def get_grid_coords(dims, resolution):
    """
    Get the center coordinates of voxels in the grid.
    
    Args:
        dims: Grid dimensions [x, y, z]
        resolution: Voxel size in meters
        
    Returns:
        coords_grid: (N, 3) array of voxel center coordinates
    """
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])
    
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz, indexing='ij')
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    
    # Convert to world coordinates (center of each voxel)
    coords_grid = (coords_grid * resolution) + resolution / 2 + VOX_ORIGIN
    
    return coords_grid


def get_fov_mask_tartanair(grid_dims=GRID_DIMS, vox_size=VOX_SIZE, vox_origin=VOX_ORIGIN,
                           img_size=IMAGE_SIZE, focal=FOCAL_LENGTH, cx=None, cy=None):
    """
    Compute FOV mask for TartanAir camera.
    
    Returns mask indicating which voxels are visible from the camera.
    """
    if cx is None:
        cx = PRINCIPAL_POINT[0]
    if cy is None:
        cy = PRINCIPAL_POINT[1]
    
    # Create voxel coordinates
    xv, yv, zv = np.meshgrid(
        range(grid_dims[0]),
        range(grid_dims[1]),
        range(grid_dims[2]),
        indexing='ij'
    )
    vox_coords = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1).astype(np.float32)
    
    # Convert to world coordinates (LiDAR frame)
    offsets = np.array([0.5, 0.5, 0.5])
    world_pts = vox_coords * vox_size + vox_size * offsets + vox_origin
    
    # Transform from LiDAR to camera frame
    world_pts_h = np.hstack([world_pts, np.ones((len(world_pts), 1))])
    cam_pts = (LIDAR_TO_CAM @ world_pts_h.T).T[:, :3]
    
    # Project to image plane
    # In camera frame: X-right, Y-down, Z-forward
    pix_x = (cam_pts[:, 0] * focal) / cam_pts[:, 2] + cx
    pix_y = (cam_pts[:, 1] * focal) / cam_pts[:, 2] + cy
    pix_z = cam_pts[:, 2]
    
    # Check if within image bounds and in front of camera
    fov_mask = (
        (pix_x >= 0) & (pix_x < img_size[0]) &
        (pix_y >= 0) & (pix_y < img_size[1]) &
        (pix_z > 0)
    )
    
    return fov_mask


def draw_voxels(
    voxels,
    vox_origin=VOX_ORIGIN,
    fov_mask=None,
    voxel_size=VOX_SIZE,
    save_name="prediction",
    save_root="./vis_output",
    show_camera=True,
    view_type="bev",  # "bev", "side", "front", "custom"
    show_outfov=True,
    azimuth=None,
    elevation=None,
    distance=None,
):
    """
    Visualize voxel grid using Mayavi.
    
    Args:
        voxels: 3D numpy array of shape (256, 256, 32) with class labels
        vox_origin: Origin of the voxel grid
        fov_mask: Boolean mask for field of view (optional)
        voxel_size: Size of each voxel in meters
        save_name: Name for the saved image
        save_root: Directory to save the visualization
        show_camera: Whether to draw camera frustum
        video_view: Use video-style camera angle
        show_outfov: Whether to show voxels outside FOV
        
    Returns:
        save_file: Path to saved image
    """
    # Get voxel coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], 
        voxel_size
    )
    
    # Attach class labels to coordinates
    grid_coords = np.hstack([grid_coords, voxels.reshape(-1, 1)])
    
    # Create FOV mask if not provided
    if fov_mask is None:
        fov_mask = get_fov_mask_tartanair()
    
    # Split by FOV
    fov_grid_coords = grid_coords[fov_mask, :]
    outfov_grid_coords = grid_coords[~fov_mask, :]
    
    # Remove empty voxels (class 0) and unknown (255)
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
    ]
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
    ]
    
    print(f"FOV voxels: {len(fov_voxels):,}, OutFOV voxels: {len(outfov_voxels):,}")
    
    if len(fov_voxels) == 0 and len(outfov_voxels) == 0:
        print("Warning: No occupied voxels to visualize!")
        return None
    
    # Create figure
    figure = mlab.figure(size=(2800, 2800), bgcolor=(1, 1, 1))
    
    # Draw camera frustum
    if show_camera:
        d = 7  # Distance for camera visualization
        fx = FOCAL_LENGTH
        x = d * IMAGE_SIZE[0] / (2 * fx)
        y = d * IMAGE_SIZE[1] / (2 * fx)
        
        # Camera frustum in camera frame
        tri_points = np.array([
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ])
        
        # Transform to LiDAR frame
        tri_points_h = np.hstack([tri_points, np.ones((5, 1))])
        tri_points_lidar = (CAM_TO_LIDAR @ tri_points_h.T).T[:, :3]
        
        # Offset by voxel origin for visualization
        cam_x = tri_points_lidar[:, 0]
        cam_y = tri_points_lidar[:, 1]
        cam_z = tri_points_lidar[:, 2]
        
        triangles = [
            (0, 1, 2),
            (0, 1, 4),
            (0, 3, 4),
            (0, 2, 3),
        ]
        
        mlab.triangular_mesh(
            cam_x, cam_y, cam_z,
            triangles,
            representation="wireframe",
            color=(0, 0, 0),
            line_width=5,
        )
    
    # Draw FOV voxels
    if len(fov_voxels) > 0:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="viridis",
            scale_factor=voxel_size - 0.05 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=1,
            vmax=len(CLASS_NAMES) - 1,
        )
        
        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = COLORS
    
    # Draw out-of-FOV voxels (darker)
    if show_outfov and len(outfov_voxels) > 0:
        plt_plot_outfov = mlab.points3d(
            outfov_voxels[:, 0],
            outfov_voxels[:, 1],
            outfov_voxels[:, 2],
            outfov_voxels[:, 3],
            colormap="viridis",
            scale_factor=voxel_size - 0.05 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=1,
            vmax=len(CLASS_NAMES) - 1,
        )
        
        outfov_colors = COLORS.copy()
        outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2  # Darken
        plt_plot_outfov.glyph.scale_mode = "scale_by_vector"
        plt_plot_outfov.module_manager.scalar_lut_manager.lut.table = outfov_colors
    
    # Set camera view
    # Grid center: X=25.6, Y=0, Z=1.2 (for grid [0,51.2], [-25.6,25.6], [-2,4.4])
    scene = figure.scene
    
    # Compute grid center for focal point
    grid_center = [
        vox_origin[0] + GRID_DIMS[0] * voxel_size / 2,  # 25.6
        vox_origin[1] + GRID_DIMS[1] * voxel_size / 2,  # 0
        vox_origin[2] + GRID_DIMS[2] * voxel_size / 2,  # 1.2
    ]
    
    if view_type == "side" or view_type == "video":
        # Side/perspective view - closer
        scene.camera.position = [-15, -40, 35]
        scene.camera.focal_point = grid_center
        scene.camera.view_angle = 45.0
        scene.camera.view_up = [0, 0, 1]
    elif view_type == "front":
        # Front view (behind camera looking forward) - closer
        scene.camera.position = [-5, 0, 10]
        scene.camera.focal_point = [30, 0, 1]
        scene.camera.view_angle = 60.0
        scene.camera.view_up = [0, 0, 1]
    elif view_type == "top":
        # Pure top-down view - closer
        scene.camera.position = [25.6, 0, 80]
        scene.camera.focal_point = grid_center
        scene.camera.view_angle = 45.0
        scene.camera.view_up = [1, 0, 0]
    elif view_type == "custom" and azimuth is not None:
        # Custom view using azimuth/elevation/distance
        dist = distance if distance else 80  # Reduced default distance
        az_rad = np.radians(azimuth if azimuth else 45)
        el_rad = np.radians(elevation if elevation else 45)
        
        cam_x = grid_center[0] + dist * np.cos(el_rad) * np.cos(az_rad)
        cam_y = grid_center[1] + dist * np.cos(el_rad) * np.sin(az_rad)
        cam_z = grid_center[2] + dist * np.sin(el_rad)
        
        scene.camera.position = [cam_x, cam_y, cam_z]
        scene.camera.focal_point = grid_center
        scene.camera.view_angle = 45.0
        scene.camera.view_up = [0, 0, 1]
    else:
        # Default: Bird's eye view (BEV) - camera closer and above
        scene.camera.position = [-20, 0, 60]
        scene.camera.focal_point = grid_center
        scene.camera.view_angle = 55.0
        scene.camera.view_up = [1, 0, 0]
    
    scene.camera.clipping_range = [1, 500]
    
    scene.camera.compute_view_plane_normal()
    scene.render()
    
    # Save
    os.makedirs(save_root, exist_ok=True)
    save_file = f"{save_name}.png"
    save_path = os.path.join(save_root, save_file)
    mlab.savefig(save_path)
    print(f"Saved: {save_path}")
    mlab.clf()
    
    return save_file


def load_voxel_file(file_path):
    """
    Load voxel data from various file formats.
    
    Supported formats:
        - .npy: NumPy array
        - .bin: Binary uint16
        - .label: Binary uint16 (SemanticKITTI format)
    
    Args:
        file_path: Path to voxel file
        
    Returns:
        voxels: numpy array of shape (256, 256, 32)
    """
    if file_path.endswith('.npy'):
        voxels = np.load(file_path)
    elif file_path.endswith('.bin') or file_path.endswith('.label'):
        # Binary format (uint16) - SemanticKITTI style
        voxels = np.fromfile(file_path, dtype=np.uint16).reshape(256, 256, 32)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return voxels


def visualize_prediction(pred_path, save_root="./vis_output", save_name=None, view_type="bev",
                         azimuth=None, elevation=None, distance=None):
    """
    Visualize a single prediction file.
    
    Args:
        pred_path: Path to prediction file (.npy, .bin, or .label)
        save_root: Output directory
        save_name: Name for saved image (default: filename without extension)
        view_type: Camera view type ("bev", "side", "front", "top", "custom")
        azimuth: Custom view azimuth angle (degrees)
        elevation: Custom view elevation angle (degrees)
        distance: Custom view distance
    """
    if save_name is None:
        save_name = os.path.splitext(os.path.basename(pred_path))[0]
    
    # Load prediction
    voxels = load_voxel_file(pred_path)
    
    print(f"Loaded voxels: {voxels.shape}, unique classes: {np.unique(voxels)}")
    
    # Visualize
    fov_mask = get_fov_mask_tartanair()
    draw_voxels(
        voxels,
        fov_mask=fov_mask,
        save_name=save_name,
        save_root=save_root,
        view_type=view_type,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
    )


def visualize_dataset(pred_root, data_root, sequence="P004", camera="lcam_front",
                      save_root="./vis_output", view_type="bev"):
    """
    Visualize predictions for a TartanAir sequence.
    
    Args:
        pred_root: Root directory containing predictions
        data_root: Root directory of TartanAir dataset
        sequence: Sequence name (e.g., "P004")
        camera: Camera name
        save_root: Output directory for visualizations
        view_type: Camera view type
    """
    # Find prediction files
    pred_dir = os.path.join(pred_root, sequence, "predictions")
    if not os.path.exists(pred_dir):
        pred_dir = os.path.join(pred_root, sequence)
    
    if not os.path.exists(pred_dir):
        print(f"Prediction directory not found: {pred_dir}")
        return
    
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.npy', '.bin', '.label'))])
    print(f"Found {len(pred_files)} prediction files in {pred_dir}")
    
    # Compute FOV mask once
    fov_mask = get_fov_mask_tartanair()
    
    for pred_file in pred_files:
        pred_path = os.path.join(pred_dir, pred_file)
        frame_id = os.path.splitext(pred_file)[0]
        
        # Output directory per frame
        frame_save_root = os.path.join(save_root, sequence, frame_id)
        output_path = os.path.join(frame_save_root, "prediction.png")
        
        if os.path.exists(output_path):
            print(f"Skipping {frame_id} (already exists)")
            continue
        
        # Load prediction
        voxels = load_voxel_file(pred_path)
        
        print(f"Visualizing {frame_id}: shape={voxels.shape}, occupied={np.sum(voxels > 0):,}")
        
        # Visualize
        draw_voxels(
            voxels,
            fov_mask=fov_mask,
            save_name="prediction",
            save_root=frame_save_root,
            view_type=view_type,
        )


def visualize_gt_vs_pred(gt_path, pred_path, save_root="./vis_output", view_type="bev"):
    """
    Visualize ground truth and prediction side by side.
    
    Args:
        gt_path: Path to ground truth file (.npy, .bin, or .label)
        pred_path: Path to prediction file (.npy, .bin, or .label)
        save_root: Output directory
        view_type: Camera view type
    """
    gt_voxels = load_voxel_file(gt_path)
    pred_voxels = load_voxel_file(pred_path)
    
    print(f"GT shape: {gt_voxels.shape}, Pred shape: {pred_voxels.shape}")
    print(f"GT occupied: {np.sum(gt_voxels > 0):,}, Pred occupied: {np.sum(pred_voxels > 0):,}")
    
    fov_mask = get_fov_mask_tartanair()
    
    frame_id = os.path.splitext(os.path.basename(pred_path))[0]
    
    # Visualize GT
    draw_voxels(
        gt_voxels,
        fov_mask=fov_mask,
        save_name=f"{frame_id}_gt",
        save_root=save_root,
        view_type=view_type,
    )
    
    # Visualize prediction
    draw_voxels(
        pred_voxels,
        fov_mask=fov_mask,
        save_name=f"{frame_id}_pred",
        save_root=save_root,
        view_type=view_type,
    )


def visualize_side_by_side(
    gt_path,
    pred_path,
    save_path="./vis_output/comparison.png",
    view_type="bev",
    title_gt="Ground Truth",
    title_pred="Prediction",
    rgb_path=None,
    depth_path=None,
):
    """
    Visualize ground truth and prediction side by side in a single image,
    with optional RGB and depth images below.
    
    Args:
        gt_path: Path to ground truth file (.npy, .bin, or .label)
        pred_path: Path to prediction file (.npy, .bin, or .label)
        save_path: Path to save the combined image
        view_type: Camera view type ("bev", "side", "front", "top")
        title_gt: Title for ground truth panel
        title_pred: Title for prediction panel
        rgb_path: Path to RGB image (optional)
        depth_path: Path to depth image (optional)
    """
    import tempfile
    from PIL import Image, ImageDraw, ImageFont
    
    # Load voxels
    gt_voxels = load_voxel_file(gt_path)
    pred_voxels = load_voxel_file(pred_path)
    
    print(f"GT shape: {gt_voxels.shape}, Pred shape: {pred_voxels.shape}")
    print(f"GT occupied: {np.sum(gt_voxels > 0):,}, Pred occupied: {np.sum(pred_voxels > 0):,}")
    
    fov_mask = get_fov_mask_tartanair()
    
    # Create temp directory for individual renders
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Render GT
        draw_voxels(
            gt_voxels,
            fov_mask=fov_mask,
            save_name="gt",
            save_root=tmp_dir,
            view_type=view_type,
            show_camera=True,
        )
        
        # Render Prediction
        draw_voxels(
            pred_voxels,
            fov_mask=fov_mask,
            save_name="pred",
            save_root=tmp_dir,
            view_type=view_type,
            show_camera=True,
        )
        
        # Load rendered images
        gt_img = Image.open(os.path.join(tmp_dir, "gt.png"))
        pred_img = Image.open(os.path.join(tmp_dir, "pred.png"))
    
    # Get dimensions
    w, h = gt_img.size
    title_height = 60
    
    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 40)
            font_small = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 32)
        except:
            font = ImageFont.load_default()
            font_small = font
    
    # Check if we have RGB and depth images
    has_rgb_depth = rgb_path is not None and depth_path is not None
    if has_rgb_depth:
        has_rgb_depth = os.path.exists(rgb_path) and os.path.exists(depth_path)
    
    if has_rgb_depth:
        # Load RGB and depth images
        rgb_img = Image.open(rgb_path).convert('RGB')
        
        # Load depth - handle TartanAir format (RGBA with float32)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is not None and len(depth_raw.shape) == 3:
            # TartanAir depth format
            depth_data = depth_raw.view("<f4")
            depth_data = np.squeeze(depth_data, axis=-1)
        else:
            depth_data = depth_raw
        
        if depth_data is not None:
            # Normalize depth for visualization
            valid_mask = (depth_data > 0) & (depth_data < 100)
            if np.any(valid_mask):
                d_min = np.percentile(depth_data[valid_mask], 5)
                d_max = np.percentile(depth_data[valid_mask], 95)
            else:
                d_min, d_max = 0, 100
            
            depth_norm = np.clip((depth_data - d_min) / (d_max - d_min + 1e-6), 0, 1)
            depth_colored = (plt.cm.viridis(depth_norm)[:, :, :3] * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_colored)
        else:
            depth_img = Image.new('RGB', rgb_img.size, (128, 128, 128))
        
        # Resize RGB and depth to match voxel image width
        rgb_img = rgb_img.resize((w, int(w * rgb_img.height / rgb_img.width)), Image.LANCZOS)
        depth_img = depth_img.resize((w, int(w * depth_img.height / depth_img.width)), Image.LANCZOS)
        
        # Make sure both have same height
        img_h = max(rgb_img.height, depth_img.height)
        if rgb_img.height != img_h:
            rgb_img = rgb_img.resize((w, img_h), Image.LANCZOS)
        if depth_img.height != img_h:
            depth_img = depth_img.resize((w, img_h), Image.LANCZOS)
        
        # Create combined image (2 rows: voxels on top, RGB/depth on bottom)
        total_height = title_height + h + title_height + img_h
        combined = Image.new('RGB', (w * 2, total_height), color=(255, 255, 255))
        
        draw = ImageDraw.Draw(combined)
        
        # Draw voxel titles
        gt_text_bbox = draw.textbbox((0, 0), title_gt, font=font)
        pred_text_bbox = draw.textbbox((0, 0), title_pred, font=font)
        gt_text_w = gt_text_bbox[2] - gt_text_bbox[0]
        pred_text_w = pred_text_bbox[2] - pred_text_bbox[0]
        
        draw.text(((w - gt_text_w) // 2, 15), title_gt, fill=(0, 0, 0), font=font)
        draw.text((w + (w - pred_text_w) // 2, 15), title_pred, fill=(0, 0, 0), font=font)
        
        # Paste voxel images
        combined.paste(gt_img, (0, title_height))
        combined.paste(pred_img, (w, title_height))
        
        # Draw RGB/Depth titles
        rgb_title = "RGB Image"
        depth_title = "Depth Image"
        rgb_text_bbox = draw.textbbox((0, 0), rgb_title, font=font_small)
        depth_text_bbox = draw.textbbox((0, 0), depth_title, font=font_small)
        rgb_text_w = rgb_text_bbox[2] - rgb_text_bbox[0]
        depth_text_w = depth_text_bbox[2] - depth_text_bbox[0]
        
        y_title2 = title_height + h + 10
        draw.text(((w - rgb_text_w) // 2, y_title2), rgb_title, fill=(0, 0, 0), font=font_small)
        draw.text((w + (w - depth_text_w) // 2, y_title2), depth_title, fill=(0, 0, 0), font=font_small)
        
        # Paste RGB and depth images
        y_img = title_height + h + title_height
        combined.paste(rgb_img, (0, y_img))
        combined.paste(depth_img, (w, y_img))
        
    else:
        # Original behavior - just voxels side by side
        combined = Image.new('RGB', (w * 2, h + title_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(combined)
        
        # Draw titles
        gt_text_bbox = draw.textbbox((0, 0), title_gt, font=font)
        pred_text_bbox = draw.textbbox((0, 0), title_pred, font=font)
        gt_text_w = gt_text_bbox[2] - gt_text_bbox[0]
        pred_text_w = pred_text_bbox[2] - pred_text_bbox[0]
        
        draw.text(((w - gt_text_w) // 2, 15), title_gt, fill=(0, 0, 0), font=font)
        draw.text((w + (w - pred_text_w) // 2, 15), title_pred, fill=(0, 0, 0), font=font)
        
        # Paste images
        combined.paste(gt_img, (0, title_height))
        combined.paste(pred_img, (w, title_height))
    
    # Save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    combined.save(save_path)
    print(f"Saved comparison: {save_path}")
    
    return save_path


def visualize_sequence_comparison(
    gt_root,
    pred_root,
    save_root="./vis_output",
    sequence="P008",
    camera="lcam_front",
    view_type="bev",
    max_frames=None,
):
    """
    Visualize GT vs Prediction for all frames in a sequence.
    
    Args:
        gt_root: Root directory for ground truth voxels
                 e.g., "data/tartanair/CarWelding/Data_easy"
        pred_root: Root directory for predictions
        save_root: Output directory
        sequence: Sequence name (e.g., "P008")
        camera: Camera name (e.g., "lcam_front")
        view_type: Camera view type
        max_frames: Maximum number of frames to process (None for all)
    
    Example:
        visualize_sequence_comparison(
            gt_root="data/tartanair/CarWelding/Data_easy",
            pred_root="./predictions",
            sequence="P008",
            camera="lcam_front",
        )
    """
    # GT path format: {gt_root}/{sequence}/voxel_label_{camera}/
    gt_dir = os.path.join(gt_root, sequence, f"voxel_label_{camera}")
    
    # Also try alternative format: voxel_{camera}
    if not os.path.exists(gt_dir):
        gt_dir = os.path.join(gt_root, sequence, f"voxel_{camera}")
    
    if not os.path.exists(gt_dir):
        print(f"GT directory not found: {gt_dir}")
        return
    
    # RGB and depth directories
    rgb_dir = os.path.join(gt_root, sequence, f"image_{camera}")
    depth_dir = os.path.join(gt_root, sequence, f"depth_{camera}")
    
    print(f"RGB directory: {rgb_dir} (exists: {os.path.exists(rgb_dir)})")
    print(f"Depth directory: {depth_dir} (exists: {os.path.exists(depth_dir)})")
    
    # Find prediction directory
    pred_dir = os.path.join(pred_root, sequence, "predictions")
    if not os.path.exists(pred_dir):
        pred_dir = os.path.join(pred_root, sequence)
    if not os.path.exists(pred_dir):
        pred_dir = pred_root  # Maybe predictions are directly in pred_root
    
    if not os.path.exists(pred_dir):
        print(f"Prediction directory not found: {pred_dir}")
        return
    
    # Find GT files
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.npy', '.bin', '.label'))])
    print(f"Found {len(gt_files)} GT files in {gt_dir}")
    print(f"Looking for predictions in {pred_dir}")
    
    if max_frames:
        gt_files = gt_files[:max_frames]
    
    fov_mask = get_fov_mask_tartanair()
    
    for gt_file in gt_files:
        # Extract frame ID
        # Format: {frame_id}_voxel_label.npy or {frame_id}_voxel.npy or {frame_id}.label
        frame_id = gt_file.split('_')[0]
        
        gt_path = os.path.join(gt_dir, gt_file)
        
        # Try to find matching prediction (check multiple formats)
        pred_candidates = [
            os.path.join(pred_dir, f"{frame_id}.label"),
            os.path.join(pred_dir, f"{frame_id}.npy"),
            os.path.join(pred_dir, f"{frame_id}_pred.npy"),
            os.path.join(pred_dir, f"{frame_id}_pred.label"),
            os.path.join(pred_dir, f"{frame_id}_voxel.npy"),
            os.path.join(pred_dir, f"{frame_id}_voxel.label"),
            os.path.join(pred_dir, gt_file),  # Same filename
        ]
        
        pred_path = None
        for candidate in pred_candidates:
            if os.path.exists(candidate):
                pred_path = candidate
                break
        
        if pred_path is None:
            print(f"No prediction found for frame {frame_id}, skipping...")
            continue
        
        # Find RGB and depth images
        rgb_candidates = [
            os.path.join(rgb_dir, f"{frame_id}_{camera}.png"),
            os.path.join(rgb_dir, f"{frame_id}.png"),
        ]
        depth_candidates = [
            os.path.join(depth_dir, f"{frame_id}_{camera}_depth.png"),
            os.path.join(depth_dir, f"{frame_id}_depth.png"),
            os.path.join(depth_dir, f"{frame_id}.png"),
        ]
        
        rgb_path = None
        depth_path = None
        
        for candidate in rgb_candidates:
            if os.path.exists(candidate):
                rgb_path = candidate
                break
        
        for candidate in depth_candidates:
            if os.path.exists(candidate):
                depth_path = candidate
                break
        
        # Output path
        save_path = os.path.join(save_root, sequence, f"{frame_id}_comparison.png")
        
        if os.path.exists(save_path):
            print(f"Skipping {frame_id} (already exists)")
            continue
        
        print(f"Processing frame {frame_id}...")
        if rgb_path:
            print(f"  RGB: {rgb_path}")
        if depth_path:
            print(f"  Depth: {depth_path}")
        
        # Visualize side by side
        visualize_side_by_side(
            gt_path=gt_path,
            pred_path=pred_path,
            save_path=save_path,
            view_type=view_type,
            title_gt=f"Ground Truth - {frame_id}",
            title_pred=f"Prediction - {frame_id}",
            rgb_path=rgb_path,
            depth_path=depth_path,
        )


def main():
    parser = argparse.ArgumentParser(description='Visualize TartanAir voxel predictions')
    parser.add_argument('--voxel_file', type=str, help='Single voxel file to visualize')
    parser.add_argument('--pred_root', type=str, help='Prediction root directory')
    parser.add_argument('--data_root', type=str, help='TartanAir data root')
    parser.add_argument('--gt_file', type=str, help='Ground truth file for comparison')
    parser.add_argument('--gt_root', type=str, help='Ground truth root directory for batch comparison')
    parser.add_argument('--rgb_file', type=str, help='RGB image file for visualization')
    parser.add_argument('--depth_file', type=str, help='Depth image file for visualization')
    parser.add_argument('--sequence', type=str, default='P008', help='Sequence name')
    parser.add_argument('--camera', type=str, default='lcam_front', help='Camera name')
    parser.add_argument('--output_dir', type=str, default='./vis_output', help='Output directory')
    parser.add_argument('--view', type=str, default='bev', 
                        choices=['bev', 'side', 'front', 'top', 'video'],
                        help='Camera view type: bev (bird eye), side, front, top')
    parser.add_argument('--azimuth', type=float, help='Custom view azimuth angle (degrees)')
    parser.add_argument('--elevation', type=float, help='Custom view elevation angle (degrees)')
    parser.add_argument('--distance', type=float, help='Custom view distance')
    parser.add_argument('--side_by_side', action='store_true', help='Create side-by-side comparison')
    parser.add_argument('--max_frames', type=int, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Determine view type
    view_type = args.view
    if args.azimuth is not None or args.elevation is not None:
        view_type = "custom"
    
    if args.voxel_file and args.gt_file:
        # Single file comparison
        if args.side_by_side:
            frame_id = os.path.splitext(os.path.basename(args.voxel_file))[0]
            save_path = os.path.join(args.output_dir, f"{frame_id}_comparison.png")
            visualize_side_by_side(
                gt_path=args.gt_file,
                pred_path=args.voxel_file,
                save_path=save_path,
                view_type=view_type,
                rgb_path=args.rgb_file,
                depth_path=args.depth_file,
            )
        else:
            visualize_gt_vs_pred(args.gt_file, args.voxel_file, args.output_dir, view_type)
    
    elif args.voxel_file:
        # Single file visualization (no GT)
        visualize_prediction(
            args.voxel_file, 
            args.output_dir, 
            view_type=view_type,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
        )
    
    elif args.gt_root and args.pred_root:
        # Batch comparison for a sequence
        visualize_sequence_comparison(
            gt_root=args.gt_root,
            pred_root=args.pred_root,
            save_root=args.output_dir,
            sequence=args.sequence,
            camera=args.camera,
            view_type=view_type,
            max_frames=args.max_frames,
        )
    
    elif args.pred_root:
        # Batch visualization (predictions only)
        visualize_dataset(
            args.pred_root,
            args.data_root or args.pred_root,
            sequence=args.sequence,
            camera=args.camera,
            save_root=args.output_dir,
            view_type=view_type,
        )
    
    else:
        parser.print_help()
        print("\n" + "="*70)
        print("EXAMPLES")
        print("="*70)
        print("\n# Visualize single prediction (bird's eye view)")
        print("python visualize_tartanair.py --voxel_file prediction.label --output_dir ./vis")
        print()
        print("# Visualize with different views")
        print("python visualize_tartanair.py --voxel_file pred.label --view side")
        print("python visualize_tartanair.py --voxel_file pred.label --view front")
        print()
        print("# Side-by-side GT vs Prediction with RGB and Depth")
        print("python visualize_tartanair.py \\")
        print("    --gt_file data/tartanair/.../P008/voxel_label_lcam_front/000000_voxel_label.npy \\")
        print("    --voxel_file predictions/000000.label \\")
        print("    --rgb_file data/tartanair/.../P008/image_lcam_front/000000_lcam_front.png \\")
        print("    --depth_file data/tartanair/.../P008/depth_lcam_front/000000_lcam_front_depth.png \\")
        print("    --side_by_side --output_dir ./vis")
        print()
        print("# Batch side-by-side comparison (auto-finds RGB/depth)")
        print("python visualize_tartanair.py \\")
        print("    --gt_root data/tartanair/CarWelding/Data_easy \\")
        print("    --pred_root ./predictions \\")
        print("    --sequence P008 \\")
        print("    --camera lcam_front \\")
        print("    --output_dir ./vis")
        print()
        print("# Limit number of frames")
        print("python visualize_tartanair.py \\")
        print("    --gt_root data/tartanair/CarWelding/Data_easy \\")
        print("    --pred_root ./predictions \\")
        print("    --sequence P008 --max_frames 10")
        print()
        print("# Supported file formats: .npy, .bin, .label")


if __name__ == "__main__":
    main()
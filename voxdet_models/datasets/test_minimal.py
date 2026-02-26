"""
Minimal dataset test without torch/mmdet dependencies.
Tests the core functionality directly.
"""

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_pose_file(pose_file):
    """Load poses from file."""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) >= 7:
                # For now, just store the raw values
                poses.append(values)
    return poses


def depth_seg_to_occupancy_simple(depth_path, seg_path, focal_length=400.0, 
                                   image_width=800, image_height=600,
                                   occ_size=(256, 256, 32), 
                                   pc_range=(-40, -40, -1, 40, 40, 5.4)):
    """
    Simplified depth + segmentation to occupancy conversion.
    """
    print(f"  Loading depth: {os.path.basename(depth_path)}")
    print(f"  Loading segmentation: {os.path.basename(seg_path)}")
    
    # Load depth
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    print(f"    Depth shape: {depth_img.shape}, dtype: {depth_img.dtype}")
    
    # Load segmentation
    seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    print(f"    Seg shape: {seg_img.shape}, dtype: {seg_img.dtype}")
    
    # Process depth - handle multi-channel images
    if len(depth_img.shape) == 3:
        # Multi-channel depth image - take first channel or convert to grayscale
        depth = depth_img[:, :, 0].astype(np.float32)
    else:
        depth = depth_img.astype(np.float32)
    
    # Scale depth based on dtype
    if depth_img.dtype == np.uint16:
        depth = depth / 1000.0  # mm to m
    elif depth_img.dtype == np.uint8:
        depth = depth / 255.0 * 100.0  # Normalized to 100m
    
    # Process segmentation
    if len(seg_img.shape) == 3:
        seg_labels = seg_img[:, :, 0]
    else:
        seg_labels = seg_img
    
    H, W = depth.shape
    print(f"    Depth range: [{depth.min():.2f}, {depth.max():.2f}] m")
    print(f"    Seg classes: {np.unique(seg_labels)}")
    
    # Create pixel coordinates
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    # Camera parameters
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Back-project to 3D
    z = depth
    x = (u - cx) * z / focal_length
    y = (v - cy) * z / focal_length
    
    # Filter valid points
    valid_mask = (z > 0.1) & (z < 100.0)
    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)
    labels = seg_labels[valid_mask]
    
    print(f"    Valid 3D points: {len(points)}")
    
    # Voxelize
    W_vox, H_vox, D_vox = occ_size
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    
    voxel_size = np.array([
        (x_max - x_min) / W_vox,
        (y_max - y_min) / H_vox,
        (z_max - z_min) / D_vox
    ])
    
    # Convert to voxel indices
    voxel_indices = np.floor(
        (points - np.array([x_min, y_min, z_min])) / voxel_size
    ).astype(np.int32)
    
    # Filter in-bounds
    valid_mask = (
        (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < W_vox) &
        (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < H_vox) &
        (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < D_vox)
    )
    
    voxel_indices = voxel_indices[valid_mask]
    labels = labels[valid_mask]
    
    print(f"    In-bounds points: {len(voxel_indices)}")
    
    # Create voxel grid
    voxel_grid = np.zeros(occ_size, dtype=np.uint8)
    voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = labels
    
    occupied = np.sum(voxel_grid > 0)
    print(f"    Occupied voxels: {occupied} / {voxel_grid.size} ({100*occupied/voxel_grid.size:.2f}%)")
    
    return voxel_grid, points, labels


def test_dataset_simple(data_root):
    """
    Simple test of dataset functionality.
    """
    print("=" * 80)
    print("MINIMAL DATASET TEST")
    print("=" * 80)
    print(f"Data root: {data_root}\n")
    
    # Find files
    front_imgs = sorted(glob.glob(os.path.join(data_root, '*_lcam_front.png')))
    print(f"Found {len(front_imgs)} frames\n")
    
    if len(front_imgs) == 0:
        print("✗ No images found!")
        return None
    
    # Test first frame
    img_path = front_imgs[0]
    filename = os.path.basename(img_path)
    frame_id = filename.split('_')[0]
    
    print(f"Testing frame: {frame_id}")
    print(f"  Front image: {filename}")
    
    # Check all files exist
    depth_front = os.path.join(data_root, f'{frame_id}_lcam_front_depth.png')
    seg_front = os.path.join(data_root, f'{frame_id}_lcam_front_seg.png')
    back_img = os.path.join(data_root, f'{frame_id}_lcam_back.png')
    
    print(f"  Depth exists: {os.path.exists(depth_front)}")
    print(f"  Seg exists: {os.path.exists(seg_front)}")
    print(f"  Back view exists: {os.path.exists(back_img)}")
    
    # Load and check dimensions
    img = cv2.imread(img_path)
    print(f"\n  Image shape: {img.shape}")
    
    # Generate occupancy
    print("\nGenerating semantic occupancy from depth + segmentation...")
    voxel_grid, points, labels = depth_seg_to_occupancy_simple(
        depth_front, seg_front,
        focal_length=400.0,
        image_width=800,
        image_height=600,
        occ_size=(256, 256, 32),
        pc_range=(-40, -40, -1, 40, 40, 5.4)
    )
    
    print("\n✓ Occupancy generation successful!")
    print(f"  Voxel grid shape: {voxel_grid.shape}")
    print(f"  Unique labels: {np.unique(voxel_grid)}")
    
    return {
        'frame_id': frame_id,
        'img_front': img_path,
        'depth_front': depth_front,
        'seg_front': seg_front,
        'voxel_grid': voxel_grid,
        'points': points,
        'labels': labels
    }


def visualize_results(result, output_dir='/mnt/user-data/outputs/test_viz'):
    """
    Visualize test results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Load images
    img = cv2.imread(result['img_front'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    depth = cv2.imread(result['depth_front'], cv2.IMREAD_UNCHANGED)
    # Handle multi-channel depth
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    depth_vis = depth.astype(np.float32)
    if depth.dtype == np.uint16:
        depth_vis = depth_vis / 1000.0
    else:
        depth_vis = depth_vis / 255.0 * 100.0
    
    seg = cv2.imread(result['seg_front'], cv2.IMREAD_UNCHANGED)
    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    
    # Create figure with 2D data
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('RGB Image (Front)')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(depth_vis, cmap='plasma')
    axes[0, 1].set_title('Depth Map')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='Depth (m)')
    
    im2 = axes[1, 0].imshow(seg, cmap='tab20')
    axes[1, 0].set_title('Semantic Segmentation')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], label='Class ID')
    
    # Bird's eye view of occupancy
    voxel_grid = result['voxel_grid']
    bev = np.max(voxel_grid, axis=2)
    im3 = axes[1, 1].imshow(bev.T, origin='lower', cmap='tab20')
    axes[1, 1].set_title('Occupancy Bird\'s Eye View')
    axes[1, 1].set_xlabel('X (voxels)')
    axes[1, 1].set_ylabel('Y (voxels)')
    plt.colorbar(im3, ax=axes[1, 1], label='Class ID')
    
    plt.tight_layout()
    path_2d = os.path.join(output_dir, 'test_2d_data.png')
    plt.savefig(path_2d, dpi=150, bbox_inches='tight')
    print(f"✓ 2D visualization: {path_2d}")
    plt.close()
    
    # 3D visualization
    occupied_mask = voxel_grid > 0
    if np.any(occupied_mask):
        indices = np.argwhere(occupied_mask)
        occ_labels = voxel_grid[occupied_mask]
        
        # Downsample for visualization
        max_points = 10000
        if len(indices) > max_points:
            sample_idx = np.random.choice(len(indices), max_points, replace=False)
            indices = indices[sample_idx]
            occ_labels = occ_labels[sample_idx]
        
        fig = plt.figure(figsize=(15, 5))
        
        # Top view
        ax1 = fig.add_subplot(131)
        ax1.scatter(indices[:, 0], indices[:, 1], c=occ_labels, cmap='tab20', s=2, alpha=0.6)
        ax1.set_xlabel('X (voxels)')
        ax1.set_ylabel('Y (voxels)')
        ax1.set_title('Top-Down View')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Side view
        ax2 = fig.add_subplot(132)
        ax2.scatter(indices[:, 0], indices[:, 2], c=occ_labels, cmap='tab20', s=2, alpha=0.6)
        ax2.set_xlabel('X (voxels)')
        ax2.set_ylabel('Z (voxels)')
        ax2.set_title('Side View')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # 3D view
        ax3 = fig.add_subplot(133, projection='3d')
        scatter = ax3.scatter(indices[:, 0], indices[:, 1], indices[:, 2], 
                            c=occ_labels, cmap='tab20', s=1, alpha=0.5)
        ax3.set_xlabel('X (voxels)')
        ax3.set_ylabel('Y (voxels)')
        ax3.set_zlabel('Z (voxels)')
        ax3.set_title('3D Occupancy')
        plt.colorbar(scatter, ax=ax3, label='Class', shrink=0.5)
        
        plt.tight_layout()
        path_3d = os.path.join(output_dir, 'test_occupancy_3d.png')
        plt.savefig(path_3d, dpi=150, bbox_inches='tight')
        print(f"✓ 3D visualization: {path_3d}")
        plt.close()
    
    # Class statistics
    print("\nClass Statistics in Occupancy Grid:")
    unique, counts = np.unique(voxel_grid, return_counts=True)
    for cls, count in zip(unique, counts):
        if cls > 0:  # Skip empty voxels
            print(f"  Class {cls:3d}: {count:7d} voxels ({100*count/voxel_grid.size:5.2f}%)")


if __name__ == '__main__':
    # Test with uploaded data
    data_root = '/mnt/user-data/uploads'
    
    result = test_dataset_simple(data_root)
    
    if result:
        visualize_results(result)
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE!")
        print("=" * 80)
        print("✓ Dataset structure verified")
        print("✓ Depth + Segmentation → Occupancy conversion works")
        print("✓ Visualizations created")
        print("\nCheck output: /mnt/user-data/outputs/test_viz/")
import torch
from torch.utils.data import DataLoader, Dataset
import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tartanair_dataset import TartanAirDataset, read_tartanair_depth, read_tartanair_seg


def visualize_sample(sample, save_dir="./visualization"):
    """Visualize a single sample from the dataset."""
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("SAMPLE INFORMATION")
    print("=" * 60)

    # Print basic info
    print(f"Sequence: {sample.get('sequence', 'N/A')}")
    print(f"Frame ID: {sample.get('frame_id', 'N/A')}")
    print(f"Occ Size: {sample.get('occ_size', 'N/A')}")
    print(f"PC Range: {sample.get('pc_range', 'N/A')}")
    print(f"Focal Length: {sample.get('focal_length', 'N/A')}")
    print(f"Baseline: {sample.get('baseline', 'N/A')}")

    # Print paths
    img_filenames = sample.get("img_filename", [])
    print(f"\nNumber of images: {len(img_filenames)}")
    for i, path in enumerate(img_filenames):
        print(f"  Image {i}: {path}")

    # Print matrices info
    lidar2img = sample.get("lidar2img", [])
    print(f"\nNumber of lidar2img matrices: {len(lidar2img)}")

    cam_intrinsics = sample.get("cam_intrinsic", [])
    print(f"Number of cam_intrinsic matrices: {len(cam_intrinsics)}")

    lidar2cam = sample.get("lidar2cam", [])
    print(f"Number of lidar2cam matrices: {len(lidar2cam)}")

    # Print matrix shapes and verify LiDAR alignment
    if lidar2img:
        print(f"\nSample lidar2img matrix shape: {np.array(lidar2img[0]).shape}")
        print(f"lidar2img[0] (front camera):\n{np.array(lidar2img[0])}")

    if cam_intrinsics:
        print(f"\nSample cam_intrinsic matrix:\n{np.array(cam_intrinsics[0])}")

    # Verify LiDAR alignment with front camera
    if lidar2cam:
        print(f"\n--- LiDAR to Camera Transformations ---")
        camera_names = ["front", "back", "left", "right", "top", "bottom"]
        for i, l2c in enumerate(lidar2cam):
            cam_name = camera_names[i] if i < len(camera_names) else f"cam_{i}"
            l2c_arr = np.array(l2c)
            is_identity = np.allclose(l2c_arr, np.eye(4))
            print(f"\nlidar2cam[{i}] ({cam_name}):")
            print(l2c_arr)
            if is_identity:
                print("  ✓ IDENTITY - LiDAR is aligned with this camera!")

    # Check if lidar_points are available
    lidar_points = sample.get("lidar_points")
    if lidar_points is not None:
        print(f"\nLiDAR points shape: {lidar_points.shape}")
        print(f"LiDAR points range:")
        print(f"  X: [{lidar_points[:, 0].min():.2f}, {lidar_points[:, 0].max():.2f}]")
        print(f"  Y: [{lidar_points[:, 1].min():.2f}, {lidar_points[:, 1].max():.2f}]")
        print(f"  Z: [{lidar_points[:, 2].min():.2f}, {lidar_points[:, 2].max():.2f}]")
    else:
        print("\nLiDAR points: Not available in sample")

    # GT Occupancy
    gt_occ = sample.get("gt_occ")
    if gt_occ is not None:
        print(f"\nGT Occupancy shape: {gt_occ.shape}")
        print(f"GT Occupancy dtype: {gt_occ.dtype}")
        print(f"GT Occupancy unique values: {np.unique(gt_occ)[:20]}...")
        print(f"Non-zero voxels: {np.count_nonzero(gt_occ)}")
    else:
        print("\nGT Occupancy: None")

    return True


def visualize_images(data_root, save_dir="./visualization"):
    """Visualize images, depth, and segmentation from raw files."""
    os.makedirs(save_dir, exist_ok=True)

    # Find available files
    files = os.listdir(data_root)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("TartanAir Data Visualization", fontsize=16)

    # Front RGB
    front_rgb_path = os.path.join(
        data_root, "CarWelding/Data_easy/P008/image_lcam_front", "000000_lcam_front.png"
    )
    if os.path.exists(front_rgb_path):
        img = cv2.imread(front_rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Front Camera RGB")
        axes[0, 0].axis("off")

    # Back RGB
    back_rgb_path = os.path.join(
        data_root, "CarWelding/Data_easy/P008/image_lcam_back", "000000_lcam_back.png"
    )
    if os.path.exists(back_rgb_path):
        img = cv2.imread(back_rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(img)
        axes[0, 1].set_title("Back Camera RGB")
        axes[0, 1].axis("off")

    # Front Depth
    front_depth_path = os.path.join(
        data_root, "CarWelding/Data_easy/P008/depth_lcam_front", "000000_lcam_front_depth.png"
    )
    if os.path.exists(front_depth_path):
        depth = read_tartanair_depth(front_depth_path)
        # Clip for visualization
        depth_viz = np.clip(depth, 0, 50)
        im = axes[0, 2].imshow(depth_viz, cmap="viridis")
        axes[0, 2].set_title("Front Camera Depth")
        axes[0, 2].axis("off")
        plt.colorbar(im, ax=axes[0, 2], label="Depth (m)")

    # Front Segmentation
    front_seg_path = os.path.join(
        data_root, "CarWelding/Data_easy/P008/seg_lcam_front", "000000_lcam_front_seg.png"
    )
    if os.path.exists(front_seg_path):
        seg = read_tartanair_seg(front_seg_path)
        axes[1, 0].imshow(seg, cmap="tab20")
        axes[1, 0].set_title("Front Camera Segmentation")
        axes[1, 0].axis("off")

    # Back Depth
    back_depth_path = os.path.join(
        data_root, "CarWelding/Data_easy/P008/depth_lcam_back", "000000_lcam_back_depth.png"
    )
    if os.path.exists(back_depth_path):
        depth = read_tartanair_depth(back_depth_path)
        depth_viz = np.clip(depth, 0, 50)
        im = axes[1, 1].imshow(depth_viz, cmap="viridis")
        axes[1, 1].set_title("Back Camera Depth")
        axes[1, 1].axis("off")
        plt.colorbar(im, ax=axes[1, 1], label="Depth (m)")

    # Back Segmentation
    back_seg_path = os.path.join(
        data_root, "CarWelding/Data_easy/P008/seg_lcam_back", "000000_lcam_back_seg.png"
    )
    if os.path.exists(back_seg_path):
        seg = read_tartanair_seg(back_seg_path)
        axes[1, 2].imshow(seg, cmap="tab20")
        axes[1, 2].set_title("Back Camera Segmentation")
        axes[1, 2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "images_overview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved images overview to: {save_path}")
    plt.close()


def visualize_pointcloud(data_root, save_dir="./visualization"):
    """Visualize 3D point cloud from depth."""
    os.makedirs(save_dir, exist_ok=True)

    # Load depth and segmentation
    front_depth_path = os.path.join(
        data_root,
        "CarWelding/Data_easy/P008/depth_lcam_front",
        "000000_lcam_front_depth.png",
    )
    front_seg_path = os.path.join(
        data_root,
        "CarWelding/Data_easy/P008/seg_lcam_front",
        "000000_lcam_front_seg.png",
    )

    if not os.path.exists(front_depth_path):
        print("Depth file not found, skipping point cloud visualization")
        return

    depth = read_tartanair_depth(front_depth_path)
    seg = read_tartanair_seg(front_seg_path) if os.path.exists(front_seg_path) else None

    # TartanAir camera intrinsics
    fx = fy = 320.0
    cx = cy = 320.0

    height, width = depth.shape

    # Create pixel coordinate grid
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # Convert to camera coordinates
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack into point cloud
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Filter valid points
    valid_mask = (depth > 0.1) & (depth < 50.0)
    valid_mask = valid_mask.reshape(-1)
    points = points[valid_mask]

    # Get colors from segmentation or use depth
    if seg is not None:
        seg_flat = seg.reshape(-1)[valid_mask]
        colors = plt.cm.tab20(seg_flat % 20)[:, :3]
    else:
        depth_flat = depth.reshape(-1)[valid_mask]
        colors = plt.cm.viridis(depth_flat / depth_flat.max())[:, :3]

    # Subsample for visualization
    stride = 5
    points_viz = points[::stride]
    colors_viz = colors[::stride]

    # 3D plot
    fig = plt.figure(figsize=(14, 6))

    # View 1: Front view
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        points_viz[:, 0],
        points_viz[:, 2],
        -points_viz[:, 1],
        c=colors_viz,
        s=0.5,
        alpha=0.5,
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z (depth)")
    ax1.set_zlabel("-Y")
    ax1.set_title("Point Cloud - Front View")
    ax1.view_init(elev=20, azim=-90)

    # View 2: Top view
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        points_viz[:, 0],
        points_viz[:, 2],
        -points_viz[:, 1],
        c=colors_viz,
        s=0.5,
        alpha=0.5,
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z (depth)")
    ax2.set_zlabel("-Y")
    ax2.set_title("Point Cloud - Top View")
    ax2.view_init(elev=90, azim=-90)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "pointcloud.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved point cloud visualization to: {save_path}")
    plt.close()

    print(f"\nPoint Cloud Stats:")
    print(f"  Total points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")


def visualize_voxel_grid(sample, save_dir="./visualization"):
    """Visualize voxel occupancy grid."""
    os.makedirs(save_dir, exist_ok=True)

    gt_occ = sample.get("gt_occ")
    if gt_occ is None:
        print("No voxel occupancy grid available")
        return

    # Get non-zero voxels
    occupied = np.argwhere(gt_occ > 0)
    labels = gt_occ[gt_occ > 0]

    if len(occupied) == 0:
        print("Voxel grid is empty")
        return

    # Create figure
    fig = plt.figure(figsize=(14, 6))

    # View 1: 3D scatter
    ax1 = fig.add_subplot(121, projection="3d")
    colors = plt.cm.tab20(labels % 20)
    ax1.scatter(
        occupied[:, 0], occupied[:, 1], occupied[:, 2], c=colors, s=1, alpha=0.5
    )
    ax1.set_xlabel("X voxels")
    ax1.set_ylabel("Y voxels")
    ax1.set_zlabel("Z voxels")
    ax1.set_title(f"Voxel Occupancy Grid\n{len(occupied)} occupied voxels")

    # View 2: BEV (bird's eye view)
    ax2 = fig.add_subplot(122)
    bev = np.max(gt_occ, axis=2)  # Max along Z axis
    ax2.imshow(bev.T, origin="lower", cmap="tab20")
    ax2.set_xlabel("X voxels")
    ax2.set_ylabel("Y voxels")
    ax2.set_title("Bird's Eye View (Max projection)")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "voxel_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved voxel grid visualization to: {save_path}")
    plt.close()


def verify_compatibility(sample):
    """Verify that sample format is compatible with SemanticKITTI pipeline."""
    print("\n" + "=" * 60)
    print("COMPATIBILITY CHECK")
    print("=" * 60)

    required_keys = [
        "occ_size",
        "pc_range",
        "sequence",
        "frame_id",
        "img_filename",
        "lidar2img",
        "cam_intrinsic",
        "lidar2cam",
        "focal_length",
        "baseline",
        "gt_occ",
    ]

    all_present = True
    for key in required_keys:
        present = key in sample
        status = "✓" if present else "✗"
        value_info = ""

        if present:
            val = sample[key]
            if isinstance(val, np.ndarray):
                value_info = f"shape={val.shape}, dtype={val.dtype}"
            elif isinstance(val, list):
                value_info = f"len={len(val)}"
            elif isinstance(val, (int, float)):
                value_info = f"value={val}"
            else:
                value_info = f"type={type(val).__name__}"
        else:
            all_present = False

        print(f"  {status} {key}: {value_info}")

    print("\n" + "-" * 40)
    if all_present:
        print("✓ All required keys present - COMPATIBLE with SemanticKITTI pipeline")
    else:
        print("✗ Some required keys missing - check dataset implementation")

    return all_present


def load_lidar_ply(ply_path, transform_to_cam=True):
    """
    Load LiDAR point cloud from PLY file.

    TartanAir LiDAR is stored in NED frame (X-forward, Y-right, Z-down).
    If transform_to_cam=True, converts to camera frame (X-right, Y-down, Z-forward).
    """
    try:
        from plyfile import PlyData

        ply = PlyData.read(ply_path)
        vertex = ply["vertex"]
        points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T

        if transform_to_cam:
            # Transform NED to camera frame
            # cam_x = ned_y, cam_y = ned_z, cam_z = ned_x
            NED_TO_CAM = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
            points = points @ NED_TO_CAM.T

        return points
    except ImportError:
        print("plyfile not installed, skipping PLY loading")
        return None


def visualize_lidar(data_root, save_dir="./visualization"):
    """Visualize LiDAR point cloud if available."""
    os.makedirs(save_dir, exist_ok=True)

    lidar_path = os.path.join(
        data_root,
        "CarWelding/Data_easy/P008/lidar",
        "000000_lcam_front_lidar.ply",
    )
    if not os.path.exists(lidar_path):
        print("LiDAR file not found, skipping LiDAR visualization")
        return

    points = load_lidar_ply(lidar_path)
    if points is None:
        return

    print(f"\nLiDAR Point Cloud Stats:")
    print(f"  Total points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    # Visualize
    fig = plt.figure(figsize=(14, 6))

    # Compute colors based on height (Y in camera frame = down)
    y_norm = (points[:, 1] - points[:, 1].min()) / (
        points[:, 1].max() - points[:, 1].min() + 1e-6
    )
    colors = plt.cm.viridis(y_norm)

    # View 1: 3D view
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(points[:, 0], points[:, 2], -points[:, 1], c=colors, s=0.5, alpha=0.5)
    ax1.set_xlabel("X (right)")
    ax1.set_ylabel("Z (forward)")
    ax1.set_zlabel("-Y (up)")
    ax1.set_title(f"LiDAR Point Cloud (Camera Frame)\n{len(points)} points")

    # View 2: BEV (X-Z plane, looking down Y axis)
    ax2 = fig.add_subplot(122)
    ax2.scatter(points[:, 0], points[:, 2], c=colors, s=0.5, alpha=0.5)
    ax2.set_xlabel("X (right)")
    ax2.set_ylabel("Z (forward)")
    ax2.set_title("LiDAR - Bird's Eye View (X-Z plane)")
    ax2.axis("equal")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "lidar.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved LiDAR visualization to: {save_path}")
    plt.close()


def visualize_lidar_projection(data_root, save_dir="./visualization"):
    """
    Visualize LiDAR points projected onto front camera image.
    This verifies that LiDAR is aligned with lcam_front frame.
    """
    os.makedirs(save_dir, exist_ok=True)

    lidar_path = os.path.join(
        data_root,
        "CarWelding/Data_easy/P008/image_lcam_front",
        "000000_lcam_front_lidar.ply",
    )
    img_path = os.path.join(
        data_root, "CarWelding/Data_easy/P008/image_lcam_front", "000000_lcam_front.png"
    )
    depth_path = os.path.join(
        data_root,
        "CarWelding/Data_easy/P008/depth_lcam_left",
        "000000_lcam_front_depth.png",
    )

    if not os.path.exists(lidar_path) or not os.path.exists(img_path):
        print("Required files not found for projection visualization")
        return

    points = load_lidar_ply(lidar_path)
    if points is None:
        return

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # TartanAir camera intrinsics
    fx = fy = 320.0
    cx = cy = 320.0

    # Project LiDAR points to image
    # LiDAR frame = lcam_front frame, so lidar2cam = identity
    # Points are in camera frame: X-right, Y-down, Z-forward
    z = points[:, 2]
    valid = z > 0.1  # Points in front of camera

    u = (points[:, 0] * fx / z + cx).astype(np.int32)
    v = (points[:, 1] * fy / z + cy).astype(np.int32)

    # Filter points within image bounds
    h, w = img.shape[:2]
    in_bounds = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    u_valid = u[in_bounds]
    v_valid = v[in_bounds]
    z_valid = z[in_bounds]

    # Create projection visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original image with projected points
    axes[0].imshow(img)
    scatter = axes[0].scatter(u_valid, v_valid, c=z_valid, cmap="jet", s=1, alpha=0.7)
    plt.colorbar(scatter, ax=axes[0], label="Depth (m)")
    axes[0].set_title(
        "LiDAR Projected onto Front Camera\n(verifies LiDAR-Camera alignment)"
    )
    axes[0].axis("off")

    # Compare with depth image
    if os.path.exists(depth_path):
        depth = read_tartanair_depth(depth_path)
        depth_viz = np.clip(depth, 0, 50)
        im = axes[1].imshow(depth_viz, cmap="jet")
        plt.colorbar(im, ax=axes[1], label="Depth (m)")
        axes[1].set_title("Depth Image (for comparison)")
        axes[1].axis("off")

    plt.suptitle(
        "LiDAR-Camera Alignment Verification\nLiDAR frame is aligned with lcam_front frame",
        fontsize=14,
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "lidar_projection.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved LiDAR projection visualization to: {save_path}")
    plt.close()

    # Print statistics
    print(f"\nLiDAR Projection Statistics:")
    print(f"  Total LiDAR points: {len(points)}")
    print(f"  Points in front of camera: {valid.sum()}")
    print(f"  Points projected in image: {in_bounds.sum()}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and visualize TartanAir dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/tartanair",
        help="Path to TartanAir data root",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./visualization",
        help="Directory to save visualizations",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TartanAir Dataset Verification and Visualization")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Save directory: {args.save_dir}")

    # Create dataset
    print("\n" + "-" * 40)
    print("Creating TartanAirDataset...")

    dataset = TartanAirDataset(
        data_root=args.data_root,
        camera_used=["front"],
        occ_size=(256, 256, 32),
        pc_range=(-25.6, -25.6, -2.0, 25.6, 25.6, 4.4),
        test_mode=True,
    )
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) == 0:
        print("No data found in dataset. Please check data_root path.")
        # Still visualize raw files if available
        print("\nVisualizing raw files directly...")
        visualize_images(args.data_root, args.save_dir)
        visualize_pointcloud(args.data_root, args.save_dir)
        visualize_lidar(args.data_root, args.save_dir)
        return

    # Get first sample
    print("\n" + "-" * 40)
    print("Loading first sample...")
    sample = dataset[0]

    # Visualize and verify
    visualize_sample(sample, args.save_dir)
    verify_compatibility(sample)

    # Visualize images
    print("\n" + "-" * 40)
    print("Visualizing images...")
    visualize_images(args.data_root, args.save_dir)

    # Visualize point cloud
    print("\n" + "-" * 40)
    print("Visualizing point cloud...")
    visualize_pointcloud(args.data_root, args.save_dir)
    # Visualize LiDAR
    print("\n" + "-" * 40)
    print("Visualizing LiDAR...")
    visualize_lidar(args.data_root, args.save_dir)

    # Visualize LiDAR projection to verify alignment
    print("\n" + "-" * 40)
    print("Visualizing LiDAR-Camera alignment...")
    visualize_lidar_projection(args.data_root, args.save_dir)

    # Visualize voxel grid
    print("\n" + "-" * 40)
    print("Visualizing voxel grid...")
    visualize_voxel_grid(sample, args.save_dir)

    print("\n" + "=" * 60)
    print("Verification complete!")
    print(f"Visualizations saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    # data_root = "data/tartanair"
    # dataset = TartanAirDataset(
    #     data_root=data_root,
    #     camera_used="front",
    #     occ_size=(256, 256, 32),
    #     pc_range=(-25.6, -25.6, -2.0, 25.6, 25.6, 4.4),
    #     difficulties=["easy"],
    #     test_mode=True,
    # )
    # train_dataloader = DataLoader(
    #     dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    # )

    # for data in train_dataloader:
    #     # print(data)
    #     pass

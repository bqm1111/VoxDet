import os
import glob
import numpy as np
import cv2

from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import DATASETS
from plyfile import PlyData
import torch


@DATASETS.register_module()
class TartanAirDataset(Dataset):
    FOCAL_LENGTH = 320.0
    PRINCIPAL_POINT = (320.0, 320.0)
    IMAGE_SIZE = (640, 640)
    STEREO_BASELINE = 0.25

    def __init__(
        self,
        data_root,
        stereo_depth_root=None,  # Not used but kept for API compatibility
        ann_file=None,  # Path for voxel annotations
        pipeline=None,
        split="train",
        camera_used=None,  # List of cameras: ['front', 'back', 'left', 'right']
        occ_size=(256, 256, 32),
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        test_mode=False,
        load_continuous=False,
        mini_split=False,
        environments=None,  # List of environments to use
        difficulties=None,  # List of difficulties: ['easy', 'hard']
        use_lidar=True,  # Whether to use LiDAR or depth-derived points
        voxel_size=0.2,  # Voxel size in meters
    ):
        super().__init__()
        self.data_root = data_root
        self.stereo_depth_root = stereo_depth_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.load_continuous = load_continuous
        self.use_lidar = use_lidar
        self.voxel_size = voxel_size

        self.occ_size = occ_size
        self.pc_range = pc_range

        # Default camera setup - front and back for stereo-like setup
        if camera_used is None:
            camera_used = ["front", "back"]
        self.camera_used = camera_used

        # Map camera names to TartanAir naming convention
        self.camera_map = {
            "front": "lcam_front",
            "back": "lcam_back",
            "left": "lcam_left",
            "right": "lcam_right",
            "top": "lcam_top",
            "bottom": "lcam_bottom",
        }

        # Default environments and difficulties
        self.environments = ["CarWelding"]
        if difficulties is None:
            difficulties = ["easy", "hard"]
        self.difficulties = difficulties

        # Define splits (can be customized based on environment names)
        if mini_split:  # for debug usage
            self.splits = {
                "train": [
                    "P000",
                    "P001",
                    "P002",
                    "P003",
                    "P004",
                    "P005",
                    "P006",
                    "P007",
                ],
                "val": ["P008"],
                "test": ["P008"],
            }
        else:
            self.splits = {
                "train": [
                    "P000",
                    "P001",
                    "P002",
                    "P003",
                    "P004",
                    "P005",
                    "P006",
                    "P007",
                ],
                "val": ["P008"],
                "test": ["P008"],
            }
        # Build camera intrinsic matrix
        self.K = self._build_intrinsic_matrix()

        # Load annotations
        self.data_infos = self.load_annotations()

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        self._set_group_flag()

    def _build_intrinsic_matrix(self):
        """Build 4x4 camera intrinsic matrix."""
        K = np.eye(4)
        K[0, 0] = self.FOCAL_LENGTH  # fx
        K[1, 1] = self.FOCAL_LENGTH  # fy
        K[0, 2] = self.PRINCIPAL_POINT[0]  # cx
        K[1, 2] = self.PRINCIPAL_POINT[1]  # cy
        return K
    
    def __len__(self):
        return len(self.data_infos)

    def prepare_train_data(self, index):
        """Training data preparation."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print("found None in training data")
            return None

        if self.pipeline is not None:
            example = self.pipeline(input_dict)
        else:
            example = input_dict
        return example

    def prepare_test_data(self, index):
        """Test data preparation."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print("found None in test data")
            return None

        if self.pipeline is not None:
            example = self.pipeline(input_dict)
        else:
            example = input_dict
        return example

    #
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_data_info(self, index):
        """Get data info for a specific index."""
        info = self.data_infos[index]

        input_dict = dict(
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range),
            sequence=info["sequence"],
            frame_id=info["frame_id"],
        )

        # Load images, intrinsics, extrinsics
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []

        for cam_name in self.camera_used:
            cam_key = self.camera_map.get(cam_name, cam_name)
            img_path = info.get(f"img_{cam_key}_path")
            if img_path and os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                # Try alternative path construction
                img_path = info.get("img_paths", {}).get(cam_key)
                if img_path:
                    image_paths.append(img_path)
            #
            # Get pose for this camera
            pose = info.get(f"pose_{cam_key}", info.get("pose_front", np.eye(4)))

            # Compute lidar2cam (in TartanAir, lidar is aligned with left cameras)
            lidar2cam = self._compute_lidar2cam(pose, cam_name)
            lidar2cam_rts.append(lidar2cam)

            # Compute lidar2img projection matrix
            lidar2img = self.K @ lidar2cam
            lidar2img_rts.append(lidar2img)

            # Camera intrinsics (same for all cameras in TartanAir)
            cam_intrinsics.append(self.K.copy())

        # Compute baseline (for stereo depth estimation compatibility)
        baseline = self.STEREO_BASELINE
        focal_length = self.FOCAL_LENGTH

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                focal_length=focal_length,
                baseline=baseline,
            )
        )

        # Stereo depth path (for compatibility)
        input_dict["stereo_depth_path"] = info.get("depth_path", None)

        # Get voxel occupancy ground truth
        input_dict["gt_occ"] = self._get_voxel_occupancy(info)

        return input_dict

    def _compute_lidar2cam(self, cam_pose, cam_name):
        """
        Compute lidar to camera transformation.

        In TartanAir, LiDAR is sampled from depth images of the left cameras
        and is PERFECTLY ALIGNED with the left front camera (lcam_front) frame.

        Therefore:
        - lidar2cam for lcam_front = Identity (LiDAR frame = lcam_front frame)
        - For other cameras, compute relative rotation based on camera orientation

        Camera arrangement (viewing directions relative to lcam_front):
        - front: 0° (reference, looking +Z in camera frame)
        - back: 180° yaw (looking -Z relative to front)
        - left: +90° yaw (looking +X relative to front)
        - right: -90° yaw (looking -X relative to front)
        - top: -90° pitch (looking -Y relative to front, i.e., up)
        - bottom: +90° pitch (looking +Y relative to front, i.e., down)

        Args:
            cam_pose: Camera pose matrix (not used here, kept for API compatibility)
            cam_name: Camera name ('front', 'back', 'left', 'right', 'top', 'bottom')

        Returns:
            4x4 transformation matrix from LiDAR frame to camera frame
        """
        lidar2cam = np.eye(4)

        if cam_name == "front":
            # LiDAR is aligned with lcam_front - identity transformation
            pass
        elif cam_name == "back":
            # Back camera: 180° rotation around Y axis (yaw)
            # This transforms points from front camera frame to back camera frame
            lidar2cam[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
        elif cam_name == "left":
            # Left camera: 90° rotation around Y axis
            lidar2cam[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        elif cam_name == "right":
            # Right camera: -90° rotation around Y axis
            lidar2cam[:3, :3] = Rotation.from_euler("y", -90, degrees=True).as_matrix()
        elif cam_name == "top":
            # Top camera: -90° rotation around X axis (pitch up)
            lidar2cam[:3, :3] = Rotation.from_euler("x", -90, degrees=True).as_matrix()
        elif cam_name == "bottom":
            # Bottom camera: 90° rotation around X axis (pitch down)
            lidar2cam[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()

        return lidar2cam

    def _get_voxel_occupancy(self, info):
        """
        Generate voxel occupancy grid from depth and segmentation.
        Returns semantic voxel grid compatible with SemanticKITTI format.

        The voxel grid is generated in the LiDAR frame, which is aligned
        with the lcam_front camera frame.
        """
        voxel_path = info.get("voxel_path")

        # If pre-computed voxels exist, load them
        if voxel_path and os.path.exists(voxel_path):
            return np.load(voxel_path)

        # Try to use LiDAR data if available and requested
        lidar_path = info.get("lidar_path")
        if self.use_lidar and lidar_path and os.path.exists(lidar_path):
            points = self._load_lidar_ply(lidar_path)
            if points is not None:
                # LiDAR points are already in lcam_front frame
                # For semantic labels, we need to project to image and get segmentation
                seg_path = info.get("seg_path")
                if seg_path and os.path.exists(seg_path):
                    seg = self._read_segmentation(seg_path)
                    labels = self._project_points_to_seg(points, seg)
                    return self._voxelize_pointcloud(points, labels)

        # Fall back to depth-based point cloud generation
        depth_path = info.get("depth_path")
        seg_path = info.get("seg_path")

        if not depth_path or not seg_path:
            return None

        if not os.path.exists(depth_path) or not os.path.exists(seg_path):
            return None

        # Read depth and segmentation
        depth = self._read_depth(depth_path)
        seg = self._read_segmentation(seg_path)

        # Generate point cloud from depth (in lcam_front = LiDAR frame)
        points_3d, valid_mask = self._depth_to_pointcloud(depth)

        # Flatten segmentation and get labels for valid points
        seg_flat = seg.reshape(-1)
        seg_labels = seg_flat[valid_mask]

        # Voxelize the point cloud with semantic labels
        voxel_grid = self._voxelize_pointcloud(points_3d[valid_mask], seg_labels)

        return voxel_grid

    def _load_lidar_ply(self, ply_path):
        """
        Load LiDAR point cloud from PLY file and transform to camera frame.

        TartanAir LiDAR data is stored in NED frame:
        - X: forward
        - Y: right
        - Z: down

        We transform to camera frame (OpenCV convention):
        - X: right
        - Y: down
        - Z: forward

        Args:
            ply_path: Path to PLY file

        Returns:
            Nx3 numpy array of points in camera frame, or None if loading fails
        """
        try:
            ply = PlyData.read(ply_path)
            vertex = ply["vertex"]
            # Points in NED frame
            points_ned = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T

            # Transform from NED to camera frame
            # cam_x = ned_y, cam_y = ned_z, cam_z = ned_x
            points_cam = points_ned @ self.NED_TO_CAM.T

            return points_cam
        except Exception as e:
            print(f"Error loading PLY file {ply_path}: {e}")
            return None

    def _project_points_to_seg(self, points, seg):
        """
        Project 3D points to segmentation image to get semantic labels.

        Args:
            points: Nx3 array of 3D points in camera frame
            seg: HxW segmentation image

        Returns:
            N array of semantic labels
        """
        height, width = seg.shape

        # Project points to image plane
        # points are in camera frame: X-right, Y-down, Z-forward
        z = points[:, 2]
        valid = z > 0.1  # Only points in front of camera

        u = np.zeros(len(points), dtype=np.int32)
        v = np.zeros(len(points), dtype=np.int32)

        u[valid] = (
            points[valid, 0] * self.FOCAL_LENGTH / z[valid] + self.PRINCIPAL_POINT[0]
        ).astype(np.int32)
        v[valid] = (
            points[valid, 1] * self.FOCAL_LENGTH / z[valid] + self.PRINCIPAL_POINT[1]
        ).astype(np.int32)

        # Clamp to image bounds
        u = np.clip(u, 0, width - 1)
        v = np.clip(v, 0, height - 1)

        # Get labels
        labels = seg[v, u]
        labels[~valid] = 0  # Invalid points get label 0

        return labels


    @staticmethod
    def _read_depth(depth_path):
        """
        Read TartanAir depth image.
        Depth is encoded as float32 in 4-channel PNG.
        """
        depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_rgba is None:
            return None
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)

    @staticmethod
    def _read_segmentation(seg_path):
        """Read segmentation image."""
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg is None:
            return None
        if len(seg.shape) == 3:
            # Use first channel as class ID
            seg = seg[:, :, 0]
        return seg

    def _depth_to_pointcloud(self, depth):
        """
        Convert depth image to 3D point cloud in camera frame.

        The output point cloud is in the camera coordinate system:
        - X: right
        - Y: down
        - Z: forward (depth direction)

        For lcam_front, this is identical to the LiDAR frame.

        Args:
            depth: HxW depth image in meters

        Returns:
            points: Nx3 array of 3D points in camera frame
            valid_mask: N boolean array indicating valid points
        """
        height, width = depth.shape

        # Create pixel coordinate grid
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)

        # Convert to camera coordinates (X-right, Y-down, Z-forward)
        z = depth
        x = (u - self.PRINCIPAL_POINT[0]) * z / self.FOCAL_LENGTH
        y = (v - self.PRINCIPAL_POINT[1]) * z / self.FOCAL_LENGTH

        # Stack into point cloud (N, 3)
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Filter invalid points (depth too far or too close)
        valid_mask = (depth > 0.1) & (depth < 100.0)
        valid_mask = valid_mask.reshape(-1)

        return points, valid_mask

    def _voxelize_pointcloud(self, points, labels):
        """
        Voxelize point cloud with semantic labels.

        The input points are expected to be in the LiDAR frame (= lcam_front frame):
        - X: right
        - Y: down
        - Z: forward

        The voxel grid uses the pc_range to define the 3D bounding box.

        Args:
            points: Nx3 array of 3D points in LiDAR/camera frame
            labels: N array of semantic labels

        Returns:
            voxel_grid: 3D array of shape (occ_size) with semantic labels
        """
        occ_size = np.array(self.occ_size)
        pc_range = np.array(self.pc_range)

        # Create empty voxel grid
        voxel_grid = np.zeros(occ_size, dtype=np.uint8)

        if len(points) == 0:
            return voxel_grid

        # Points are already in LiDAR frame (= lcam_front frame)
        points_transformed = points.copy()

        # Filter points within range
        mask = (
            (points_transformed[:, 0] >= pc_range[0])
            & (points_transformed[:, 0] < pc_range[3])
            & (points_transformed[:, 1] >= pc_range[1])
            & (points_transformed[:, 1] < pc_range[4])
            & (points_transformed[:, 2] >= pc_range[2])
            & (points_transformed[:, 2] < pc_range[5])
        )

        points_filtered = points_transformed[mask]
        labels_filtered = labels[mask]

        if len(points_filtered) == 0:
            return voxel_grid

        # Compute voxel indices
        voxel_size = (pc_range[3:6] - pc_range[0:3]) / occ_size
        voxel_indices = ((points_filtered - pc_range[0:3]) / voxel_size).astype(
            np.int32
        )

        # Clamp to valid range
        voxel_indices = np.clip(voxel_indices, 0, occ_size - 1)

        # Assign labels to voxels (last write wins, or could use voting)
        for i, (idx, label) in enumerate(zip(voxel_indices, labels_filtered)):
            voxel_grid[idx[0], idx[1], idx[2]] = label

        return voxel_grid

    def load_annotations(self):
        """Load all annotations from data_root."""
        scans = []

        for env in self.environments:
            env_path = os.path.join(self.data_root, env)

            for difficulty in self.difficulties:
                diff_path = os.path.join(env_path, "Data_" + difficulty)
                if not os.path.exists(diff_path):
                    continue

                # Find all trajectories (P000, P001, etc.)
                if self.test_mode:
                    trajectories = self.splits["test"]
                else:
                    trajectories = self.splits["train"]
                for traj_path in trajectories:
                    traj_id = os.path.basename(traj_path)
                    sequence = f"{env}_{difficulty}_{traj_id}"
                    traj_path = os.path.join(diff_path, traj_path)

                    # Load poses for each camera
                    poses = self._load_poses(traj_path)

                    # Find all frames. This is used to find all the frame_id. Thats why we will traverse only the image_lcam_front folder
                    frame_pattern = os.path.join(traj_path, "image_lcam_front", "*.png")
                    frame_files = sorted(glob.glob(frame_pattern))

                    for frame_file in frame_files:
                        frame_id = os.path.basename(frame_file).replace(".png", "").split("_")[0]

                        # Build paths for all modalities
                        scan_info = {
                            "sequence": sequence,
                            "frame_id": frame_id,
                            "traj_path": traj_path,
                        }

                        # Add image paths for each camera
                        for cam_name, cam_key in self.camera_map.items():
                            if not cam_name in self.camera_used:
                                continue

                            img_path = os.path.join(
                                traj_path, f"image_{cam_key}", f"{frame_id}.png"
                            )
                            if os.path.exists(img_path):
                                scan_info[f"img_{cam_key}_path"] = img_path

                            depth_path = os.path.join(
                                traj_path, f"depth_{cam_key}", f"{frame_id}_{cam_key}_depth.png"
                            )
                            if os.path.exists(depth_path):
                                scan_info[f"depth_{cam_key}_path"] = depth_path

                            seg_path = os.path.join(
                                traj_path, f"seg_{cam_key}", f"{frame_id}_{cam_key}_seg.png"
                            )
                            if os.path.exists(seg_path):
                                scan_info[f"seg_{cam_key}_path"] = seg_path


                        # Use front camera for main depth/seg
                        scan_info["depth_path"] = scan_info.get("depth_lcam_front_path")
                        scan_info["seg_path"] = scan_info.get("seg_lcam_front_path")
                        print(scan_info)

                        # Add lidar path if available
                        lidar_path = os.path.join(traj_path, "lidar", f"{frame_id}_{cam_key}_lidar.ply")
                        if os.path.exists(lidar_path):
                            scan_info["lidar_path"] = lidar_path

                        # Add poses
                        frame_idx = int(frame_id)
                        for cam_name, cam_key in self.camera_map.items():
                            pose_key = f"pose_{cam_key}"
                            if pose_key in poses and frame_idx < len(poses[pose_key]):
                                scan_info[f"pose_{cam_key}"] = poses[pose_key][
                                    frame_idx
                                ]

                        # Voxel annotation path (if pre-computed)
                        if self.ann_file:
                            voxel_path = os.path.join(
                                self.ann_file, sequence, f"{frame_id}.npy"
                            )
                            if os.path.exists(voxel_path):
                                scan_info["voxel_path"] = voxel_path

                        scans.append(scan_info)

        return scans
    
    def _load_poses(self, traj_path):
        """Load camera poses from trajectory folder."""
        poses = {}

        for cam_name, cam_key in self.camera_map.items():
            if cam_name in self.camera_used:
                pose_file = os.path.join(traj_path, f"pose_{cam_key}.txt")
                if not os.path.exists(pose_file):

                    print("Pose file does not exist")
                    continue

                pose_data = np.loadtxt(pose_file)
                if pose_data.ndim == 1:
                    pose_data = pose_data.reshape(1, -1)

                pose_matrices = []
                for row in pose_data:
                    pose_mat = self._pose_to_matrix(row)
                    pose_matrices.append(pose_mat)

                poses[f"pose_{cam_key}"] = pose_matrices

        return poses

    @staticmethod
    def _pose_to_matrix(pose_row):
        """
        Convert pose row (tx, ty, tz, qx, qy, qz, qw) to 4x4 matrix.
        TartanAir uses NED frame.
        """
        tx, ty, tz = pose_row[0:3]
        qx, qy, qz, qw = pose_row[3:7]

        # Create rotation matrix from quaternion
        rot = Rotation.from_quat([qx, qy, qz, qw])

        # Build 4x4 transformation matrix
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = rot.as_matrix()
        pose_mat[:3, 3] = [tx, ty, tz]

        return pose_mat

    def get_ann_info(self, index, key="voxel_path"):
        """Get annotation info."""
        info = self.data_infos[index].get(key)
        if info is None:
            return None
        if isinstance(info, str) and os.path.exists(info):
            return np.load(info)
        return info

    def _rand_another(self, idx):
        """Randomly get another item."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        """Set flag for all samples."""
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def dynamic_baseline(self, infos):
        """Return stereo baseline (fixed for TartanAir)."""
        return self.STEREO_BASELINE


# Convenience function to read TartanAir depth
def read_tartanair_depth(depth_path):
    """Read and decode TartanAir depth image."""
    return TartanAirDataset._read_depth(depth_path)


# Convenience function to read TartanAir segmentation
def read_tartanair_seg(seg_path):
    """Read TartanAir segmentation image."""
    return TartanAirDataset._read_segmentation(seg_path)

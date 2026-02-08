import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from mmdet.datasets.builder import PIPELINES


def read_tartanair_depth(depth_path):
    """
    Read TartanAir depth image (float32 encoded as RGBA).
    
    Args:
        depth_path: Path to depth PNG file
        
    Returns:
        Depth array (H, W) in meters
    """
    depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_rgba is None:
        raise ValueError(f"Could not load depth: {depth_path}")
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_TartanAir(object):
    """
    Load TartanAir images and prepare for model input.
    
    Matches the output format of LoadMultiViewImageFromFiles_SemanticKitti.
    
    Output format of img_inputs tuple:
        (imgs, rots, trans, intrins, post_rots, post_trans, depth, cam2lidar, focal_length, baseline)
    
    Args:
        data_config: Dictionary with 'input_size', 'resize', 'rot', 'flip', 'crop_h'
        is_train: Whether in training mode
        img_norm_cfg: Image normalization config (unused, uses ImageNet defaults)
        load_depth: Whether to load TartanAir dense depth
        color_jitter: Color jitter parameters (brightness, contrast, saturation)
    """
    
    def __init__(
        self,
        data_config,
        is_train=False,
        img_norm_cfg=None,
        load_depth=True,
        color_jitter=(0.4, 0.4, 0.4)
    ):
        super().__init__()
        
        self.is_train = is_train
        self.data_config = data_config
        self.img_norm_cfg = img_norm_cfg
        self.load_depth = load_depth
        
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        
        self.normalize_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.ToTensor = transforms.ToTensor()
    
    def get_rot(self, h):
        """Get 2D rotation matrix."""
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])
    
    def sample_augmentation(self, H, W, flip=None, scale=None):
        """Sample augmentation parameters."""
        fH, fW = self.data_config['input_size']
        
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        
        return resize, resize_dims, crop, flip, rotate
    
    def img_transform(self, img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate):
        """Apply image transformation and update post-processing matrices."""
        # Transform image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)
        
        # Update post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        
        return img, post_rot, post_tran
    
    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        """Core image transformation (resize, crop, flip, rotate)."""
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img
    
    def get_inputs(self, results, flip=None, scale=None):
        """
        Process images and prepare model inputs.
        
        Returns:
            result_list: List of tensors [imgs, rots, trans, intrins, post_rots, post_trans, depth, cam2lidar, focal_length, baseline]
        """
        img_filenames = results['img_filename']
        focal_length = results['focal_length']
        baseline = results['baseline']
        
        data_lists = []
        raw_img_list = []
        img_augs = None
        
        for i in range(len(img_filenames)):
            img_filename = img_filenames[i]
            img = Image.open(img_filename).convert('RGB')
            
            # Perform image-view augmentation
            post_rot = torch.eye(2)
            post_trans = torch.zeros(2)
            
            if i == 0:
                img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip, scale=scale)
            
            resize, resize_dims, crop, flip_aug, rotate = img_augs
            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_trans, resize=resize,
                resize_dims=resize_dims, crop=crop, flip=flip_aug, rotate=rotate
            )
            
            # Make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            
            # Intrinsics
            intrin = torch.Tensor(results['cam_intrinsic'][i])
            
            # Extrinsics: lidar2cam and cam2lidar
            lidar2cam = torch.Tensor(results['lidar2cam'][i])
            cam2lidar = lidar2cam.inverse()
            rot = cam2lidar[:3, :3]
            tran = cam2lidar[:3, 3]
            
            # Store raw image for visualization
            canvas = np.array(img)
            
            # Apply color jitter for training
            if self.color_jitter and self.is_train:
                img = self.color_jitter(img)
            
            # Normalize image
            img = self.normalize_img(img)
            
            # Placeholder depth (will be replaced by CreateDepthFromTartanAir)
            depth = torch.zeros(1)
            
            result = [img, rot, tran, intrin, post_rot, post_tran, depth, cam2lidar]
            result = [x[None] for x in result]
            
            data_lists.append(result)
            raw_img_list.append(canvas)
        
        # Load TartanAir depth if available
        if self.load_depth and 'depth_path' in results and results['depth_path'] is not None:
            depth_path = results['depth_path']
            if depth_path and isinstance(depth_path, str):
                try:
                    depth_raw = read_tartanair_depth(depth_path)
                    depth_img = Image.fromarray(depth_raw)
                    resize, resize_dims, crop, flip_aug, rotate = img_augs
                    depth_img = self.img_transform_core(
                        depth_img, resize_dims=resize_dims,
                        crop=crop, flip=flip_aug, rotate=rotate
                    )
                    results['stereo_depth'] = self.ToTensor(np.array(depth_img))
                except Exception as e:
                    print(f"Warning: Could not load depth from {depth_path}: {e}")
                    results['stereo_depth'] = torch.zeros(1, *self.data_config['input_size'])
            else:
                results['stereo_depth'] = torch.zeros(1, *self.data_config['input_size'])
        elif 'stereo_depth_path' in results and results['stereo_depth_path'] is not None:
            # Fallback to stereo_depth_path (for compatibility)
            depth_path = results['stereo_depth_path']
            if depth_path and isinstance(depth_path, str):
                try:
                    depth_raw = read_tartanair_depth(depth_path)
                    depth_img = Image.fromarray(depth_raw)
                    resize, resize_dims, crop, flip_aug, rotate = img_augs
                    depth_img = self.img_transform_core(
                        depth_img, resize_dims=resize_dims,
                        crop=crop, flip=flip_aug, rotate=rotate
                    )
                    results['stereo_depth'] = self.ToTensor(np.array(depth_img))
                except Exception as e:
                    print(f"Warning: Could not load depth from {depth_path}: {e}")
                    results['stereo_depth'] = torch.zeros(1, *self.data_config['input_size'])
            else:
                results['stereo_depth'] = torch.zeros(1, *self.data_config['input_size'])
        else:
            results['stereo_depth'] = torch.zeros(1, *self.data_config['input_size'])
        
        # Concatenate results from all cameras
        num = len(data_lists[0])
        result_list = []
        for i in range(num):
            result_list.append(torch.cat([x[i] for x in data_lists], dim=0))
        
        result_list.append(torch.tensor(focal_length, dtype=torch.float32))
        result_list.append(torch.tensor(baseline, dtype=torch.float32))
        results['raw_img'] = raw_img_list
        
        return result_list
    
    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class CreateDepthFromTartanAir(object):
    """
    Create depth maps from TartanAir dense depth images.
    
    Unlike KITTI which projects LiDAR points to create sparse depth,
    TartanAir has dense depth images that can be used directly.
    
    Args:
        data_root: Root directory of TartanAir dataset
    """
    
    def __init__(self, data_root=None):
        self.data_root = data_root
    
    def __call__(self, results):
        imgs = results['img_inputs'][0]  # (N, C, H, W)
        img_h, img_w = imgs.shape[-2:]
        num_cams = imgs.shape[0]
        
        gt_depths = []
        
        for img_index in range(num_cams):
            # TartanAir depth is already loaded as stereo_depth
            if 'stereo_depth' in results and results['stereo_depth'] is not None:
                stereo_depth = results['stereo_depth']
                if stereo_depth.dim() == 3:
                    gt_depth = stereo_depth[0]  # Take first channel
                else:
                    gt_depth = stereo_depth
                
                # Resize if needed
                if gt_depth.shape[-2:] != (img_h, img_w):
                    gt_depth = torch.nn.functional.interpolate(
                        gt_depth.unsqueeze(0).unsqueeze(0),
                        size=(img_h, img_w),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
            else:
                gt_depth = torch.zeros((img_h, img_w))
            
            gt_depths.append(gt_depth)
        
        gt_depths = torch.stack(gt_depths)
        results['gt_depths'] = gt_depths
        
        return results


@PIPELINES.register_module()
class LoadTartanAirAnnotation(object):
    """
    Load TartanAir voxel occupancy annotations.
    
    Matches the output format of LoadAnnotationOcc for SemanticKITTI.
    
    Args:
        bda_aug_conf: BEV data augmentation config
        is_train: Whether in training mode
        apply_bda: Whether to apply BEV augmentation
        point_cloud_range: Point cloud range for transformation center
    """
    
    def __init__(
        self,
        bda_aug_conf,
        is_train=True,
        apply_bda=False,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]
    ):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.apply_bda = apply_bda
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.transform_center = (self.point_cloud_range[:3] + self.point_cloud_range[3:]) / 2
    
    def sample_bda_augmentation(self):
        """Generate BDA augmentation values."""
        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        flip_dz = np.random.uniform() < self.bda_aug_conf['flip_dz_ratio']
        return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz
    
    def forward_test(self, results):
        """Process for test mode (no augmentation, no gt_occ)."""
        bda_rot = torch.eye(4).float()
        
        # Unpack img_inputs
        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
        cam2lidar = results['img_inputs'][7]
        
        results['img_inputs'] = (
            imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, cam2lidar
        )
        results['img_shape'] = imgs.shape[-2:]
        
        return results
    
    def __call__(self, results):
        if results['gt_occ'] is None:
            return self.forward_test(results)
        
        # Convert gt_occ to tensor
        if isinstance(results['gt_occ'], list):
            gt_occ = [torch.tensor(x) for x in results['gt_occ']]
        else:
            gt_occ = torch.tensor(results['gt_occ'])
        
        # Apply BDA if configured
        if self.is_train and self.apply_bda:
            rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz = self.sample_bda_augmentation()
            gt_occ, bda_rot = voxel_transform(
                gt_occ, rotate_bda, scale_bda,
                flip_dx, flip_dy, flip_dz, self.transform_center
            )
        else:
            bda_rot = torch.eye(4).float()
        
        # Unpack and repack img_inputs
        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
        cam2lidar = results['img_inputs'][7]
        
        results['img_inputs'] = (
            imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, cam2lidar
        )
        results['img_shape'] = imgs.shape[-2:]
        results['gt_occ'] = gt_occ.long()
        
        return results


def voxel_transform(voxel_labels, rotate_angle, scale_ratio, flip_dx, flip_dy, flip_dz, transform_center=None):
    """
    Apply transformation to voxel labels.
    
    Args:
        voxel_labels: 3D voxel label tensor
        rotate_angle: Rotation angle in degrees
        scale_ratio: Scale ratio (unused for voxels)
        flip_dx: Whether to flip along X axis
        flip_dy: Whether to flip along Y axis
        flip_dz: Whether to flip along Z axis
        transform_center: Center point for transformation
        
    Returns:
        Transformed voxel labels and BDA rotation matrix
    """
    assert transform_center is not None
    
    trans_norm = torch.eye(4)
    trans_norm[:3, -1] = -transform_center
    trans_denorm = torch.eye(4)
    trans_denorm[:3, -1] = transform_center
    
    # Bird's-eye-view rotation
    rotate_degree = rotate_angle
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([
        [rot_cos, -rot_sin, 0, 0],
        [rot_sin, rot_cos, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Flip matrices
    flip_mat = torch.eye(4)
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    if flip_dz:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
    
    # Combined transformation matrix
    bda_mat = trans_denorm @ flip_mat @ rot_mat @ trans_norm
    
    # Transform voxel labels
    voxel_labels = voxel_labels.numpy().astype(np.uint8)
    
    if not np.isclose(rotate_degree, 0):
        voxel_labels = custom_rotate_3d(voxel_labels, rotate_degree)
    
    if flip_dz:
        voxel_labels = voxel_labels[:, :, ::-1]
    
    if flip_dy:
        voxel_labels = voxel_labels[:, ::-1]
    
    if flip_dx:
        voxel_labels = voxel_labels[::-1]
    
    voxel_labels = torch.from_numpy(voxel_labels.copy()).long()
    
    return voxel_labels, bda_mat


def custom_rotate_3d(voxel_labels, rotate_degree):
    """Rotate 3D voxel labels around Z axis."""
    is_tensor = False
    if isinstance(voxel_labels, torch.Tensor):
        is_tensor = True
        voxel_labels = voxel_labels.numpy().astype(np.uint8)
    
    voxel_labels_list = []
    for height_index in range(voxel_labels.shape[-1]):
        bev_labels = voxel_labels[..., height_index]
        bev_labels = Image.fromarray(bev_labels.astype(np.uint8))
        bev_labels = bev_labels.rotate(rotate_degree, resample=Image.Resampling.NEAREST, fillcolor=255)
        bev_labels = np.array(bev_labels)
        voxel_labels_list.append(bev_labels)
    
    voxel_labels = np.stack(voxel_labels_list, axis=-1)
    
    if is_tensor:
        voxel_labels = torch.from_numpy(voxel_labels).long()
    
    return voxel_labels
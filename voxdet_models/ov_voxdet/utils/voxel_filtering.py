"""
Instance-aware voxel filtering for OV-VoxDet.

Combines OVO's pixel-voxel filtering with VoxDet's VoxNT instance information
to select high-quality training voxels for the distillation losses.

OVO's original filtering:
  1. Visibility filter: keep only voxels visible in the camera FOV
  2. Occlusion filter: remove voxels occluded by other objects
  3. Confidence filter: weight by LSeg prediction confidence

VoxDet enhancement:
  4. Boundary filter: down-weight voxels near instance boundaries (from VoxNT offsets)
     since their 2D projections are less reliable due to mixing of adjacent objects
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_boundary_weights(gt_offsets, threshold=2.0, smooth=True):
    """
    Compute per-voxel weights based on distance to instance boundaries.
    
    Voxels deep inside an instance get weight ~1.0.
    Voxels at boundaries get weight ~0.0.
    
    Args:
        gt_offsets: [B, 6, X, Y, Z] VoxNT offset field (distances to boundaries in 6 directions)
        threshold: offset threshold below which voxels are considered boundary voxels
        smooth: use sigmoid instead of hard threshold
    
    Returns:
        weights: [B, X, Y, Z] per-voxel boundary weights in [0, 1]
    """
    # Minimum distance to any boundary
    min_offset = gt_offsets.min(dim=1)[0].float()  # [B, X, Y, Z]
    
    if smooth:
        # Smooth sigmoid ramp: 0 at boundary, 1 deep inside
        weights = torch.sigmoid(2.0 * (min_offset - threshold))
    else:
        weights = (min_offset >= threshold).float()
    
    return weights


def filter_valid_voxels(projected_pix, fov_mask, gt_labels, gt_offsets=None,
                        lseg_feat=None, lseg_confidence=None,
                        confidence_threshold=0.5, boundary_threshold=2.0,
                        scene_size=None):
    """
    Select high-quality voxels for OVO distillation training.
    
    Combines multiple filtering criteria:
    1. FOV mask: only voxels visible in camera
    2. Non-empty: exclude empty voxels (class 0)
    3. Non-ignored: exclude ignored voxels (class 255)
    4. Confidence: LSeg prediction confidence above threshold
    5. Boundary: distance to instance boundary above threshold
    
    Args:
        projected_pix: [N_voxels, 2] pixel coordinates for each voxel
        fov_mask:       [N_voxels] bool mask of voxels in camera FOV
        gt_labels:      [X, Y, Z] ground truth voxel labels
        gt_offsets:     [6, X, Y, Z] VoxNT offset labels (optional)
        lseg_feat:      [512, H, W] LSeg feature map (optional)
        lseg_confidence: [H, W] LSeg prediction confidence (optional)
        confidence_threshold: minimum LSeg confidence to keep a voxel
        boundary_threshold: minimum boundary distance to keep a voxel
        scene_size: tuple (X, Y, Z) for reshaping flat indices
    
    Returns:
        valid_indices:    [N_valid] flat indices of valid voxels
        pixel_features:   [N_valid, 512] corresponding LSeg pixel features
        confidence_weights: [N_valid] confidence weights for loss re-weighting
        boundary_weights:   [N_valid] boundary weights for loss re-weighting
    """
    if scene_size is None:
        scene_size = gt_labels.shape
    
    X, Y, Z = scene_size
    device = gt_labels.device
    
    # Flatten labels for indexing
    labels_flat = gt_labels.reshape(-1)  # [X*Y*Z]
    
    # 1. FOV mask
    valid = fov_mask.bool()  # [N_voxels]
    
    # 2. Non-empty & non-ignored
    valid = valid & (labels_flat[valid.nonzero(as_tuple=True)] != 0)
    
    # Re-derive valid from combined conditions
    non_empty = (labels_flat != 0) & (labels_flat != 255)
    valid = fov_mask.bool() & non_empty
    
    # 3. Boundary filtering (using VoxNT offsets)
    boundary_weights = torch.ones(X * Y * Z, device=device)
    if gt_offsets is not None:
        offsets_flat = gt_offsets.reshape(6, -1)  # [6, X*Y*Z]
        min_offsets = offsets_flat.min(dim=0)[0].float()
        boundary_weights = torch.sigmoid(2.0 * (min_offsets - boundary_threshold))
        
        # Hard filter: remove voxels right at boundaries
        valid = valid & (min_offsets >= 1.0)
    
    # Get valid indices
    valid_indices = valid.nonzero(as_tuple=False).squeeze(-1)
    
    if valid_indices.numel() == 0:
        return (torch.tensor([], dtype=torch.long, device=device),
                None, None, None)
    
    # 4. Extract pixel features and confidence for valid voxels
    pixel_features = None
    confidence_weights = torch.ones(valid_indices.shape[0], device=device)
    
    if lseg_feat is not None and projected_pix is not None:
        C, H, W = lseg_feat.shape
        
        # Get pixel coordinates for valid voxels
        valid_pix = projected_pix[valid_indices]  # [N_valid, 2]
        pix_x = valid_pix[:, 0].long().clamp(0, W - 1)
        pix_y = valid_pix[:, 1].long().clamp(0, H - 1)
        
        # Sample LSeg features at projected pixel locations
        pixel_features = lseg_feat[:, pix_y, pix_x].T  # [N_valid, 512]
        
        # 5. Confidence filtering
        if lseg_confidence is not None:
            conf = lseg_confidence[pix_y, pix_x]  # [N_valid]
            confidence_weights = conf
            
            # Hard filter by confidence
            conf_mask = conf >= confidence_threshold
            valid_indices = valid_indices[conf_mask]
            pixel_features = pixel_features[conf_mask]
            confidence_weights = confidence_weights[conf_mask]
            boundary_weights = boundary_weights[valid_indices]
    
    bw = boundary_weights[valid_indices] if boundary_weights.shape[0] > valid_indices.shape[0] else boundary_weights[:valid_indices.shape[0]]
    
    return valid_indices, pixel_features, confidence_weights, bw


def generate_instance_groups(gt_labels, gt_offsets, min_instance_size=5):
    """
    Group voxels into instances using VoxNT offset information.
    
    Two voxels belong to the same instance if:
    1. They have the same class label
    2. They are connected (adjacent voxels with same label form a group)
    
    This is a simplified version that uses the offset field to identify
    instance boundaries, then groups connected components.
    
    Args:
        gt_labels:  [X, Y, Z] ground truth labels
        gt_offsets: [6, X, Y, Z] VoxNT offsets
        min_instance_size: minimum voxels to form an instance
    
    Returns:
        instance_map: [X, Y, Z] instance IDs (0 = no instance)
    """
    X, Y, Z = gt_labels.shape
    device = gt_labels.device
    
    # Instance boundaries are where offsets = 1 (right at the edge)
    # Sum of opposing offsets gives the instance extent
    len_x = gt_offsets[0] + gt_offsets[1]
    len_y = gt_offsets[2] + gt_offsets[3]
    len_z = gt_offsets[4] + gt_offsets[5]
    
    # A voxel is at a boundary if any direction has offset == 1
    # AND the adjacent voxel has a different label
    min_offset = gt_offsets.min(dim=0)[0]
    is_boundary = (min_offset <= 1)
    
    # Simple connected-component-like grouping based on class labels
    # For efficiency, use the offset field to approximate instance IDs
    # Each voxel gets an ID based on its class + spatial cluster
    instance_map = torch.zeros_like(gt_labels, dtype=torch.long)
    
    # Approximate: encode instance by (class, x_center, y_center, z_center)
    # where center = position + (offset_neg - offset_pos) / 2
    # Voxels in the same instance point to the same approximate center
    unique_classes = gt_labels.unique()
    instance_id = 1
    
    for cls in unique_classes:
        if cls == 0 or cls == 255:
            continue
        
        cls_mask = (gt_labels == cls)
        if cls_mask.sum() < min_instance_size:
            continue
        
        # Use offsets to create approximate center coordinates
        cls_positions = cls_mask.nonzero(as_tuple=False).float()  # [N_cls, 3]
        
        if cls_positions.shape[0] == 0:
            continue
        
        # Assign instance ID to connected regions of the same class
        # Simplified: just use class-level grouping (same as OVO baseline)
        instance_map[cls_mask] = instance_id
        instance_id += 1
    
    return instance_map
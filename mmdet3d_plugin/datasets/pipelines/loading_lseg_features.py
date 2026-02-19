"""
Pipeline transform to load pre-extracted LSeg features for OV-VoxDet training.

Pre-extracted data is stored as .npz files at:
    {lseg_feat_root}/sequences/{sequence}/{frame_id}.npz

Each .npz file contains:
    - lseg_2d_feat:     [512, H, W]   LSeg per-pixel feature map
    - lseg_pixel_feat:  [N_valid, 512] LSeg features at valid voxel projections
    - lseg_confidence:  [N_valid]      LSeg prediction confidence per valid voxel
    - valid_vox_indices:[N_valid]      flat indices of valid voxels (after filtering)

These are generated offline by utils/preprocess_ov_data.py using:
    1. LSeg model inference on each image
    2. Voxel-to-pixel projection using camera calibration
    3. OVO-style pixel-voxel filtering (visibility + occlusion + confidence)

See utils/preprocess_ov_data.py for the generation script.
"""

import os
import numpy as np
import torch
from mmdet.registry import TRANSFORMS as PIPELINES



@PIPELINES.register_module()
class LoadLSegFeatures():
    """Load pre-extracted LSeg features and voxel-pixel correspondences.
    
    This transform loads the OVO distillation targets that were pre-computed
    offline. It inserts them into the results dict so they flow through
    to CollectData and into the training batch.
    
    Args:
        lseg_feat_root (str): Root directory containing LSeg feature .npz files.
            Expected layout: {lseg_feat_root}/sequences/{sequence}/{frame_id}.npz
        load_2d_feat (bool): Whether to load the full 2D feature map (for L_2d loss).
            Set False to save memory if not using the 2D alignment loss.
    """
    
    def __init__(self, lseg_feat_root, load_2d_feat=True):
        self.lseg_feat_root = lseg_feat_root
        self.load_2d_feat = load_2d_feat
    
    def __call__(self, results):
        """Load LSeg features for the current frame.
        
        Args:
            results (dict): Result dict from previous pipeline steps.
                Must contain 'sequence' and 'frame_id' keys.
        
        Returns:
            dict: Updated results with LSeg feature keys added.
        """
        sequence = results['sequence']
        frame_id = results['frame_id']
        
        feat_path = os.path.join(
            self.lseg_feat_root, 'sequences', sequence, f'{frame_id}.npz'
        )
        
        if os.path.exists(feat_path):
            data = np.load(feat_path, allow_pickle=True)
            
            # Full 2D LSeg feature map for L_2d loss
            if self.load_2d_feat and 'lseg_2d_feat' in data:
                results['lseg_2d_feat'] = torch.from_numpy(
                    data['lseg_2d_feat'].astype(np.float32)
                )
            else:
                results['lseg_2d_feat'] = None
            
            # Per-valid-voxel LSeg pixel features for L_vox_pix loss
            if 'lseg_pixel_feat' in data:
                results['lseg_pixel_feat'] = torch.from_numpy(
                    data['lseg_pixel_feat'].astype(np.float32)
                )
            else:
                results['lseg_pixel_feat'] = None
            
            # LSeg confidence scores for confidence-weighted loss
            if 'lseg_confidence' in data:
                results['lseg_confidence'] = torch.from_numpy(
                    data['lseg_confidence'].astype(np.float32)
                )
            else:
                results['lseg_confidence'] = None
            
            # Flat indices of valid voxels (after pixel-voxel filtering)
            if 'valid_vox_indices' in data:
                results['valid_vox_indices'] = torch.from_numpy(
                    data['valid_vox_indices'].astype(np.int64)
                )
            else:
                results['valid_vox_indices'] = None
        
        else:
            # Feature file not found — set all to None so training can
            # gracefully skip distillation losses for this sample
            results['lseg_2d_feat'] = None
            results['lseg_pixel_feat'] = None
            results['lseg_confidence'] = None
            results['valid_vox_indices'] = None
        
        return results
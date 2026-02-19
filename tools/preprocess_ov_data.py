"""
Data preprocessing for OV-VoxDet.

This script handles the generation of pseudo-labels from LSeg for training:
1. Extract LSeg pixel embeddings for each image
2. Project 3D voxels to 2D to establish voxel-pixel correspondences
3. Apply OVO-style pixel-voxel filtering
4. Compute VoxNT offsets and boundary weights
5. Save preprocessed data for efficient training

Usage:
    python preprocess_ov_data.py \
        --kitti_root /path/to/semantic_kitti \
        --lseg_model_path /path/to/lseg_model \
        --output_dir /path/to/ov_preprocess \
        --clip_model ViT-B/32
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def extract_lseg_features(image, lseg_model, target_size=None):
    """
    Extract LSeg pixel-level features from an image.
    
    Args:
        image: [3, H, W] input image tensor
        lseg_model: pre-trained LSeg model
        target_size: optional (H, W) to resize features
    
    Returns:
        features: [512, H', W'] per-pixel LSeg features
        confidence: [H', W'] prediction confidence per pixel
    """
    with torch.no_grad():
        # LSeg outputs per-pixel embeddings aligned with CLIP text space
        features = lseg_model.forward_features(image.unsqueeze(0))  # [1, 512, H', W']
        features = features.squeeze(0)  # [512, H', W']
        
        if target_size is not None:
            features = F.interpolate(
                features.unsqueeze(0), size=target_size,
                mode='bilinear', align_corners=True
            ).squeeze(0)
        
        # Confidence = norm of feature vector (higher = more certain)
        confidence = features.norm(dim=0)  # [H', W']
        confidence = confidence / confidence.max()  # normalize to [0, 1]
        
        # Normalize features
        features = F.normalize(features, dim=0)
    
    return features, confidence


def extract_clip_text_embeddings(class_names, clip_model, tokenizer, 
                                  prompt_template="a photo of a {}"):
    """
    Generate CLIP text embeddings for class names.
    
    Args:
        class_names: list of class name strings
        clip_model: pre-trained CLIP model
        tokenizer: CLIP tokenizer
        prompt_template: text prompt template
    
    Returns:
        embeddings: [num_classes, 512] normalized text embeddings
    """
    embeddings = []
    with torch.no_grad():
        for name in class_names:
            text = prompt_template.format(name)
            tokens = tokenizer(text)
            emb = clip_model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
            embeddings.append(emb.squeeze(0))
    
    return torch.stack(embeddings)  # [num_classes, 512]


def project_voxels_to_pixels(voxel_coords, calib, img_shape):
    """
    Project 3D voxel centers to 2D pixel coordinates.
    
    Args:
        voxel_coords: [N, 3] voxel center coordinates in camera frame
        calib: camera calibration matrix [3, 4] or [4, 4]
        img_shape: (H, W) image dimensions
    
    Returns:
        pixel_coords: [N, 2] projected pixel coordinates (x, y)
        valid_mask: [N] bool mask of voxels that project inside the image
    """
    N = voxel_coords.shape[0]
    
    # Homogeneous coordinates
    ones = torch.ones(N, 1, device=voxel_coords.device)
    voxel_homo = torch.cat([voxel_coords, ones], dim=1)  # [N, 4]
    
    # Project to image plane
    if calib.shape[0] == 3:
        pixel_homo = torch.matmul(calib, voxel_homo.T).T  # [N, 3]
    else:
        pixel_homo = torch.matmul(calib[:3], voxel_homo.T).T  # [N, 3]
    
    # Normalize by depth
    depth = pixel_homo[:, 2:3]
    depth = depth.clamp(min=1e-5)
    pixel_coords = pixel_homo[:, :2] / depth  # [N, 2]
    
    # Valid mask: inside image bounds and positive depth
    H, W = img_shape
    valid_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H) &
        (depth.squeeze() > 0)
    )
    
    return pixel_coords, valid_mask


def compute_voxnt_offsets(gt_occ):
    """
    Compute VoxNT offset labels (distance to instance boundaries in 6 directions).
    
    This is a NumPy implementation for offline preprocessing.
    
    Args:
        gt_occ: [X, Y, Z] ground truth occupancy labels
    
    Returns:
        offsets: [6, X, Y, Z] offset field (x+, x-, y+, y-, z+, z-)
    """
    X, Y, Z = gt_occ.shape
    offsets = np.zeros((6, X, Y, Z), dtype=np.int32)
    
    # For each of the 6 directions, compute run-length of same label
    for dim, (pos_idx, neg_idx) in enumerate([(0, 1), (2, 3), (4, 5)]):
        # Positive direction
        for start in range(gt_occ.shape[dim]):
            # Build slice for current position
            idx = [slice(None)] * 3
            idx[dim] = gt_occ.shape[dim] - 1
            
        # Simplified: use the torch implementation
        gt_tensor = torch.from_numpy(gt_occ).unsqueeze(0)
        
        # Import VoxNT
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        try:
            from tools.VoxNT_Trick import compute_all_direction_distances
            offsets_tensor = compute_all_direction_distances(gt_tensor)
            return offsets_tensor.squeeze(0).numpy()
        except ImportError:
            # Fallback: simple boundary detection
            for dim in range(3):
                for direction in range(2):
                    out_idx = dim * 2 + direction
                    shape = list(gt_occ.shape)
                    
                    if direction == 0:  # positive
                        for i in range(shape[dim] - 1, -1, -1):
                            slc = [slice(None)] * 3
                            slc[dim] = i
                            slc_next = [slice(None)] * 3
                            slc_next[dim] = min(i + 1, shape[dim] - 1)
                            
                            if i == shape[dim] - 1:
                                offsets[out_idx][tuple(slc)] = 1
                            else:
                                same = (gt_occ[tuple(slc)] == gt_occ[tuple(slc_next)])
                                offsets[out_idx][tuple(slc)] = np.where(
                                    same, offsets[out_idx][tuple(slc_next)] + 1, 1
                                )
                    else:  # negative
                        for i in range(shape[dim]):
                            slc = [slice(None)] * 3
                            slc[dim] = i
                            slc_prev = [slice(None)] * 3
                            slc_prev[dim] = max(i - 1, 0)
                            
                            if i == 0:
                                offsets[out_idx][tuple(slc)] = 1
                            else:
                                same = (gt_occ[tuple(slc)] == gt_occ[tuple(slc_prev)])
                                offsets[out_idx][tuple(slc)] = np.where(
                                    same, offsets[out_idx][tuple(slc_prev)] + 1, 1
                                )
            return offsets


def preprocess_sequence(sequence_dir, lseg_model, output_dir, calib,
                        voxel_size=0.2, scene_size=(256, 256, 32)):
    """
    Preprocess one sequence of SemanticKITTI for OV-VoxDet training.
    
    For each frame:
    1. Load image and voxel labels
    2. Extract LSeg features
    3. Compute voxel-pixel correspondences
    4. Compute VoxNT offsets
    5. Apply filtering
    6. Save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # List all frames
    img_dir = os.path.join(sequence_dir, 'image_2')
    label_dir = os.path.join(sequence_dir, 'voxels')
    
    if not os.path.exists(img_dir):
        print(f"Skipping {sequence_dir}: no images found")
        return
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    for img_file in img_files:
        frame_id = img_file.split('.')[0]
        output_path = os.path.join(output_dir, f'{frame_id}.npz')
        
        if os.path.exists(output_path):
            continue
        
        print(f"  Processing frame {frame_id}")
        
        # Load image (placeholder - actual loading depends on dataset format)
        # img = load_image(os.path.join(img_dir, img_file))
        
        # Load voxel labels (placeholder)
        # gt_occ = load_voxel_labels(os.path.join(label_dir, f'{frame_id}.bin'))
        
        # For the actual implementation, you would:
        # 1. lseg_feat, confidence = extract_lseg_features(img, lseg_model)
        # 2. pixel_coords, fov_mask = project_voxels_to_pixels(voxel_centers, calib, img.shape[-2:])
        # 3. offsets = compute_voxnt_offsets(gt_occ)
        # 4. valid_idx, pixel_feat, conf_w, bnd_w = filter_valid_voxels(...)
        # 5. np.savez(output_path, lseg_feat=..., valid_idx=..., ...)
        
        print(f"  Would save to {output_path}")


# ============================================================
# SemanticKITTI class definitions for open-vocabulary
# ============================================================

SEMANTICKITTI_CLASSES = {
    0: 'empty',
    1: 'car',
    2: 'bicycle', 
    3: 'motorcycle',
    4: 'truck',
    5: 'other-vehicle',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'traffic-sign',
}
# 
# Example base/novel split for open-vocabulary evaluation
BASE_CLASSES = [0, 1, 4, 6, 9, 11, 13, 15, 17]  # Common classes
NOVEL_CLASSES = [2, 3, 5, 7, 8, 10, 12, 14, 16, 18, 19]  # Rare classes


def main():
    parser = argparse.ArgumentParser(description='Preprocess data for OV-VoxDet')
    parser.add_argument('--kitti_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--lseg_model_path', type=str, default=None)
    parser.add_argument('--clip_model', type=str, default='ViT-B-32')
    parser.add_argument('--clip_pretrained', type=str, default='laion2b_s34b_b79k')
    parser.add_argument('--generate_text_embeddings', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.generate_text_embeddings:
        print("Generating CLIP text embeddings...")
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                args.clip_model, pretrained=args.clip_pretrained
            )
            tokenizer = open_clip.get_tokenizer(args.clip_model)
            class_names = [SEMANTICKITTI_CLASSES[i] for i in sorted(SEMANTICKITTI_CLASSES.keys())]
            embeddings = extract_clip_text_embeddings(
                class_names, model, tokenizer
            )
            output_path = os.path.join(args.output_dir, 'text_embeddings.pt')
            torch.save(embeddings, output_path)
            print(f"Saved text embeddings to {output_path}")
            
            # Also save as JSON for compatibility with OVO format
            import json
            emb_dict = {}
            for i, name in enumerate(class_names):
                emb_dict[name] = embeddings[i].cpu().tolist()
            json_path = os.path.join(args.output_dir, 'text_embeddings.json')
            with open(json_path, 'w') as f:
                json.dump(emb_dict, f)
            print(f"Saved text embeddings JSON to {json_path}")
            
        except ImportError:
            print("open_clip not installed. Install with: pip install open_clip_torch")
            return
    
    print("Preprocessing complete.")
    print(f"Output directory: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Generate LSeg features: run extract_lseg_features() for each image")
    print(f"  2. Compute voxel-pixel correspondences using camera calibration")
    print(f"  3. Run VoxNT to get offset labels")
    print(f"  4. Train OV-VoxDet with the preprocessed data")


if __name__ == '__main__':
    main()
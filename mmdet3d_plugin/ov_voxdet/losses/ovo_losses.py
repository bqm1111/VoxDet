"""
OVO-style distillation losses adapted for the VoxDet architecture.

OVO defines three alignment losses:
  1. L_vox-pix: Voxel-to-Pixel alignment (cosine similarity between 3D voxel 
     features and 2D LSeg pixel features, with confidence re-weighting)
  2. L_vox-txt: Voxel-to-Text alignment (replace classifier with CLIP text embeddings)
  3. L_2d: 2D alignment (regularize 2D features to match LSeg)

We add a fourth loss specific to the VoxDet integration:
  4. L_instance: Instance-consistency loss (voxels in the same instance should
     map to similar embeddings, using VoxNT-derived instance info)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelPixelAlignmentLoss(nn.Module):
    """
    Voxel-to-Pixel alignment loss (L_vox-pix).
    
    Aligns 3D voxel embeddings with 2D LSeg pixel embeddings via cosine similarity.
    Uses confidence-based re-weighting from LSeg predictions.
    
    Enhanced with VoxDet's instance-aware boundary weighting:
    voxels near instance boundaries (low offset values) get lower weight
    since their 2D projections are less reliable.
    """
    
    def __init__(self, use_confidence_weight=True, use_boundary_weight=True, 
                 boundary_threshold=2.0):
        super().__init__()
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        self.use_confidence_weight = use_confidence_weight
        self.use_boundary_weight = use_boundary_weight
        self.boundary_threshold = boundary_threshold
    
    def forward(self, aligned_vox_feat, lseg_pixel_feat, valid_indices, 
                confidence_weights=None, offset_field=None):
        """
        Args:
            aligned_vox_feat: [N_valid, 512] projected 3D voxel features (selected valid voxels)
            lseg_pixel_feat:  [N_valid, 512] corresponding LSeg 2D pixel features
            valid_indices:    indices of valid voxels (after pixel-voxel filtering)
            confidence_weights: [N_valid] LSeg prediction confidence for re-weighting
            offset_field:     [N_valid, 6] VoxNT offsets for boundary-aware weighting
        
        Returns:
            loss: scalar alignment loss
        """
        # Core cosine similarity loss
        similarity = self.cosine_sim(aligned_vox_feat, lseg_pixel_feat)
        loss = 1.0 - similarity  # [N_valid]
        
        # OVO-style confidence re-weighting
        if self.use_confidence_weight and confidence_weights is not None:
            loss = loss * confidence_weights
        
        # VoxDet-specific: boundary-aware weighting
        # Voxels near instance boundaries have less reliable 2D projections
        if self.use_boundary_weight and offset_field is not None:
            # Min offset across 6 directions = distance to nearest boundary
            min_offset = offset_field.min(dim=1)[0].float()  # [N_valid]
            # Soft boundary weight: sigmoid ramp from 0 to 1
            boundary_weight = torch.sigmoid(min_offset - self.boundary_threshold)
            loss = loss * boundary_weight
        
        return loss.mean()


class VoxelTextAlignmentLoss(nn.Module):
    """
    Voxel-to-Text alignment loss (L_vox-txt).
    
    Trains voxel embeddings to be classifiable by CLIP text embeddings.
    Uses cross-entropy between cosine similarity logits and ground truth labels.
    This replaces the standard learnable classifier with text embeddings.
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    
    def forward(self, aligned_vox_feat, text_embeddings, gt_labels, valid_mask=None):
        """
        Args:
            aligned_vox_feat: [B, 512, X, Y, Z] CLIP-aligned voxel features
            text_embeddings:  [num_classes, 512] CLIP text embeddings (base classes only for training)
            gt_labels:        [B, X, Y, Z] ground truth semantic labels
            valid_mask:       [B, X, Y, Z] bool mask of valid voxels for training
        
        Returns:
            loss: scalar cross-entropy loss
        """
        B, C, X, Y, Z = aligned_vox_feat.shape
        num_classes = text_embeddings.shape[0]
        
        # Normalize
        vox_norm = F.normalize(aligned_vox_feat, dim=1)           # [B, 512, X, Y, Z]
        text_norm = F.normalize(text_embeddings, dim=-1)           # [num_classes, 512]
        
        # Compute cosine similarity logits
        vox_flat = vox_norm.view(B, C, -1).permute(0, 2, 1)       # [B, N, 512]
        logits = torch.matmul(vox_flat, text_norm.T.to(vox_flat.device))  # [B, N, num_classes]
        logits = logits / self.temperature
        logits = logits.permute(0, 2, 1).view(B, num_classes, X, Y, Z)
        
        # Apply valid mask
        if valid_mask is not None:
            gt_labels = gt_labels.clone()
            gt_labels[~valid_mask] = 255
        
        loss = self.ce_loss(logits, gt_labels.long())
        return loss


class Align2DLoss(nn.Module):
    """
    2D feature alignment loss (L_2d).
    
    Regularizes VoxDet's 2D image features to align with LSeg features.
    Acts as an auxiliary loss to improve the quality of the 2D-to-3D lifting.
    """
    
    def __init__(self):
        super().__init__()
        self.cosine_sim = nn.CosineSimilarity(dim=1)
    
    def forward(self, aligned_2d_feat, lseg_2d_feat):
        """
        Args:
            aligned_2d_feat: [B, 512, H, W] projected 2D features
            lseg_2d_feat:    [B, 512, H, W] LSeg 2D features (target)
        
        Returns:
            loss: scalar alignment loss
        """
        # Resize if needed
        if aligned_2d_feat.shape[-2:] != lseg_2d_feat.shape[-2:]:
            aligned_2d_feat = F.interpolate(
                aligned_2d_feat, size=lseg_2d_feat.shape[-2:],
                mode='bilinear', align_corners=True
            )
        
        # Normalize
        aligned_2d_feat = F.normalize(aligned_2d_feat, dim=1)
        lseg_2d_feat = F.normalize(lseg_2d_feat, dim=1)
        
        # Flatten spatial dims
        B, C, H, W = aligned_2d_feat.shape
        aligned_flat = aligned_2d_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, 512]
        lseg_flat = lseg_2d_feat.flatten(2).permute(0, 2, 1)        # [B, H*W, 512]
        
        similarity = self.cosine_sim(
            aligned_flat.reshape(-1, C), lseg_flat.reshape(-1, C)
        )
        loss = (1.0 - similarity).mean()
        return loss


class InstanceConsistencyLoss(nn.Module):
    """
    Instance-consistency loss (L_instance) — novel to OV-VoxDet.
    
    Leverages VoxDet's VoxNT-derived instance information to enforce that
    voxels belonging to the same instance map to similar CLIP embeddings.
    
    This provides a self-consistency regularization that is unique to the
    VoxDet + OVO combination: OVO alone cannot use instance information,
    and VoxDet alone doesn't use language-aligned features.
    """
    
    def __init__(self, num_samples=256, margin=0.1):
        super().__init__()
        self.num_samples = num_samples
        self.margin = margin
    
    def forward(self, aligned_vox_feat, gt_labels, gt_offsets):
        """
        Args:
            aligned_vox_feat: [B, 512, X, Y, Z] CLIP-aligned voxel features
            gt_labels:        [B, X, Y, Z] ground truth semantic labels
            gt_offsets:       [B, 6, X, Y, Z] VoxNT offset labels
        
        Returns:
            loss: scalar instance-consistency loss
        """
        B, C, X, Y, Z = aligned_vox_feat.shape
        device = aligned_vox_feat.device
        total_loss = 0.0
        count = 0
        
        for b in range(B):
            feat = aligned_vox_feat[b]   # [512, X, Y, Z]
            labels = gt_labels[b]         # [X, Y, Z]
            offsets = gt_offsets[b]        # [6, X, Y, Z]
            
            # Find non-empty, non-ignored voxels
            valid = (labels != 0) & (labels != 255)
            if valid.sum() < 2:
                continue
            
            valid_idx = valid.nonzero(as_tuple=False)  # [N_valid, 3]
            
            if valid_idx.shape[0] > self.num_samples:
                perm = torch.randperm(valid_idx.shape[0], device=device)[:self.num_samples]
                valid_idx = valid_idx[perm]
            
            # Get features and labels for sampled voxels
            sampled_feat = feat[:, valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]  # [512, N]
            sampled_feat = F.normalize(sampled_feat, dim=0).T  # [N, 512]
            sampled_labels = labels[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]  # [N]
            
            # Compute pairwise similarity
            sim_matrix = torch.mm(sampled_feat, sampled_feat.T)  # [N, N]
            
            # Same-class mask (proxy for same-instance since VoxNT groups by class boundaries)
            label_match = (sampled_labels.unsqueeze(0) == sampled_labels.unsqueeze(1))  # [N, N]
            
            # Also check if voxels are in the same local instance using offsets
            sampled_offsets = offsets[:, valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]  # [6, N]
            
            # Intra-class: maximize similarity (same class voxels should be similar)
            if label_match.sum() > 1:
                pos_sim = sim_matrix[label_match]
                loss_pos = (1.0 - pos_sim).clamp(min=0).mean()
                total_loss += loss_pos
                count += 1
            
            # Inter-class: minimize similarity with margin
            if (~label_match).sum() > 0:
                neg_sim = sim_matrix[~label_match]
                loss_neg = (neg_sim - self.margin).clamp(min=0).mean()
                total_loss += loss_neg
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / count


class OVVoxDetLoss(nn.Module):
    """
    Combined loss module for OV-VoxDet.
    
    Integrates VoxDet's original losses with OVO's distillation losses:
    
    VoxDet losses (supervised):
      - L_ce: Cross-entropy for semantic classification
      - L_sem_scal / L_geo_scal: Semantic/geometric scaling losses  
      - L_reg: Offset regression loss (VoxNT supervision)
    
    OVO distillation losses (for open-vocabulary):
      - L_vox_pix: Voxel-to-pixel alignment
      - L_vox_txt: Voxel-to-text alignment
      - L_2d: 2D feature alignment
      - L_inst: Instance consistency (novel)
    """
    
    def __init__(self, 
                 lambda_vox_pix=1.0,
                 lambda_vox_txt=1.0, 
                 lambda_2d=0.1,
                 lambda_instance=0.5,
                 use_confidence_weight=True,
                 use_boundary_weight=True,
                 temperature=0.1):
        super().__init__()
        
        self.lambda_vox_pix = lambda_vox_pix
        self.lambda_vox_txt = lambda_vox_txt
        self.lambda_2d = lambda_2d
        self.lambda_instance = lambda_instance
        
        self.vox_pix_loss = VoxelPixelAlignmentLoss(
            use_confidence_weight=use_confidence_weight,
            use_boundary_weight=use_boundary_weight
        )
        self.vox_txt_loss = VoxelTextAlignmentLoss(temperature=temperature)
        self.align_2d_loss = Align2DLoss()
        self.instance_loss = InstanceConsistencyLoss()
    
    def forward(self, aligned_vox_feat, aligned_2d_feat=None,
                lseg_pixel_feat=None, lseg_2d_feat=None,
                text_embeddings=None, gt_labels=None,
                valid_vox_indices=None, confidence_weights=None,
                offset_field=None, gt_offsets=None, valid_mask=None):
        """
        Compute all OVO distillation losses.
        
        Returns:
            loss_dict: dict of individual loss terms
        """
        loss_dict = {}
        
        # 1. Voxel-to-Pixel alignment
        if lseg_pixel_feat is not None and valid_vox_indices is not None:
            B, C, X, Y, Z = aligned_vox_feat.shape
            # Select valid voxel features
            vox_flat = aligned_vox_feat.view(B, C, -1).permute(0, 2, 1)  # [B, N, 512]
            
            # For batch size 1 (common in SSC)
            valid_vox_feat = vox_flat[0].index_select(0, valid_vox_indices)
            
            loss_vox_pix = self.vox_pix_loss(
                valid_vox_feat.unsqueeze(0).permute(0, 2, 1),  # reshape for cosine sim
                lseg_pixel_feat.unsqueeze(0).permute(0, 2, 1),
                valid_vox_indices,
                confidence_weights=confidence_weights,
                offset_field=offset_field
            )
            loss_dict['loss_vox_pix'] = self.lambda_vox_pix * loss_vox_pix
        
        # 2. Voxel-to-Text alignment
        if text_embeddings is not None and gt_labels is not None:
            loss_vox_txt = self.vox_txt_loss(
                aligned_vox_feat, text_embeddings, gt_labels, valid_mask
            )
            loss_dict['loss_vox_txt'] = self.lambda_vox_txt * loss_vox_txt
        
        # 3. 2D alignment
        if aligned_2d_feat is not None and lseg_2d_feat is not None:
            loss_2d = self.align_2d_loss(aligned_2d_feat, lseg_2d_feat)
            loss_dict['loss_align_2d'] = self.lambda_2d * loss_2d
        
        # 4. Instance consistency
        if gt_offsets is not None and gt_labels is not None:
            loss_inst = self.instance_loss(aligned_vox_feat, gt_labels, gt_offsets)
            loss_dict['loss_instance'] = self.lambda_instance * loss_inst
        
        return loss_dict
"""
OV-VoxDet: Open-Vocabulary VoxDet

A subclass of VoxDet that adds OVO-style knowledge distillation for
open-vocabulary 3D semantic occupancy prediction.

Drops into VoxDet's existing training pipeline unchanged:
  - pl_model calls build_model(config['model']) → OVVoxDet
  - pl_model.training_step calls self.model(data_dict)
  - Returns {'losses': dict, 'pred': tensor, 'gt_occ': tensor}

The only changes vs VoxDet:
  1. Additional distiller modules project cls_feat → CLIP space
  2. forward_train adds distillation losses to the loss dict
  3. forward_test supports open-vocabulary inference via text embeddings

Usage:
  In config, set model.type = 'OVVoxDet' and add model.ov_config dict.
  Everything else (training script, pl_model, dataset) stays the same.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmdet3d_plugin.models.detectors.VoxDet import VoxDet, compute_all_direction_distances
from ...ov_voxdet.models.distiller import VoxDetDistiller3D, VoxDetDistiller2D, TextEmbeddingClassifier
from ...ov_voxdet.losses.ovo_losses import OVVoxDetLoss



@DETECTORS.register_module()
class OVVoxDet(VoxDet):
    """Open-Vocabulary VoxDet.
    
    Extends VoxDet with OVO knowledge distillation. The regression branch
    (offset prediction) is unchanged; distillation targets only the
    classification branch features.
    
    Args:
        ov_config (dict): Open-vocabulary configuration containing:
            - cls_feat_channels (int): Channels of cls_feat from SpatiallyDecoupledFPN (128)
            - img_feat_channels (int): Channels of 2D image features (128)
            - embedding_dim (int): CLIP/LSeg embedding dimension (512)
            - text_embedding_path (str): Path to pre-computed CLIP text embeddings JSON
            - lambda_vox_pix (float): Weight for voxel-pixel alignment loss
            - lambda_vox_txt (float): Weight for voxel-text alignment loss
            - lambda_2d (float): Weight for 2D alignment loss
            - lambda_instance (float): Weight for instance consistency loss
            - temperature (float): Temperature for cosine similarity
            - use_confidence_weight (bool): Use LSeg confidence weighting
            - use_boundary_weight (bool): Use VoxNT boundary weighting
            - confidence_threshold (float): Min LSeg confidence
            - boundary_threshold (float): Min offset for boundary detection
            - ov_mode (str): 'hybrid' | 'ov_only' | 'supervised'
        **kwargs: All VoxDet arguments (forwarded to super().__init__)
    """
    
    def __init__(self, ov_config, **kwargs):
        super().__init__(**kwargs)
        
        self.ov_config = ov_config
        self.ov_mode = ov_config.get('ov_mode', 'hybrid')
        
        cls_feat_ch = ov_config.get('cls_feat_channels', 128)
        img_feat_ch = ov_config.get('img_feat_channels', 128)
        embed_dim = ov_config.get('embedding_dim', 512)
        temperature = ov_config.get('temperature', 0.1)
        
        # --- Distiller modules ---
        # Projects cls_feat [B, 128, X, Y, Z] → CLIP space [B, 512, X, Y, Z]
        self.distiller_3d = VoxDetDistiller3D(
            in_channels=cls_feat_ch,
            embedding_dim=embed_dim,
        )
        
        # Projects 2D image features → LSeg space (regularizer)
        self.distiller_2d = VoxDetDistiller2D(
            in_channels=img_feat_ch,
            embedding_dim=embed_dim,
        )
        
        # Text embedding classifier for open-vocabulary inference
        self.text_classifier = TextEmbeddingClassifier(
            embedding_dim=embed_dim,
            temperature=temperature,
        )
        
        # --- Distillation loss ---
        self.ov_loss = OVVoxDetLoss(
            lambda_vox_pix=ov_config.get('lambda_vox_pix', 1.0),
            lambda_vox_txt=ov_config.get('lambda_vox_txt', 1.0),
            lambda_2d=ov_config.get('lambda_2d', 0.1),
            lambda_instance=ov_config.get('lambda_instance', 0.5),
            use_confidence_weight=ov_config.get('use_confidence_weight', True),
            use_boundary_weight=ov_config.get('use_boundary_weight', True),
            temperature=temperature,
        )
        
        # --- Load text embeddings ---
        text_emb_path = ov_config.get('text_embedding_path', None)
        if text_emb_path is not None:
            self._load_text_embeddings(text_emb_path)
    
    def _load_text_embeddings(self, path):
        """Load pre-computed CLIP text embeddings from JSON file."""
        if not path or not __import__('os').path.exists(path):
            return
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'embeddings' in data:
            emb = torch.tensor(data['embeddings'], dtype=torch.float32)
        elif isinstance(data, list):
            emb = torch.tensor(data, dtype=torch.float32)
        else:
            return
        
        self.register_buffer('_text_embeddings', emb)
        self.text_classifier.set_text_embeddings(emb)
    
    def forward_train(self, data_dict):
        """Training forward pass.
        
        Runs VoxDet's full supervised pipeline, then adds OVO distillation
        losses if ov_mode is 'hybrid' or 'ov_only'.
        
        Returns:
            dict with 'losses', 'pred', 'gt_occ' — same interface as VoxDet.
        """
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']
        
        # ============================================================
        # Stage 1: VoxDet feature extraction (identical to parent)
        # ============================================================
        img_voxel_feats, depth, proposal = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        
        # SpatiallyDecoupledFPN returns (cls_feat_list, reg_feat_list)
        if type(voxel_feats_enc) is tuple:
            voxel_feats_enc = list(voxel_feats_enc)
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        # ============================================================
        # Stage 2: Extract cls_feat for distillation
        # ============================================================
        # voxel_feats_enc[0] = cls_feat_list, voxel_feats_enc[1] = reg_feat_list
        # cls_feat_list[0] is [B, 128, X, Y, Z]
        cls_feat = voxel_feats_enc[0][0]  # [B, 128, X, Y, Z]
        
        # ============================================================
        # Stage 3: Compute VoxNT offsets (same as parent)
        # ============================================================
        with torch.no_grad():
            gt_occ_ = gt_occ.clone()
            gt_offset = compute_all_direction_distances(gt_occ_)
        
        # Apply gt_occ filtering for oversized cars (same as parent)
        if self.use_gt_refine:
            len_x = gt_offset[:, 0] + gt_offset[:, 1]
            len_y = gt_offset[:, 2] + gt_offset[:, 3]
            len_z = gt_offset[:, 4] + gt_offset[:, 5]
            if self.car_scale_filter_max is not None:
                x_mask_max = len_x > self.car_scale_filter_max[0]
                y_mask_max = len_y > self.car_scale_filter_max[1]
                z_mask_max = len_z > self.car_scale_filter_max[2]
                cls_mask = (gt_occ_ == 1)
                car_mask_max = (x_mask_max | y_mask_max | z_mask_max).bool() & cls_mask
                gt_occ_[car_mask_max] = 255
            if self.car_scale_filter_min is not None:
                x_mask_min = len_x < self.car_scale_filter_min[0]
                y_mask_min = len_y < self.car_scale_filter_min[1]
                z_mask_min = len_z < self.car_scale_filter_min[2]
                car_mask_min = (x_mask_min & y_mask_min & z_mask_min).bool() & cls_mask
                gt_occ_[car_mask_min] = 255
            if self.global_scale_filter_min is not None:
                x_mask_min_g = len_x < self.global_scale_filter_min[0]
                y_mask_min_g = len_x < self.global_scale_filter_min[1]
                z_mask_min_g = len_x < self.global_scale_filter_min[2]
                mask_min_g = x_mask_min_g & y_mask_min_g & z_mask_min_g
                gt_occ_[mask_min_g] = 255
            gt_occ = gt_occ_
        
        # ============================================================
        # Stage 4: VoxDet supervised losses (detection head)
        # ============================================================
        losses = dict()
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ,
            gt_offset=gt_offset,
        )
        
        # Auxiliary head loss
        if hasattr(self, 'pts_bbox_head_aux'):
            if type(img_voxel_feats) is not list:
                img_voxel_feats_list = [img_voxel_feats]
            else:
                img_voxel_feats_list = img_voxel_feats
            output_aux = self.pts_bbox_head_aux(
                voxel_feats=img_voxel_feats_list,
                img_metas=img_metas,
                img_feats=None,
                gt_occ=gt_occ,
            )
            if 'output_bbox' in output_aux:
                losses_aux = self.pts_bbox_head_aux.loss(
                    output_voxels=output_aux['output_voxels'],
                    target_voxels=gt_occ,
                    output_bbox=output_aux['output_bbox'],
                )
            else:
                losses_aux = self.pts_bbox_head_aux.loss(
                    output_voxels=output_aux['output_voxels'],
                    target_voxels=gt_occ,
                )
            for key in losses_aux:
                losses[key.replace('loss', 'loss_aux')] = losses_aux[key]
        
        # Depth loss
        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(
                data_dict['img_metas']['gt_depths'], depth
            )
        
        # Main VoxDet head loss (classification + regression)
        if self.ov_mode != 'ov_only':
            losses_occ = self.pts_bbox_head.loss(
                output_voxels=output['output_voxels'],
                target_voxels=gt_occ,
                output_bbox=output['output_bbox'],
                img_metas=img_metas,
                gt_offset=gt_offset,
            )
            losses.update(losses_occ)
        
        # ============================================================
        # Stage 5: OVO distillation losses
        # ============================================================
        if self.ov_mode != 'supervised':
            # Project cls_feat to CLIP space
            aligned_vox_feat = self.distiller_3d(cls_feat)  # [B, 512, X, Y, Z]
            
            # 2D feature alignment (optional regularizer)
            aligned_2d_feat = None
            lseg_2d_feat = data_dict.get('lseg_2d_feat', None)
            if lseg_2d_feat is not None:
                # Use img_enc features from the image encoder
                # Re-extract 2D features (already computed in extract_img_feat)
                img_enc_feats = self.image_encoder(img_inputs[0])  # [B, N, C, H, W]
                B, N, C, H, W = img_enc_feats.shape
                aligned_2d_feat = self.distiller_2d(
                    img_enc_feats.view(B * N, C, H, W)
                )  # [B*N, 512, H, W]
            
            # Get LSeg features from data pipeline
            lseg_pixel_feat = data_dict.get('lseg_pixel_feat', None)
            lseg_confidence = data_dict.get('lseg_confidence', None)
            valid_vox_indices = data_dict.get('valid_vox_indices', None)
            
            # Text embeddings for L_vox_txt
            text_emb = getattr(self, '_text_embeddings', None)
            
            # Valid mask: non-empty, non-ignored voxels
            valid_mask = (gt_occ != 0) & (gt_occ != 255)
            
            # Offset field from regression branch for boundary weighting
            offset_field = output.get('output_bbox', None)
            
            # Compute all distillation losses
            ov_losses = self.ov_loss(
                aligned_vox_feat=aligned_vox_feat,
                aligned_2d_feat=aligned_2d_feat,
                lseg_pixel_feat=lseg_pixel_feat,
                lseg_2d_feat=lseg_2d_feat,
                text_embeddings=text_emb,
                gt_labels=gt_occ,
                valid_vox_indices=valid_vox_indices,
                confidence_weights=lseg_confidence,
                offset_field=offset_field,
                gt_offsets=gt_offset,
                valid_mask=valid_mask,
            )
            losses.update(ov_losses)
        
        # ============================================================
        # Stage 6: Prediction (same as parent)
        # ============================================================
        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)
        
        return {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ,
        }
    
    def forward_test(self, data_dict):
        """Test forward pass.
        
        Supports two modes:
          - Supervised: uses VoxDetHead's learned cls_convs (same as VoxDet)
          - Open-vocabulary: uses distiller_3d + text_classifier
        
        Returns:
            dict with 'pred', 'gt_occ' — same interface as VoxDet.
        """
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict.get('gt_occ', None)
        
        img_voxel_feats, depth, proposal = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        
        if type(voxel_feats_enc) is tuple:
            voxel_feats_enc = list(voxel_feats_enc)
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        # Check if text embeddings are available for OV inference
        has_text_emb = (
            self.text_classifier.text_embeddings is not None
            and self.ov_mode != 'supervised'
        )
        
        if has_text_emb:
            # Open-vocabulary path: distiller + text classifier
            cls_feat = voxel_feats_enc[0][0]  # [B, 128, X, Y, Z]
            aligned_vox_feat = self.distiller_3d(cls_feat)  # [B, 512, X, Y, Z]
            ov_logits = self.text_classifier(aligned_vox_feat)  # [B, num_classes, X, Y, Z]
            
            # Upsample to full occ_size if needed
            target_size = tuple(self.pts_bbox_head.occ_size)
            if ov_logits.shape[2:] != target_size:
                ov_logits = F.interpolate(
                    ov_logits, size=target_size,
                    mode='trilinear', align_corners=False
                )
            pred = torch.argmax(ov_logits, dim=1)
        else:
            # Supervised path: standard VoxDetHead classification
            output = self.pts_bbox_head(
                voxel_feats=voxel_feats_enc,
                img_metas=img_metas,
                img_feats=None,
                gt_occ=gt_occ,
            )
            pred = torch.argmax(output['output_voxels'], dim=1)
        
        return {
            'pred': pred,
            'gt_occ': gt_occ,
        }
    
    def set_ov_classes(self, class_names, clip_model=None, tokenizer=None,
                       text_embeddings=None):
        """Set new classes for open-vocabulary inference at test time.
        
        Either provide pre-computed text_embeddings or class_names + clip_model
        + tokenizer to compute them on the fly.
        
        Args:
            class_names (list[str]): List of class names for OV inference
            clip_model: open_clip model for encoding class names (optional)
            tokenizer: open_clip tokenizer returned by open_clip.get_tokenizer() (optional)
            text_embeddings (Tensor): [num_classes, 512] pre-computed embeddings (optional)
        """
        if text_embeddings is not None:
            self.text_classifier.set_text_embeddings(text_embeddings)
            self.register_buffer('_text_embeddings', text_embeddings)
        elif clip_model is not None:
            import open_clip
            device = next(self.parameters()).device
            if tokenizer is None:
                tokenizer = open_clip.get_tokenizer('ViT-B-32')
            text_tokens = tokenizer(
                [f"a photo of a {c}" for c in class_names]
            ).to(device)
            with torch.no_grad():
                emb = clip_model.encode_text(text_tokens).float()
            self.text_classifier.set_text_embeddings(emb)
            self.register_buffer('_text_embeddings', emb)
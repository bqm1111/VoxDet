import torch
from voxdet_core import BaseModule
from voxdet_core import DETECTORS, MODELS
from voxdet_core import build_from_cfg
import torch.nn.functional as F

def run_length_positive(t, dim):
    """Vectorized run-length encoding along a dimension.

    For each position, counts how many consecutive equal values follow it
    (including itself). Uses cumsum + cummax reset trick: O(L) parallel work,
    no Python-level loops.
    """
    t_moved = t.movedim(dim, -1)  # [..., L]

    # boundary[i] = True means position i is the LAST in its forward run
    boundary = torch.ones_like(t_moved, dtype=torch.bool)
    boundary[..., :-1] = (t_moved[..., :-1] != t_moved[..., 1:])

    # Reverse: boundary_rev marks segment STARTs in the reversed order
    boundary_rev = boundary.flip(-1)

    # Cumsum of 1s gives global position; reset at segment starts via cummax
    ones = torch.ones_like(t_moved, dtype=torch.int32)
    cum = ones.cumsum(-1)  # 1, 2, 3, ...
    reset = torch.where(boundary_rev, cum - 1, torch.zeros_like(cum))
    reset_cummax = reset.cummax(-1)[0]
    result = cum - reset_cummax  # 1-based run length within each segment

    return result.flip(-1).movedim(-1, dim)


def run_length_along_dim(t, dim, direction):
    if direction == 'positive':
        return run_length_positive(t, dim)
    else:
        t_flip = torch.flip(t, dims=(dim,))
        out_flip = run_length_positive(t_flip, dim)
        return torch.flip(out_flip, dims=(dim,))

def compute_all_direction_distances(gt_occ):
    B, X, Y, Z = gt_occ.shape

    dist_x_pos = run_length_along_dim(gt_occ, 1, 'positive')
    dist_x_neg = run_length_along_dim(gt_occ, 1, 'negative')
    dist_y_pos = run_length_along_dim(gt_occ, 2, 'positive')
    dist_y_neg = run_length_along_dim(gt_occ, 2, 'negative')
    dist_z_pos = run_length_along_dim(gt_occ, 3, 'positive')
    dist_z_neg = run_length_along_dim(gt_occ, 3, 'negative')

    distances = torch.stack([dist_x_pos, dist_x_neg, dist_y_pos, dist_y_neg, dist_z_pos, dist_z_neg], dim=1)
    return distances

@DETECTORS.register_module()
class VoxDet(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        pts_bbox_head_aux=None,
        depth_loss=False,
        train_cfg=None,
        test_cfg=None,
        use_gt_refine=False,
        car_scale_filter_max=None,
        car_scale_filter_min=None,
        global_scale_filter_min=None,
    ):
        super().__init__()
        
        self.img_backbone = build_from_cfg(img_backbone, MODELS)
        self.img_neck = build_from_cfg(img_neck, MODELS)
        self.global_scale_filter_min = global_scale_filter_min
        self.depth_net = build_from_cfg(depth_net, MODELS)
        if img_view_transformer is not None:
            self.img_view_transformer = build_from_cfg(img_view_transformer, MODELS)
        self.proposal_layer = build_from_cfg(proposal_layer, MODELS)
        self.VoxFormer_head = build_from_cfg(VoxFormer_head, MODELS)
        self.use_gt_refine = use_gt_refine
        self.car_scale_filter_max = car_scale_filter_max
        self.car_scale_filter_min = car_scale_filter_min

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = build_from_cfg(occ_encoder_backbone, MODELS)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = build_from_cfg(occ_encoder_neck, MODELS)

        self.pts_bbox_head = build_from_cfg(pts_bbox_head, MODELS)
        if pts_bbox_head_aux is not None:
            self.pts_bbox_head_aux = build_from_cfg(pts_bbox_head_aux, MODELS)
            
        self.depth_loss = depth_loss

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x

    def extract_img_feat(self, img_inputs, img_metas):
        img_enc_feats = self.image_encoder(img_inputs[0]) # torch.Size([1, 1, 640, 48, 160])
        if self.training and torch.isnan(img_enc_feats).any():
            raise ValueError(f"NaN in image_encoder output: shape={list(img_enc_feats.shape)}, nan={torch.isnan(img_enc_feats).sum().item()}")
        B,N,C,H,W =img_inputs[0].size()

        mlp_input = self.depth_net.get_mlp_input(*img_inputs[1:7])
        context, depth = self.depth_net([img_enc_feats] + img_inputs[1:7] + [mlp_input], img_metas)
        if self.training:
            if torch.isnan(context).any():
                raise ValueError(f"NaN in depth_net context: shape={list(context.shape)}, nan={torch.isnan(context).sum().item()}")
            if torch.isnan(depth).any():
                raise ValueError(f"NaN in depth_net depth: shape={list(depth.shape)}, nan={torch.isnan(depth).sum().item()}")
        #1, 1, 128, 48, 160
        if hasattr(self, 'img_view_transformer'):
            coarse_queries = self.img_view_transformer(context, depth, img_inputs[1:7]) # V_QA
        else:
            coarse_queries = None
        if self.training and coarse_queries is not None and torch.isnan(coarse_queries).any():
            raise ValueError(f"NaN in img_view_transformer output: shape={list(coarse_queries.shape)}, nan={torch.isnan(coarse_queries).sum().item()}")

        proposal = self.proposal_layer(img_inputs[1:7], img_metas)
        # torch.Size([1, 1, 128, 128, 16])
        # torch.Size([1, 1, 128, 48, 160])
        if B > 1:
            x_list = []
            for i in range(B):
                camera_paras = img_inputs[1:7]
                camera_paras_batch = []
                
                for j in range(6):
                    camera_paras_batch.append(camera_paras[j][i:i+1])

                x = self.VoxFormer_head(
                    [context[i:i+1]],
                    proposal[i:i+1],
                    cam_params=camera_paras_batch,
                    lss_volume=coarse_queries[i:i+1],
                    img_metas=img_metas,
                    mlvl_dpt_dists=[depth[i:i+1].unsqueeze(1)]
                )
                if self.training and torch.isnan(x).any():
                    raise ValueError(f"NaN in VoxFormer_head output for batch item {i}: shape={list(x.shape)}, nan={torch.isnan(x).sum().item()}")
                x_list.append(x)
            x = torch.cat(x_list, dim=0)
        else:
            x = self.VoxFormer_head(
                [context],
                proposal,
                cam_params=img_inputs[1:7],
                lss_volume=coarse_queries,
                img_metas=img_metas,
                mlvl_dpt_dists=[depth.unsqueeze(1)]
            )
        
        # ([1, 1, 128, 128, 16])
        # print(x.shape)
        # torch.Size([1, 128, 128, 128, 16])
        # torch.Size([1, 112, 48, 160])
        return x, depth, proposal

    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)

        return x

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth, proposal = self.extract_img_feat(img_inputs, img_metas)
        # --- NaN trace ---
        if torch.isnan(img_voxel_feats).any():
            raise ValueError(f"NaN after extract_img_feat: img_voxel_feats "
                             f"shape={list(img_voxel_feats.shape)}, "
                             f"nan_count={torch.isnan(img_voxel_feats).sum().item()}")
        if torch.isnan(depth).any():
            raise ValueError(f"NaN after extract_img_feat: depth")
        # --- end NaN trace ---
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        # if len(voxel_feats_enc) > 1:
        #     voxel_feats_enc = [voxel_feats_enc[0]]
        if type(voxel_feats_enc) is tuple:
            voxel_feats_enc = list(voxel_feats_enc)

        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        # --- NaN trace ---
        def _check_nan(obj, prefix):
            if isinstance(obj, torch.Tensor):
                if torch.isnan(obj).any():
                    raise ValueError(f"NaN at {prefix}: shape={list(obj.shape)}, "
                                     f"nan_count={torch.isnan(obj).sum().item()}")
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    _check_nan(item, f"{prefix}[{i}]")
        _check_nan(voxel_feats_enc, "occ_encoder output")
        # --- end NaN trace ---

        with torch.no_grad():
            gt_occ_ = gt_occ.clone() 
            gt_offset = compute_all_direction_distances(gt_occ_)

        
        if self.use_gt_refine:
            # Remove super long cars as mentioned in the Appendix
            len_x = gt_offset[:,0,:,:,:] + gt_offset[:,1,:,:,:] # X axis b x y z
            len_y = gt_offset[:,2,:,:,:] + gt_offset[:,3,:,:,:]
            len_z = gt_offset[:,4,:,:,:] + gt_offset[:,5,:,:,:]
            if self.car_scale_filter_max is not None:

                x_mask_max = len_x > self.car_scale_filter_max[0]
                y_mask_max = len_y > self.car_scale_filter_max[1]
                z_mask_max = len_z > self.car_scale_filter_max[2] 

                cls_mask = (gt_occ_ == 1) # for car
                car_mask_max =(x_mask_max | y_mask_max | z_mask_max).bool() & cls_mask
                gt_occ_[car_mask_max] = 255


            if self.car_scale_filter_min is not None:
                x_mask_min = len_x < self.car_scale_filter_min[0]
                y_mask_min = len_y < self.car_scale_filter_min[1]
                z_mask_min = len_z < self.car_scale_filter_min[2]
                
                car_mask_min = (x_mask_min & y_mask_min & z_mask_min).bool()  & cls_mask # all thre axes are one
                gt_occ_[car_mask_min] = 255


            if self.global_scale_filter_min is not None:

                x_mask_min_g = len_x < self.global_scale_filter_min[0]
                y_mask_min_g  = len_x < self.global_scale_filter_min[1]
                z_mask_min_g  = len_x < self.global_scale_filter_min[2]

                mask_min_g = x_mask_min_g & y_mask_min_g & z_mask_min_g # isolated points
                gt_occ_[mask_min_g] = 255

                # gt_occ_[] = 255
                # gt_occ_[len_y < self.global_scale_filter_min[1]] = 255
                # gt_occ_[len_z < self.global_scale_filter_min[2]] = 255

            gt_occ = gt_occ_
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ,
            gt_offset=gt_offset,
        )

        # --- DEBUG: validate gt_occ and model outputs before loss ---
        n_classes = output['output_voxels'].shape[1]
        valid_mask = gt_occ != 255
        if valid_mask.any():
            valid_vals = gt_occ[valid_mask]
            bad = (valid_vals < 0) | (valid_vals >= n_classes)
            if bad.any():
                bad_vals = valid_vals[bad].unique().cpu().tolist()
                raise ValueError(
                    f"forward_train: gt_occ has values {bad_vals} outside "
                    f"[0, {n_classes}) (ignore=255)")
        if torch.isnan(output['output_voxels']).any():
            raise ValueError("forward_train: NaN detected in main head output_voxels")
        # --- END DEBUG ---

        losses = dict()
        if hasattr(self, 'pts_bbox_head_aux'):
            if type(img_voxel_feats) is not list:
                img_voxel_feats = [img_voxel_feats]
            output_aux = self.pts_bbox_head_aux(
                voxel_feats=img_voxel_feats,
                img_metas=img_metas,
                img_feats=None,
                gt_occ=gt_occ
            )
            if 'output_bbox' in output_aux.keys():
                losses_occupancy_aux = self.pts_bbox_head_aux.loss(
                    output_voxels=output_aux['output_voxels'],
                    target_voxels=gt_occ,
                    output_bbox=output_aux['output_bbox'],
                    )
            else:
                losses_occupancy_aux = self.pts_bbox_head_aux.loss(
                    output_voxels=output_aux['output_voxels'],
                    target_voxels=gt_occ,
                )
            
            loss_dict = {}
            for key in losses_occupancy_aux.keys():
                loss_dict[key.replace('loss', 'loss_aux')] = losses_occupancy_aux[key]
            losses.update(loss_dict)


        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(data_dict['img_metas']['gt_depths'], depth)

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
            output_bbox=output['output_bbox'],
            img_metas=img_metas,
            gt_offset=gt_offset,
        )
        
        losses.update(losses_occupancy)
        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output
    # 
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ'] if 'gt_occ' in data_dict.keys() else None

        img_voxel_feats, depth, proposal = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if type(voxel_feats_enc) is tuple:
            voxel_feats_enc = list(voxel_feats_enc)

        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )
        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ
        }

        return test_output


    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)
        
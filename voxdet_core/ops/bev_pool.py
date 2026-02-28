import torch


def bev_pool(feats, geom_feats, B, D, H, W):
    """Pure-PyTorch BEV pooling using scatter_add.

    Drop-in replacement for the CUDA bev_pool kernel from mmdetection3d.
    The CUDA kernel uses a non-standard column ordering for geom_feats:

        column 0 → H dimension (x-axis / height in output)
        column 1 → W dimension (y-axis / width in output)
        column 2 → D dimension (z-axis / depth in output)
        column 3 → B dimension (batch index)

    This matches the caller convention in LSSViewTransformer.voxel_pooling,
    which constructs geom_feats as [x, y, z, batch_ix].

    Args:
        feats (Tensor): Feature tensor of shape (N, C).
        geom_feats (Tensor): Geometry indices of shape (N, 4) where columns
            are (x/H_idx, y/W_idx, z/D_idx, batch_idx) — matching the
            original CUDA kernel convention.
        B (int): Batch size.
        D (int): Depth dimension (z-axis).
        H (int): Height dimension (x-axis).
        W (int): Width dimension (y-axis).

    Returns:
        Tensor: BEV feature map of shape (B, C, D, H, W).
    """
    B, D, H, W = int(B), int(D), int(H), int(W)
    C = feats.shape[1]

    # Match the CUDA kernel's non-standard column ordering:
    #   col 0 -> H (x), col 1 -> W (y), col 2 -> D (z), col 3 -> B (batch)
    h_idx = geom_feats[:, 0].long()
    w_idx = geom_feats[:, 1].long()
    d_idx = geom_feats[:, 2].long()
    batch_idx = geom_feats[:, 3].long()

    # Filter valid indices
    valid = ((batch_idx >= 0) & (batch_idx < B) &
             (d_idx >= 0) & (d_idx < D) &
             (h_idx >= 0) & (h_idx < H) &
             (w_idx >= 0) & (w_idx < W))

    feats = feats[valid]
    batch_idx = batch_idx[valid]
    d_idx = d_idx[valid]
    h_idx = h_idx[valid]
    w_idx = w_idx[valid]

    linear_idx = batch_idx * D * H * W + d_idx * H * W + h_idx * W + w_idx

    # Scatter add
    output = torch.zeros(B * D * H * W, C, dtype=feats.dtype, device=feats.device)
    output.scatter_add_(0, linear_idx.unsqueeze(1).expand(-1, C), feats)

    output = output.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
    return output

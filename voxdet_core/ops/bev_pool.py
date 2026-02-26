import torch


def bev_pool(feats, geom_feats, B, D, H, W):
    """Pure-PyTorch BEV pooling using scatter_add.

    Args:
        feats (Tensor): Feature tensor of shape (N, C).
        geom_feats (Tensor): Geometry indices of shape (N, 4) where columns
            are (batch_idx, depth_idx, height_idx, width_idx).
        B (int): Batch size.
        D (int): Depth dimension.
        H (int): Height dimension.
        W (int): Width dimension.

    Returns:
        Tensor: BEV feature map of shape (B, C, D, H, W).
    """
    C = feats.shape[1]

    # Compute linear indices
    batch_idx = geom_feats[:, 0].long()
    d_idx = geom_feats[:, 1].long()
    h_idx = geom_feats[:, 2].long()
    w_idx = geom_feats[:, 3].long()

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

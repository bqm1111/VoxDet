import os
import sys
import torch

# ---------------------------------------------------------------------------
# Load the CUDA extension. Priority:
#   1. Pre-compiled .so in this directory (instant, no JIT overhead)
#   2. JIT compilation via torch.utils.cpp_extension.load
#   3. Pure-PyTorch fallback (scatter_add)
# ---------------------------------------------------------------------------
_USE_CUDA = False
bev_pool_ext = None

# Try pre-compiled .so first
_ops_dir = os.path.dirname(os.path.abspath(__file__))
if _ops_dir not in sys.path:
    sys.path.insert(0, _ops_dir)
try:
    import bev_pool_ext  # noqa: F811
    _USE_CUDA = True
except ImportError:
    # Try JIT compilation
    try:
        from torch.utils.cpp_extension import load as _load_ext
        _csrc_dir = os.path.join(_ops_dir, 'csrc')
        bev_pool_ext = _load_ext(
            name='bev_pool_ext',
            sources=[
                os.path.join(_csrc_dir, 'bev_pool.cpp'),
                os.path.join(_csrc_dir, 'bev_pool_cuda.cu'),
            ],
            verbose=False,
        )
        _USE_CUDA = True
    except Exception as e:
        import warnings
        warnings.warn(
            f'Failed to load/compile bev_pool CUDA extension, falling back '
            f'to pure-PyTorch scatter_add implementation: {e}')


# ---------------------------------------------------------------------------
# CUDA path (original mmdetection3d implementation)
# ---------------------------------------------------------------------------

class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None


def _bev_pool_cuda(feats, geom_feats, B, D, H, W):
    """CUDA bev_pool using pre-sorted intervals and fused summation kernel."""
    assert feats.shape[0] == geom_feats.shape[0]

    # Compute ranks for sorting — must match the CUDA kernel's memory layout:
    #   col 0 -> H (x), col 1 -> W (y), col 2 -> D (z), col 3 -> B (batch)
    #   output shape is [B, D, H, W, C], linear index = b*D*H*W + d*H*W + h*W + w
    ranks = (
        geom_feats[:, 0] * (W * D * B)
        + geom_feats[:, 1] * (D * B)
        + geom_feats[:, 2] * B
        + geom_feats[:, 3]
    )
    indices = ranks.argsort()
    feats, geom_feats, ranks = feats[indices], geom_feats[indices], ranks[indices]

    x = QuickCumsumCuda.apply(feats, geom_feats, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback path
# ---------------------------------------------------------------------------

def _bev_pool_pytorch(feats, geom_feats, B, D, H, W):
    """Pure-PyTorch BEV pooling using scatter_add.

    Drop-in replacement for the CUDA bev_pool kernel from mmdetection3d.
    The CUDA kernel uses a non-standard column ordering for geom_feats:

        column 0 -> H dimension (x-axis / height in output)
        column 1 -> W dimension (y-axis / width in output)
        column 2 -> D dimension (z-axis / depth in output)
        column 3 -> B dimension (batch index)

    This matches the caller convention in LSSViewTransformer.voxel_pooling,
    which constructs geom_feats as [x, y, z, batch_ix].
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


# ---------------------------------------------------------------------------
# Public API — auto-selects CUDA or PyTorch
# ---------------------------------------------------------------------------

def bev_pool(feats, geom_feats, B, D, H, W):
    """BEV pooling that uses the CUDA kernel when available, otherwise
    falls back to a pure-PyTorch scatter_add implementation.

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
    if _USE_CUDA:
        return _bev_pool_cuda(feats, geom_feats, B, D, H, W)
    else:
        return _bev_pool_pytorch(feats, geom_feats, B, D, H, W)


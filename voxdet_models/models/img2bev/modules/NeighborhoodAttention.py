#################################################################################################
# Copyright (c) 2023 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

from natten.functional import na2d


def _compute_window_starts(positions, half, d, ks, L):
    """
    Compute window start positions matching natten's clamped-window behavior.
    The window must: (1) contain the query position on the dilation grid,
    (2) stay within [0, L-1], (3) have exactly ks positions spaced by d.

    We compute m (query's index within the kernel window), then start = pos - m*d.
    m is ideally half (centered), clamped by boundary constraints.
    """
    # m_max: largest m such that start = pos - m*d >= 0
    m_max = positions // d  # [N]
    # m_min: smallest m such that start + (ks-1)*d <= L-1
    # pos - m*d + (ks-1)*d <= L-1  =>  m >= ceil((pos + (ks-1)*d - L + 1) / d)
    m_min = ((positions + (ks - 1) * d - L + 1).float() / d).ceil().long().clamp(min=0)
    m = half * torch.ones_like(positions)
    m = m.clamp(min=m_min, max=m_max)
    return positions - m * d


def na2d_with_rpb(q, k, v, kernel_size, dilation, rpb, scale):
    """
    Neighborhood attention 2D with relative positional bias.
    natten 0.20+ dropped RPB support. When RPB is needed, we implement
    neighborhood attention manually matching natten's clamped-window behavior.

    Args:
        q: [B, H, W, heads, head_dim]
        k: [B, H, W, heads, head_dim]
        v: [B, H, W, heads, head_dim]
        kernel_size: int, odd number
        dilation: int
        rpb: [heads, 2*kernel_size-1, 2*kernel_size-1] or None
        scale: float
    """
    if rpb is None:
        return na2d(q, k, v, kernel_size=kernel_size, dilation=dilation, scale=scale)

    B, H, W, heads, head_dim = q.shape
    ks = kernel_size
    d = dilation
    half = ks // 2

    kernel_offsets = torch.arange(ks, device=q.device) * d  # [ks]: 0, d, 2d, ...

    # Window starts for each query position, with grid-aligned clamping
    h_positions = torch.arange(H, device=q.device)
    start_h = _compute_window_starts(h_positions, half, d, ks, H)  # [H]
    neigh_h = start_h.unsqueeze(1) + kernel_offsets.unsqueeze(0)  # [H, ks]

    w_positions = torch.arange(W, device=q.device)
    start_w = _compute_window_starts(w_positions, half, d, ks, W)  # [W]
    neigh_w = start_w.unsqueeze(1) + kernel_offsets.unsqueeze(0)  # [W, ks]

    # Build 2D neighbor linear indices: [H, W, ks*ks]
    # For query (h,w), neighbor (r,c) is at (neigh_h[h,r], neigh_w[w,c])
    row_for_neighbor = neigh_h.unsqueeze(2).expand(H, ks, ks).reshape(H, 1, ks * ks).expand(H, W, ks * ks)
    col_for_neighbor = neigh_w.unsqueeze(1).expand(W, ks, ks).reshape(1, W, ks * ks).expand(H, W, ks * ks)
    linear_idx = row_for_neighbor * W + col_for_neighbor  # [H, W, ks*ks]

    # Gather k and v neighborhoods
    k_flat = k.reshape(B, H * W, heads, head_dim)
    v_flat = v.reshape(B, H * W, heads, head_dim)

    gather_idx = linear_idx.reshape(H * W * ks * ks)
    gather_idx = gather_idx[None, :, None, None].expand(B, -1, heads, head_dim)

    k_neigh = k_flat.gather(1, gather_idx).reshape(B, H, W, ks * ks, heads, head_dim)
    v_neigh = v_flat.gather(1, gather_idx).reshape(B, H, W, ks * ks, heads, head_dim)

    # Compute attention scores
    attn = (q.unsqueeze(3) * k_neigh).sum(-1) * scale  # [B, H, W, ks*ks, heads]

    # Compute RPB indices
    # RPB[head, dy, dx] where dy = (neigh_row - query_row)/d + (ks-1)
    query_rows = h_positions.view(H, 1, 1)
    query_cols = w_positions.view(1, W, 1)
    dy = (row_for_neighbor - query_rows) // d + (ks - 1)  # [H, W, ks*ks]
    dx = (col_for_neighbor - query_cols) // d + (ks - 1)  # [H, W, ks*ks]

    # Gather RPB values
    rpb_size = 2 * ks - 1
    rpb_linear = dy * rpb_size + dx  # [H, W, ks*ks]
    # rpb: [heads, rpb_size, rpb_size] -> index with [H, W, ks*ks] per head
    rpb_flat = rpb.reshape(heads, rpb_size * rpb_size)  # [heads, rpb_size^2]
    rpb_idx = rpb_linear.reshape(H * W * ks * ks).unsqueeze(0).expand(heads, -1)  # [heads, H*W*ks*ks]
    rpb_values = rpb_flat.gather(1, rpb_idx).reshape(heads, H, W, ks * ks)  # [heads, H, W, ks*ks]
    rpb_values = rpb_values.permute(1, 2, 3, 0).unsqueeze(0)  # [1, H, W, ks*ks, heads]

    attn = attn + rpb_values

    # Softmax over neighbors
    attn = attn.softmax(dim=3)  # [B, H, W, ks*ks, heads]

    # Weighted sum
    x = (attn.unsqueeze(-1) * v_neigh).sum(3)  # [B, H, W, heads, head_dim]

    return x


class NeighborhoodCrossAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        B, Hp, Wp, C = q.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape

        q = self.q(q).reshape(B, H, W, self.num_heads, self.head_dim)  # [B, H, W, heads, head_dim]
        kv = self.kv(kv).reshape(B, H, W, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, :, 0], kv[:, :, :, 1]  # [B, H, W, heads, head_dim]

        x = na2d_with_rpb(q, k, v, kernel_size=self.kernel_size, dilation=self.dilation, rpb=self.rpb, scale=self.scale)
        x = x.reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )

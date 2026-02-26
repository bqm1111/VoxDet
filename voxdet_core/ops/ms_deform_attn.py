import torch
import torch.nn.functional as F

def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations,
                                        attention_weights):
    """Pure-PyTorch multi-scale deformable attention.

    Args:
        value (Tensor): Shape (bs, num_keys, num_heads, embed_dims//num_heads).
        value_spatial_shapes (Tensor): Shape (num_levels, 2).
        sampling_locations (Tensor): Shape
            (bs, num_queries, num_heads, num_levels, num_points, 2).
        attention_weights (Tensor): Shape
            (bs, num_queries, num_heads, num_levels, num_points).

    Returns:
        Tensor: Shape (bs, num_queries, embed_dims).
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split(
        [H * W for H, W in value_spatial_shapes], dim=1)

    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H, W) in enumerate(value_spatial_shapes):
        H, W = int(H), int(W)
        # bs, H*W, num_heads, embed_dims -> bs*num_heads, embed_dims, H, W
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H, W)
        # bs, num_queries, num_heads, num_points, 2
        #   -> bs, num_heads, num_queries, num_points, 2
        #   -> bs*num_heads, num_queries, num_points, 2
        sampling_grid_l = sampling_grids[:, :, :, level].transpose(
            1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l = F.grid_sample(
            value_l,
            sampling_grid_l,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l)

    # (bs, num_queries, num_heads, num_levels, num_points)
    #   -> (bs, num_heads, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)

    # (bs*num_heads, embed_dims, num_queries, num_levels*num_points)
    output = torch.stack(sampling_value_list, dim=-1).flatten(-2)
    output = (output * attention_weights).sum(-1).view(
        bs, num_heads * embed_dims, num_queries)

    return output.transpose(1, 2).contiguous()

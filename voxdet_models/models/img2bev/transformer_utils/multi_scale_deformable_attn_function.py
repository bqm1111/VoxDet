# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Replaced CUDA ext_module with pure-PyTorch implementation from voxdet_core.

import torch
# from torch.cuda.amp import custom_bwd, custom_fwd
from torch.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable

from voxdet_core.ops import multi_scale_deformable_attn_pytorch


class MultiScaleDeformableAttnFunction_fp16(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16, device_type='cuda')
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return multi_scale_deformable_attn_pytorch(
            value, value_spatial_shapes, sampling_locations,
            attention_weights)

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        raise NotImplementedError(
            'Backward for MultiScaleDeformableAttnFunction_fp16 is not '
            'supported in pure-PyTorch mode. Use the standard '
            'multi_scale_deformable_attn_pytorch function directly instead '
            'of this custom autograd Function.')


class MultiScaleDeformableAttnFunction_fp32(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return multi_scale_deformable_attn_pytorch(
            value, value_spatial_shapes, sampling_locations,
            attention_weights)

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        raise NotImplementedError(
            'Backward for MultiScaleDeformableAttnFunction_fp32 is not '
            'supported in pure-PyTorch mode. Use the standard '
            'multi_scale_deformable_attn_pytorch function directly instead '
            'of this custom autograd Function.')

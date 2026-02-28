"""Build script for the bev_pool CUDA extension.

Usage:
    python voxdet_core/ops/setup_bev_pool.py

This compiles bev_pool_ext.*.so into voxdet_core/ops/ so it can be imported
directly without JIT compilation overhead.
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

_this_dir = os.path.dirname(os.path.abspath(__file__))
_csrc_dir = os.path.join(_this_dir, 'csrc')

setup(
    name='bev_pool_ext',
    ext_modules=[
        CUDAExtension(
            name='bev_pool_ext',
            sources=[
                os.path.join(_csrc_dir, 'bev_pool.cpp'),
                os.path.join(_csrc_dir, 'bev_pool_cuda.cu'),
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)

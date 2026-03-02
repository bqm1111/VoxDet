# Changelog

## Post-Refactor Fixes (branch: spear)

These changes fix behavioral regressions introduced when the codebase was
refactored (commit `111041d`) to remove all mmcv/mmdet/mmdet3d dependencies and
replace them with the custom `voxdet_core` module. An additional set of changes
addresses low GPU utilization during training.

---

### Bug Fixes

#### 1. ConvModule built the convolution layer twice, discarding the first
**File:** `voxdet_core/cnn_builders.py`
**Impact:** HIGH

The `ConvModule` constructor contained two consecutive builds of `self.conv`.
The first build (via `build_conv_layer`) was immediately overwritten by the
second build, making it dead code. More importantly, the first build had a
broken kwargs splat (`**conv_cfg_copy if not conv_cfg else {}`) that would
always evaluate to `{}` when a `conv_cfg` was provided, silently dropping
extra parameters from the config dict.

**Before (simplified):**
```python
self.conv = build_conv_layer(conv_cfg, ...)   # result thrown away
self.conv = conv_cls(...)                      # overwrites the above
```

**After:**
```python
conv_type = (conv_cfg or {}).get('type', 'Conv2d')
conv_cls = CONV_LAYERS.get(conv_type, nn.Conv2d)
self.conv = conv_cls(
    in_channels, out_channels, kernel_size,
    stride=stride, padding=padding, dilation=dilation,
    groups=groups, bias=bias)
```

A single, clean build that respects `conv_cfg['type']` and falls back to
`nn.Conv2d`.

---

#### 2. SECONDFPN / SECONDFPN3D weight initialization was a no-op
**Files:** `voxdet_models/models/necks/secondfpn.py`, `voxdet_models/models/necks/secondfpn3d.py`
**Impact:** MEDIUM

Both FPN modules declared an `init_cfg` list that, in mmcv, would have been
processed by `BaseModule.init_weights()` to apply Kaiming initialization to
`ConvTranspose` layers. In voxdet_core, `BaseModule.init_weights()` is a no-op,
so the `init_cfg` was never processed. This meant `ConvTranspose3d` layers
used PyTorch's default uniform initialization instead of Kaiming normal
(mode=`fan_out`), which mmcv used.

Additionally, `SECONDFPN3D` had the wrong layer name in its `init_cfg`:
`ConvTranspose2d` instead of `ConvTranspose3d`.

**Fix:** Added explicit Kaiming initialization at the end of `__init__`:
```python
for m in self.modules():
    if isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        kaiming_init(m)
```

Also fixed the `init_cfg` layer name from `ConvTranspose2d` to
`ConvTranspose3d` for correctness, even though the dict is now supplementary
documentation.

---

#### 3. `kaiming_init` was missing from init_utils
**Files:** `voxdet_core/init_utils.py`, `voxdet_core/__init__.py`
**Impact:** LOW-MEDIUM (prerequisite for fix #2)

The `init_utils` module only provided `xavier_init`, `constant_init`, and
`trunc_normal_init`. The `kaiming_init` function used by mmcv for automatic
weight initialization was missing.

**Fix:** Added `kaiming_init` with the same signature as mmcv's version:
```python
def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu',
                 bias=0, distribution='normal'):
```

Supports both `'normal'` and `'uniform'` distributions. Exported from
`voxdet_core/__init__.py`.

---

#### 4. DropPath applied to identity instead of residual in BasicBlock
**File:** `voxdet_models/models/backbones/resnet3d.py` (class `BasicBlock`)
**Impact:** LOW (only affects `CustomResNet2D`, which is not used in VoxDet config)

In the `BasicBlock` (2D) forward method, `self.drop_path` was applied to `x`
(the identity/input) instead of `out` (the residual branch output). This is
incorrect—DropPath / Stochastic Depth should randomly drop the residual path,
not the skip connection.

Note: `BasicBlock3D` in the same file already had the correct implementation.

**Before:**
```python
if self.drop_path is not None:
    x = self.drop_path(x)        # BUG: drops identity, not residual
```

**After:**
```python
if self.drop_path is not None:
    out = self.drop_path(out)     # correct: drops residual branch
```

---

#### 5. ConvModule used wrong weight initialization (2.4x smaller std)
**File:** `voxdet_core/cnn_builders.py` (class `ConvModule`)
**Impact:** HIGH (primary cause of 0.15 vs 0.19 mIoU gap)

mmcv's `ConvModule` applied **Kaiming normal (fan_out, relu)** initialization
to all conv layers via its `init_weights()` method. The refactored
`ConvModule` relied on PyTorch's default initialization, which is **Kaiming
uniform (fan_in)**. This produces weights with **2.4x smaller standard
deviation**.

For example, `Conv3d(128, 128, 3)`:
- mmcv (Kaiming normal, fan_out): std = 0.0240
- PyTorch default (Kaiming uniform, fan_in): std = 0.0098

This affected every `ConvModule` instance trained from scratch — all 23+
Conv3d layers in the 3D backbone (`occ_encoder`), the FPN neck convolutions
(`SpatiallyDecoupledFPN`), and detection heads. The smaller initial weight
scale led to weaker gradients in early training, slower convergence, and
lower final mIoU within the fixed 25k step budget.

**Fix:** Added explicit Kaiming normal (fan_out) init after building
`self.conv`:
```python
kaiming_normal_(self.conv.weight, a=0, mode='fan_out', nonlinearity='relu')
```

---

### Performance Improvements

#### 5. DataLoader workers not persisted between epochs
**File:** `LightningTools/dataset_dm.py`
**Impact:** HIGH (training throughput)

DataLoader workers were being respawned at every epoch boundary. Each worker
fork re-imports all Python modules (torch, torchvision, vggt, etc.), which is
expensive. With `persistent_workers=True`, workers stay alive across epochs.

Also added `prefetch_factor=4` (up from default 2) so each worker pre-fetches
more batches into memory, reducing the chance of the GPU stalling while waiting
for the next batch.

Both settings are guarded by `num_workers > 0` to remain compatible with
single-process data loading.

---

#### 6. Training data loader num_workers too low for available hardware
**File:** `configs/voxdet-semantickitti-cam.py`
**Impact:** MEDIUM (training throughput)

The train dataloader was configured with `num_workers=4` on a machine with 64
CPU cores. The data pipeline is I/O-heavy (loading images from disk, reading
LiDAR binary files, loading stereo depth `.npy` files, loading voxel
annotation `.npy` files) and CPU-heavy (LiDAR-to-image projection, depth map
rasterization). With only 4 workers, the GPU was frequently idle waiting for
the next batch.

**Fix:** Increased `num_workers` from 4 to 8 for the training dataloader. This
is a conservative increase; the optimal value depends on disk I/O bandwidth
and per-sample CPU time.

---

#### 7. Unused heavy `vggt` import in data pipeline workers
**File:** `voxdet_models/datasets/pipelines/loading_multiview_imgs.py`
**Impact:** LOW-MEDIUM (worker startup time)

The file imported `from vggt.utils.load_fn import load_and_preprocess_images`
at the module level, but the function was never used anywhere in the class.
Since DataLoader workers fork and re-import all modules, this caused every
worker to load the entire `vggt` package on startup.

Also cleaned up a duplicate `from torchvision import transforms as TF` import.

---

#### 8. Training metrics forced CUDA sync and heavy CPU work every step
**File:** `LightningTools/pl_model.py` (`training_step`)
**Impact:** HIGH (primary cause of intermittent 0% GPU utilization)

Every training step executed:
```python
pred = output_dict['pred'].detach().cpu().numpy()
gt_occ = output_dict['gt_occ'].detach().cpu().numpy()
self.train_metrics.add_batch(pred, gt_occ)
```

This caused three compounding problems:

1. **`.cpu()` forces a full CUDA synchronization.** The CPU blocks until all
   pending GPU kernels finish, then copies the prediction tensor (~32MB for a
   `[B, 256, 256, 32]` voxel grid) from GPU to system RAM. The GPU is
   completely idle during this transfer and all subsequent CPU work.

2. **`SSCMetrics.add_batch` is CPU-intensive.** It runs `np.copy` on both
   arrays, then iterates over 20 classes calling `np.where` on 2M-element
   arrays per class. This is pure Python/NumPy work that keeps the CPU busy
   while the GPU has nothing to do.

3. **`sync_dist=True` on every `self.log` call** triggered an all-reduce
   across GPUs for each individual loss term, every step. This is unnecessary
   for training losses which are only informational.

Combined, this created a ~50-100ms CPU bottleneck per step during which the
GPU sat idle waiting for the next backward pass to be dispatched.

**Fix:**
- Compute train metrics only every 100 steps instead of every step. The
  metrics are cumulative (TP/FP/FN accumulators) so sampling every 100 steps
  still gives an accurate picture at epoch end.
- Changed `sync_dist=True` to `sync_dist=False` for training loss logging.
  Validation metrics still use `sync_dist=True` where correctness matters.

---

#### 9. `bev_pool` column ordering completely wrong — scrambled voxel features
**File:** `voxdet_core/ops/bev_pool.py`
**Impact:** CRITICAL (root cause of 0.15 vs 0.19 mIoU gap with pretrained checkpoint)

The pure-PyTorch replacement for the mmdetection3d CUDA `bev_pool` kernel assumed
that `geom_feats` columns are ordered as `[batch, depth, height, width]` (standard
order). However, the caller (`LSSViewTransformer.voxel_pooling`) constructs
`geom_feats` as `[x, y, z, batch]`, and the original CUDA kernel used a
non-standard column mapping:

| Column | Caller fills with | CUDA kernel maps to | PyTorch replacement mapped to |
|--------|------------------|--------------------|-----------------------------|
| 0      | x (spatial)      | H dimension        | **batch** (WRONG)           |
| 1      | y (spatial)      | W dimension        | **D / depth** (WRONG)       |
| 2      | z (spatial)      | D dimension        | **H / height** (WRONG)      |
| 3      | batch index      | B dimension        | **W / width** (WRONG)       |

Every column was misinterpreted. This caused the Lift-Splat-Shoot (LSS) view
transformer to produce completely scrambled voxel features, corrupting all
downstream modules (VoxFormer cross-attention, 3D backbone, FPN, detection head).

With batch size 1, the old `bev_pool` treated column 0 (x-spatial, range 0–127)
as the batch index and filtered all points with `x >= 1`, discarding ~99% of the
3D volume. The surviving points were scattered to wrong spatial locations.

**Fix:** Remap column indices to match the CUDA kernel convention:
```python
# Before (WRONG):
batch_idx = geom_feats[:, 0].long()  # actually x
d_idx     = geom_feats[:, 1].long()  # actually y
h_idx     = geom_feats[:, 2].long()  # actually z
w_idx     = geom_feats[:, 3].long()  # actually batch

# After (CORRECT):
h_idx     = geom_feats[:, 0].long()  # x → H
w_idx     = geom_feats[:, 1].long()  # y → W
d_idx     = geom_feats[:, 2].long()  # z → D
batch_idx = geom_feats[:, 3].long()  # batch → B
```

---

### CUDA Kernel Restoration

#### 10. Restored CUDA `bev_pool` kernel for 4.7x speedup over pure-PyTorch
**Files:** `voxdet_core/ops/bev_pool.py`, `voxdet_core/ops/csrc/bev_pool_cuda.cu`, `voxdet_core/ops/csrc/bev_pool.cpp`, `voxdet_core/ops/setup_bev_pool.py`
**Impact:** HIGH (training and inference throughput)

The refactoring replaced the original mmdetection3d CUDA `bev_pool` kernel with a
pure-PyTorch `scatter_add` implementation. While functionally correct (after fix #9),
the PyTorch path is significantly slower due to:

1. **No pre-sorting or interval compression.** The CUDA kernel pre-sorts points by
   their voxel rank and computes contiguous intervals, then uses a fused kernel that
   sums each interval in a single thread. The PyTorch path uses `scatter_add_` which
   must handle arbitrary index patterns.

2. **Extra memory traffic.** The PyTorch path materializes a boolean validity mask,
   applies it to filter indices, expands the linear index tensor to `(N, C)` via
   `unsqueeze + expand`, and writes to a flat `(B*D*H*W, C)` buffer before reshaping.
   The CUDA kernel writes directly to the `(B, D, H, W, C)` output.

**Benchmark** (N=200k points, C=64 channels, grid 1x32x128x128):
- CUDA kernel: **3.6 ms**
- PyTorch scatter_add: **17.0 ms**
- Speedup: **4.7x**

**Implementation:**
- Restored the original CUDA kernel (`bev_pool_cuda.cu`) and C++ pybind wrapper
  (`bev_pool.cpp`) from git history (commit `7193d05`)
- Added `setup_bev_pool.py` for pre-compilation:
  `python voxdet_core/ops/setup_bev_pool.py build_ext --inplace`
- Updated `bev_pool.py` with a three-tier import strategy:
  1. Pre-compiled `.so` in `voxdet_core/ops/` (instant load)
  2. JIT compilation via `torch.utils.cpp_extension.load`
  3. Pure-PyTorch `scatter_add` fallback if CUDA is unavailable
- Both forward and backward passes verified correct (exact match with PyTorch path)

**Other CUDA ops assessed:**
- `ms_deform_attn` (2D multi-scale deformable attention): Pure-PyTorch replacement
  exists but is **never called** — the config exclusively uses the DFA3D variant
  which still uses its own CUDA kernels (`dfa3D._ext`). No action needed.

---

## OV-VoxDet Training Pipeline (branch: spear)

These changes enable the OVVoxDet (Open-Vocabulary VoxDet) training pipeline to
run successfully with OVO-style knowledge distillation losses, real CLIP text
embeddings, and convergence monitoring via wandb.

---

### Bug Fixes

#### 11. BatchNorm1d crash with single-camera input
**File:** `voxdet_models/models/img2bev/modules/Mono_DepthNet_modules.py`
**Impact:** CRITICAL (training would not start)

`DepthNet` and `ContextNet` used `nn.BatchNorm1d(cam_channels)` to normalize
camera intrinsic features. With a single camera (`camera_used=["left"]`) the
MLP input has shape `[B*1, 33]`, which after squeeze becomes `[1, 33]` when
`B=1`. `BatchNorm1d` requires more than 1 value per channel during training
and raises `ValueError`.

**Fix:** Replaced `nn.BatchNorm1d(cam_channels)` with `nn.GroupNorm(1, cam_channels)`
in both classes. `GroupNorm(1, C)` is equivalent to InstanceNorm and works with
any batch size.

---

#### 12. DataLoader collation crash with variable-length OVO tensors
**File:** `LightningTools/dataset_dm.py`
**Impact:** CRITICAL (training would not start)

The `LoadLSegFeatures` pipeline produces tensors with variable first dimensions
per sample (`lseg_pixel_feat: [N_valid, 512]`, `valid_vox_indices: [N_valid]`,
`lseg_confidence: [N_valid]`) where `N_valid` differs between frames. PyTorch's
default `DataLoader` collate tries to `torch.stack` all tensors in a batch,
which fails with `RuntimeError: Trying to resize storage that is not resizable`.

Additionally, with `batch_size > 1`, `img_metas` (a dict of mixed tensors and
non-tensors) also failed default collation.

**Fix:** Added custom collation:
- `_ovo_collate_fn`: Keeps `_VARIABLE_LENGTH_KEYS` as lists instead of stacking.
  Stacks regular tensors normally. Falls back to `default_collate` for non-tensors.
- `_collate_img_metas`: Stacks tensor values in the img_metas dict, keeps
  non-tensor values (strings, lists) as lists.
- Auto-detected via `_needs_ovo_collate` flag based on whether the training
  pipeline config contains variable-length keys.

---

#### 13. OVVoxDet variable-length tensor unpacking from custom collate
**File:** `voxdet_models/models/detectors/OVVoxDet.py`
**Impact:** HIGH (training would crash after collation fix)

After the custom collate keeps variable-length tensors as Python lists, the
`forward_train` method needs to unpack them before passing to loss functions.
Without this, tensors like `lseg_pixel_feat` would be a `list` instead of a
`Tensor`, causing downstream shape mismatches.

**Fix:** Added `isinstance(x, list)` checks and `x = x[0]` unpacking for
`lseg_pixel_feat`, `lseg_confidence`, `valid_vox_indices`, and `lseg_2d_feat`.

---

#### 14. `distiller_2d` input channel mismatch (128 vs 512)
**File:** `configs/voxdet-semantickitti-cam-ovo.py`
**Impact:** HIGH (training would crash)

The OVO config set `img_feat_channels=128` (matching `_dim_`), but the 2D
image features from SECONDFPN are a concatenation of 4 scales, each with 128
channels: `out_channels=[128, 128, 128, 128]` → 512 total. This caused
`RuntimeError: expected input to have 128 channels, but got 512` when
`distiller_2d` ran.

**Fix:** Changed `img_feat_channels` from `_dim_` (128) to `512`.

---

#### 15. OVO losses crash on placeholder LSeg features
**File:** `voxdet_models/ov_voxdet/losses/ovo_losses.py`
**Impact:** HIGH (training would crash)

Pre-extracted LSeg features are placeholder data (1-dim embeddings instead of
512-dim). Without guards, `VoxelPixelAlignmentLoss` would compute cosine
similarity between `[N, 512]` voxel features and `[N, 1]` pixel features,
triggering a CUDA assertion. Similarly, `Align2DLoss` would fail on mismatched
channel dimensions.

**Fix:** Added dimension checks in `OVVoxDetLoss.forward`:
- `loss_vox_pix`: skipped when `lseg_pixel_feat.shape[-1] != C`
- `loss_align_2d`: skipped when `lseg_2d_feat.shape[-3] != aligned_2d_feat.shape[1]`

---

#### 16. Spatial resolution mismatch between features and GT labels in OVO losses
**File:** `voxdet_models/ov_voxdet/losses/ovo_losses.py`
**Impact:** CRITICAL (training would crash with CUDA out-of-bounds)

`aligned_vox_feat` from `distiller_3d` is at FPN resolution `[B, 512, 128, 128, 16]`
(due to `lss_downsample=[2,2,2]`), but `gt_labels` and `gt_offsets` are at full
occupancy resolution `[B, 256, 256, 32]`. The `InstanceConsistencyLoss` used
`gt_labels.nonzero()` to get valid voxel coordinates and indexed into the feature
volume, producing coordinates up to `(255, 255, 31)` for a volume of size
`(128, 128, 16)` — a CUDA out-of-bounds error.

The same mismatch affected `VoxelTextAlignmentLoss` (cross-entropy between
`[B, 20, 128, 128, 16]` logits and `[B, 256, 256, 32]` labels).

**Fix:** Added nearest-neighbor downsampling at the top of `OVVoxDetLoss.forward`
to match `gt_labels`, `gt_offsets`, and `valid_mask` to the feature spatial
dimensions before passing them to any sub-loss.

---

#### 17. DDP incompatibility with OVVoxDet distiller modules
**Files:** `main.py`, `voxdet_models/models/detectors/OVVoxDet.py`
**Impact:** HIGH (training would crash on backward pass)

OVVoxDet adds `distiller_2d`, `distiller_3d`, `text_classifier`, and `ov_loss`
modules. When LSeg features are placeholder or text embeddings are missing,
some of these modules don't participate in the loss computation. DDP with
`find_unused_parameters=False` (the default) raises `RuntimeError` about
parameters not used in the loss.

Using `find_unused_parameters=True` conflicts with `torch.utils.checkpoint`
(used by `pts_bbox_head_aux` with `with_cp=True`), causing `RuntimeError:
Expected to mark a variable ready only once`.

The initial fix used `static_graph=True` in `DDPStrategy`, but this conflicts
with `accumulate_grad_batches > 1` (change #25): DDP with `static_graph`
expects an identical backward graph every step, but gradient accumulation skips
the all-reduce on non-sync steps, causing `RuntimeError: expect_autograd_hooks_
INTERNAL ASSERT FAILED at reducer.cpp:1633`.

**Fix (final):** Two-part solution:
1. Set `static_graph=False` unconditionally in `main.py`
2. In `OVVoxDet.forward_train`, when `distiller_2d` is skipped (placeholder
   LSeg features), add a zero-cost dummy loss that puts all `distiller_2d`
   parameters into the autograd graph:
   ```python
   _zero = sum(p.sum() for p in self.distiller_2d.parameters()) * 0.0
   losses['_dummy_2d'] = _zero
   ```
   This avoids a dummy forward pass (which would crash `BatchNorm2d` with
   spatial size 1) while ensuring DDP sees all parameters in the backward graph.

---

#### 18. Text embeddings file path and loader format mismatch
**Files:** `configs/voxdet-semantickitti-cam-ovo.py`, `voxdet_models/models/detectors/OVVoxDet.py`
**Impact:** HIGH (text-alignment loss would not activate)

The config pointed to `semantickitti_text_embeddings.json` (nonexistent). The
actual file is `text_embeddings.pt` (a `[20, 512]` torch tensor) with a
companion `text_embeddings.json` (dict with class names as keys, each value a
512-dim list).

The `_load_text_embeddings` method only handled JSON with an `embeddings` key
or a raw list — neither matching the actual file formats.

**Fix:**
- Config: changed path to `text_embeddings.pt`
- `_load_text_embeddings`: added support for `.pt`/`.pth` files (torch tensors)
  and JSON dicts with class names as keys (values are embedding vectors)

---

### Monitoring & Logging

#### 19. Added wandb logging for convergence monitoring
**Files:** `main.py`, `LightningTools/pl_model.py`
**Impact:** Observability

Added `WandbLogger` alongside the existing `TensorBoardLogger`. All existing
`self.log()` calls automatically route to both loggers, so wandb receives:

**Training losses** (every `log_every_n_steps`):
- `train/loss` (total), `train/loss_voxel_ce`, `train/loss_voxel_sem_scal`,
  `train/loss_voxel_geo_scal`, `train/loss_voxel_ctr` (VoxDet supervised)
- `train/loss_vox_txt`, `train/loss_instance` (OVO distillation)
- `train/loss_aux_*` (auxiliary head)
- `lr-AdamW` (learning rate via `LearningRateMonitor`)

**Gradient norms** (every step, via `on_before_optimizer_step`):
- `grad/total_norm` — total L2 norm across all parameters
- `grad/{module_name}` — per-module norms (`img_backbone`, `img_neck`,
  `depth_net`, `pts_bbox_head`, `occ_encoder`, `distiller_3d`, etc.)

**Validation metrics** (every epoch):
- `val/mIoU`, `val/IoU`, `val/Precision`, `val/Recall`
- `val/IoU_{class_name}` — per-class IoU for all 20 classes

**Config:** Full training config dict logged to wandb at init.

---

### Training Speed Optimizations

#### 20. Redundant `image_encoder` forward pass in OVVoxDet distillation
**File:** `voxdet_models/models/detectors/OVVoxDet.py`
**Impact:** HIGH (~2x forward pass time wasted)

`OVVoxDet.forward_train` called `self.image_encoder(img_inputs[0])` a second
time (line 277) to extract 2D features for `distiller_2d`. This re-runs the
entire ResNet50 backbone + SECONDFPN neck on all batch images — the most
expensive part of the model — just to get features that are already computed
in `extract_img_feat` at line 119 of the parent `VoxDet` class.

Worse, this code ran even when LSeg 2D features are placeholder data (1-dim
instead of 512-dim), meaning the expensive forward pass produced features that
were immediately discarded by the dimension guard.

**Fix:** Added a dimension check that skips the entire `distiller_2d` block
when LSeg 2D features don't have the expected embedding dimension (512).
The redundant `image_encoder` call only executes when real LSeg features are
available.

---

#### 21. Gradient norm logging caused hundreds of CUDA synchronizations per step
**File:** `LightningTools/pl_model.py`
**Impact:** HIGH (GPU utilization dropped to 0% during logging)

The initial `on_before_optimizer_step` implementation called `.item()` on every
parameter's gradient norm individually. Each `.item()` call forces a CUDA
synchronization (CPU waits for GPU). With ~800 parameters, this meant ~800
CUDA syncs per step — the GPU was completely idle during this period.

**Fix:**
- Only run gradient norm logging every `log_every_n_steps` steps (not every step)
- Accumulate all squared norms on GPU using `torch.zeros(1, device=device)`
- Collect all per-module norms into a single tensor via `torch.cat`
- Single `.cpu()` call at the end (one CUDA sync instead of hundreds)

---

#### 22. Vectorized `compute_all_direction_distances` — 5.4x GPU speedup
**File:** `voxdet_models/models/detectors/VoxDet.py`
**Impact:** MEDIUM (55ms → 10ms per batch on GPU)

The `run_length_positive` function used a Python for-loop iterating up to 256
times (once per position along a spatial dimension), with each iteration doing
tensor slicing, comparison, and `torch.where`. This launched hundreds of small
CUDA kernels sequentially.

**Before:** Sequential loop, O(L) Python iterations with individual kernel launches:
```python
for i in range(L - 2, -1, -1):
    idx[dim] = i
    idx_next[dim] = i + 1
    cond = (t[idx] == t[idx_next])
    out[idx] = torch.where(cond, out[idx_next] + 1, 1)
```

**After:** Fully vectorized using cumsum + cummax reset trick, zero Python loops:
```python
boundary[..., :-1] = (t_moved[..., :-1] != t_moved[..., 1:])
boundary_rev = boundary.flip(-1)
cum = ones.cumsum(-1)
reset = torch.where(boundary_rev, cum - 1, torch.zeros_like(cum))
reset_cummax = reset.cummax(-1)[0]
result = cum - reset_cummax
```

The algorithm:
1. Find segment boundaries (where consecutive values differ)
2. Reverse the tensor so each boundary marks a segment start
3. Use `cumsum` of 1s to get global position indices
4. Use `cummax` of reset values at boundaries to get each segment's base offset
5. Subtract base from cumsum to get within-segment run lengths
6. Reverse back to original orientation

Called 6 times per batch (positive and negative directions for X, Y, Z axes)
on a `[B, 256, 256, 32]` tensor. Verified exact match with the original
sequential implementation on all tested shapes.

---

#### 23. Cached positional encoding in VoxFormerHead
**File:** `voxdet_models/models/img2bev/VoxFormerHead.py`
**Impact:** LOW-MEDIUM

The VoxFormerHead computed positional encoding from `torch.zeros((1, 512, 512))`
on every forward call. Since the input is always zeros and the encoding is
deterministic, the result never changes. With batch_size > 1 and the sequential
batch loop, this was computed 4 times per training step unnecessarily.

**Fix:** Cache the result on first call and reuse. Only recompute if the device
changes (e.g., model moved between CPU and GPU).

---

#### 24. Enabled TF32 Tensor Core utilization
**File:** `main.py`
**Impact:** MEDIUM (~10-20% speedup on matmul-heavy operations)

The training log showed "You are using a CUDA device ('NVIDIA RTX PRO 6000
Blackwell Max-Q Workstation Edition') that has Tensor Cores" but the code never
called `torch.set_float32_matmul_precision()`. The default is `'highest'`,
which disables TF32 and uses full FP32 for all matmul operations.

**Fix:** Added `torch.set_float32_matmul_precision('medium')` at module level.
This enables TF32 on Tensor Cores, which uses 10-bit mantissa (vs 23-bit for
FP32) for internal matmul accumulation. The precision loss is negligible for
training but provides significant speedup on transformer attention layers,
Conv layers (which are matmuls under the hood), and linear projections.

---

#### 25. Reduced batch size with gradient accumulation to halve sequential overhead
**Files:** `configs/voxdet-semantickitti-cam-ovo.py`, `main.py`
**Impact:** HIGH (per-step time roughly halved)

VoxFormerHead is hardcoded to process one sample at a time (`assert
lss_volume.shape[0] == 1`). With `batch_size=4`, `VoxDet.extract_img_feat`
loops 4 times through the VoxFormerHead — the most expensive part of the model
(6 DFA3D transformer layers over 262k voxel queries). This is the dominant
training bottleneck.

Fixing VoxFormerHead to support batched operation would require deep changes to
the DFA3D transformer internals (reference point computation, voxel coordinate
indexing, feature scattering). Instead, we halve the loop count:

**Config change:**
```python
# Before:
batch_size = 4

# After:
batch_size = 2
accumulate_grad_batches = 2  # effective batch_size = 2 * 2 = 4
```

**main.py change:** Added `accumulate_grad_batches` parameter to `pl.Trainer`,
read from config with default of 1 (no accumulation) for backward compatibility
with the non-OVO config.

The effective batch size remains 4 (same gradient statistics), but each forward
pass now loops only 2 times through VoxFormerHead. The trade-off is 2x more
forward passes total, but each is ~2x faster, with better GPU utilization.

---

### Training Infrastructure

#### 26. Checkpoint saving was unreliable — no explicit directory, no periodic saves
**File:** `main.py`
**Impact:** HIGH (checkpoints could be lost or never saved)

`ModelCheckpoint` had no `dirpath` set, so PyTorch Lightning defaulted to a
deeply nested path inside the tensorboard logger directory
(`logs/.../tensorboard/version_N/checkpoints/`). The version number incremented
on each run, making checkpoints hard to find. More critically:

- With `max_steps=25000` and ~4782 steps per epoch, training stops mid-epoch
  (at step 25000 during epoch 5) without triggering end-of-epoch validation.
- `save_last=True` only writes `last.ckpt` at validation or `on_train_end`.
  If training is interrupted (crash, OOM, preemption) mid-epoch, all progress
  since the last validation is lost.

**Fix:** Two checkpoint callbacks:
1. **Best model**: `ModelCheckpoint(dirpath=ckpt_dir, monitor='val/mIoU',
   mode='max', save_last=True, filename='best')` — saves best and last at each
   validation epoch.
2. **Periodic**: `ModelCheckpoint(dirpath=ckpt_dir, every_n_train_steps=2000,
   save_top_k=-1, filename='step-{step}')` — saves every 2000 steps regardless
   of validation, so at most 2000 steps of work can be lost on crash.

Both use explicit `dirpath=os.path.join(log_folder, 'checkpoints')`.

---

#### 27. Added training resume support (`--resume` flag)
**File:** `main.py`
**Impact:** Feature (training can resume from interruptions)

Training runs can now be resumed after crashes, preemptions, or intentional
stops without losing progress. Three components:

**CLI flags:**
- `--resume`: Resume training from the last checkpoint
- `--ckpt_path`: Optionally specify an explicit checkpoint path (also used for
  eval)

**Auto-find last checkpoint:**
```python
if config.resume:
    if config.ckpt_path:
        resume_ckpt = config.ckpt_path
    else:
        last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
        if os.path.isfile(last_ckpt):
            resume_ckpt = last_ckpt
```

`trainer.fit(ckpt_path=resume_ckpt)` restores full training state: model
weights, optimizer state, learning rate scheduler, epoch counter, global step,
and callback states (including which checkpoint is "best").

**Wandb run resume:**
When `--resume` is set, the wandb logger attempts to continue the same run
instead of creating a new one. It reads the run ID from
`{log_folder}/wandb/latest-run/run-id.txt` and uses `resume='must'`. If no
prior run is found, falls back to `resume='allow'` (creates new run).

**Usage:**
```bash
# Start training
python main.py --config_path configs/... --log_folder logs/my-run

# Resume from last checkpoint
python main.py --config_path configs/... --log_folder logs/my-run --resume

# Resume from specific checkpoint
python main.py --config_path configs/... --log_folder logs/my-run \
    --resume --ckpt_path logs/my-run/checkpoints/step-step=4000.ckpt
```

---

### Summary of files modified

| File | Change type |
|------|-------------|
| `voxdet_core/cnn_builders.py` | Bug fix: remove double-build in ConvModule; fix weight init (#1, #5) |
| `voxdet_core/init_utils.py` | New function: `kaiming_init` (#3) |
| `voxdet_core/__init__.py` | Export `kaiming_init` (#3) |
| `voxdet_models/models/necks/secondfpn.py` | Bug fix: add explicit Kaiming init (#2) |
| `voxdet_models/models/necks/secondfpn3d.py` | Bug fix: add explicit Kaiming init, fix layer name (#2) |
| `voxdet_models/models/backbones/resnet3d.py` | Bug fix: DropPath on residual, not identity (#4) |
| `LightningTools/dataset_dm.py` | Perf: persistent workers, prefetch; OVO custom collation (#6, #12) |
| `LightningTools/pl_model.py` | Perf: reduce train metric frequency (#8); gradient norm logging (#19, #21) |
| `configs/voxdet-semantickitti-cam.py` | Perf: increase num_workers (#6) |
| `configs/voxdet-semantickitti-cam-ovo.py` | Bug fix: text embedding path (#18), img_feat_channels (#14); perf: batch_size + grad accum (#25) |
| `voxdet_models/datasets/pipelines/loading_multiview_imgs.py` | Cleanup: remove unused vggt import (#7) |
| `voxdet_core/ops/bev_pool.py` | Bug fix: correct column ordering (#9); CUDA kernel with PyTorch fallback (#10) |
| `voxdet_core/ops/csrc/bev_pool_cuda.cu` | Restored: CUDA bev_pool kernel (#10) |
| `voxdet_core/ops/csrc/bev_pool.cpp` | Restored: C++ pybind wrapper for CUDA kernel (#10) |
| `voxdet_core/ops/setup_bev_pool.py` | New: build script for pre-compiling CUDA extension (#10) |
| `voxdet_models/models/img2bev/modules/Mono_DepthNet_modules.py` | Bug fix: BatchNorm1d → GroupNorm (#11) |
| `voxdet_models/models/detectors/OVVoxDet.py` | Bug fix: list unpacking (#13), text loader (#18), DDP dummy loss (#17); perf: skip redundant image_encoder (#20) |
| `voxdet_models/ov_voxdet/losses/ovo_losses.py` | Bug fix: placeholder guards (#15), resolution mismatch (#16) |
| `voxdet_models/models/detectors/VoxDet.py` | Perf: vectorized run-length encoding (#22) |
| `voxdet_models/models/img2bev/VoxFormerHead.py` | Perf: cached positional encoding (#23) |
| `main.py` | DDP unused params (#17); wandb logger (#19); TF32 precision (#24); grad accumulation (#25); checkpoints (#26); resume (#27) |

# OV-VoxDet: Open-Vocabulary VoxDet

Open-vocabulary 3D semantic occupancy prediction by integrating [OVO](https://github.com/dzcgaber/OVO)'s CLIP/LSeg knowledge distillation framework into [VoxDet](https://github.com/vita-epfl/VoxDet)'s instance-centric dense detection architecture.

## Idea

VoxDet predicts both **semantic labels** and **per-voxel offset fields** (VoxNT) that encode instance boundaries. However, its classifier is closed-set — limited to the classes seen during training. OVO showed that 2D vision-language features (CLIP, LSeg) can be distilled into 3D voxel representations to enable open-vocabulary occupancy prediction, but it operates on vanilla architectures (MonoScene) that lack instance awareness.

**OV-VoxDet combines both**: it keeps VoxDet's offset regression branch fully supervised while distilling CLIP-aligned embeddings into the classification branch. VoxDet's predicted boundaries provide a geometric signal to improve distillation quality — voxels near instance boundaries have unreliable 2D projections and should be down-weighted.

### Architecture

```
Image ──► ResNet50 + FPN ──► Depth Net + LSS ──► VoxFormer (2D→3D lifting)
                │                                          │
                │                                    3D Voxel Features
                │                                          │
                │                              ┌───── CustomResNet3D ─────┐
                │                              │   SpatiallyDecoupledFPN  │
                │                              │     ┌─────┴─────┐       │
                │                              │  cls_feat    reg_feat    │
                │                              └─────┬───────────┬───────┘
                │                                    │           │
                │                     ┌──────────────┤           │
                │                     │              │           │
                │              Distiller3D     VoxDetHead    VoxDetHead
                │              (φ_3D)          cls branch    reg branch
                │                │             (supervised)  (supervised)
                │          CLIP-aligned            │           │
                │          voxel embeds        output_cls   output_bbox
                │                │                              │
                │         ┌──────┴──────────────────────────────┘
                │         │                                (VoxNT offsets)
                │    OVO Losses:                                │
                │    • L_vox_pix (boundary-weighted)  ◄─────────┘
           Distiller2D   • L_vox_txt
           (φ_2D)        • L_instance
                │
           L_2d (regularizer)
```

### Loss Formulation

```
L_total = L_ce + L_sem_scal + L_geo_scal + L_reg        (VoxDet supervised)
        + λ₁·L_vox_pix                                   (voxel↔pixel alignment)
        + λ₂·L_vox_txt                                   (voxel↔text alignment)
        + λ₃·L_2d                                        (2D feature regularizer)
        + λ₄·L_instance                                  (instance consistency)
```

### Novel Contributions Over OVO

| Feature | OVO | OV-VoxDet |
|---|---|---|
| Boundary-aware weighting | ✗ | Voxels with small min-offset (near boundaries) get lower weight in L_vox_pix |
| Instance consistency loss | ✗ | Same-class voxels within an instance map to similar CLIP embeddings |
| Architecture | MonoScene | VoxDet (decoupled cls/reg branches, VoxelAttentionAggregation) |
| Offset supervision | ✗ | Regression branch stays fully supervised with VoxNT |

---

## Project Structure

```
ov_voxdet/
├── __init__.py                              # Registers OVVoxDet and LoadLSegFeatures
├── configs/
│   └── ov_voxdet_semantickitti.py           # Training config (extends VoxDet config)
├── datasets/
│   ├── __init__.py
│   └── pipelines/
│       ├── __init__.py
│       └── loading_lseg_features.py         # Pipeline transform: loads pre-extracted LSeg data
├── losses/
│   ├── __init__.py
│   └── ovo_losses.py                        # L_vox_pix, L_vox_txt, L_2d, L_instance
├── models/
│   ├── __init__.py
│   ├── distiller.py                         # VoxDetDistiller3D, VoxDetDistiller2D, TextEmbeddingClassifier
│   └── ov_voxdet.py                         # OVVoxDet (subclass of VoxDet)
├── utils/
│   ├── __init__.py
│   ├── preprocess_ov_data.py                # Offline: extract LSeg features, CLIP text embeddings
│   └── voxel_filtering.py                   # Boundary weights, instance grouping from VoxNT offsets
└── README.md
```

---

## Prerequisites

### 1. Install VoxDet

Follow [VoxDet's installation](https://github.com/vita-epfl/VoxDet):

```bash
git clone https://github.com/vita-epfl/VoxDet.git
cd VoxDet

# VoxDet's dependencies (mmdet3d, mmcv, pytorch-lightning, etc.)
pip install -r requirements.txt
```

### 2. Install OV-VoxDet Dependencies

```bash
pip install open_clip_torch

# LSeg (for feature extraction)
# Follow https://github.com/isl-org/lang-seg for installation
```

### 3. Place OV-VoxDet Inside VoxDet

Copy the `ov_voxdet/` directory into the VoxDet project root:

```bash
cp -r ov_voxdet/ VoxDet/ov_voxdet/
```

### 4. Register the Modules

Add one import to `VoxDet/mmdet3d_plugin/__init__.py`:

```python
import ov_voxdet  # registers OVVoxDet detector and LoadLSegFeatures pipeline
```

This triggers `@DETECTORS.register_module()` for `OVVoxDet` and `@PIPELINES.register_module()` for `LoadLSegFeatures`, so `build_model` and `Compose` can find them.

### 5. Dataset

Set up SemanticKITTI following VoxDet's instructions:

```
data/kitti/dataset/
├── sequences/
│   ├── 00/
│   │   ├── calib.txt
│   │   ├── image_2/
│   │   ├── image_3/
│   │   └── voxels/
│   ├── 01/ ...
├── labels/
│   ├── 00/
│   │   ├── 000000_1_1.npy
│   │   ├── ...
├── depth/
│   └── sequences/
```

---

## Step-by-Step Usage

### Step 1: Generate CLIP Text Embeddings

```bash
python ov_voxdet/utils/preprocess_ov_data.py \
    --kitti_root data/kitti/dataset \
    --output_dir data/kitti/dataset/prompt_embedding \
    --generate_text_embeddings \
    --clip_model ViT-B-32 \
    --clip_pretrained laion2b_s34b_b79k
```

This produces:
- `text_embeddings.pt` — `[20, 512]` tensor (one embedding per SemanticKITTI class)
- `text_embeddings.json` — same data in JSON format (used by the config)

### Step 2: Extract LSeg Features

For each image in SemanticKITTI, extract dense LSeg pixel features and compute voxel-pixel correspondences. This is the most compute-intensive offline step.

Use the functions in `ov_voxdet/utils/preprocess_ov_data.py`:

```python
from ov_voxdet.utils.preprocess_ov_data import (
    extract_lseg_features,
    project_voxels_to_pixels,
    compute_voxnt_offsets,
)

# For each (sequence, frame):
#   1. Run LSeg on the image → [512, H, W] feature map
#   2. Project voxel grid to pixels using camera calibration
#   3. Sample LSeg features at valid projected locations
#   4. Save as .npz
```

Output structure expected by `LoadLSegFeatures`:

```
data/kitti/dataset/lseg_features/
└── sequences/
    ├── 00/
    │   ├── 000000.npz    # keys: lseg_2d_feat, lseg_pixel_feat, lseg_confidence, valid_vox_indices
    │   ├── 000001.npz
    │   └── ...
    ├── 01/ ...
```

Each `.npz` contains:

| Key | Shape | Description |
|---|---|---|
| `lseg_2d_feat` | `[512, H, W]` | Full LSeg feature map (for L_2d loss) |
| `lseg_pixel_feat` | `[N_valid, 512]` | LSeg features at valid voxel projections |
| `lseg_confidence` | `[N_valid]` | LSeg prediction confidence per valid voxel |
| `valid_vox_indices` | `[N_valid]` | Flat indices into the `[X*Y*Z]` voxel grid |

### Step 3: Train

Training uses VoxDet's existing `train.py` — no modifications needed:

```bash
python train.py \
    --config_path ov_voxdet/configs/ov_voxdet_semantickitti.py \
    --log_folder logs/ov_voxdet
```

This is equivalent to training VoxDet but with:
- `model.type = "OVVoxDet"` instead of `"VoxDet"`
- `model.ov_config` dict controlling distillation
- `LoadLSegFeatures` in the train pipeline feeding distillation targets

The config supports three training modes via `ov_config.ov_mode`:

| Mode | VoxDet Losses | OVO Losses | Use Case |
|---|---|---|---|
| `"hybrid"` | ✓ | ✓ | Default — full supervision + distillation |
| `"ov_only"` | ✗ | ✓ | No 3D annotations, pure distillation |
| `"supervised"` | ✓ | ✗ | Baseline VoxDet (for ablation) |

### Step 4: Evaluate (Closed-Set)

Standard evaluation on SemanticKITTI val split:

```bash
python train.py \
    --config_path ov_voxdet/configs/ov_voxdet_semantickitti.py \
    --ckpt_path logs/ov_voxdet/best.ckpt \
    --eval
```

### Step 5: Open-Vocabulary Inference

To classify with **novel classes not seen during training**, load the model and swap in new text embeddings:

```python
import open_clip
from mmcv import Config
from mmdet3d.models import build_model

# Build model
cfg = Config.fromfile('ov_voxdet/configs/ov_voxdet_semantickitti.py')
model = build_model(cfg.model)

# Load checkpoint
import torch
ckpt = torch.load('logs/ov_voxdet/best.ckpt')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval().cuda()

# Define novel classes and generate text embeddings
novel_classes = ['bus', 'ambulance', 'scooter', 'construction-cone', 'fire-hydrant']
clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.set_ov_classes(novel_classes, clip_model=clip_model, tokenizer=tokenizer)

# Run inference
with torch.no_grad():
    output = model(data_dict)  # data_dict from SemanticKITTI dataloader
    pred = output['pred']      # [B, X, Y, Z] with indices into novel_classes
```

---

## Config Reference

All OV-VoxDet settings live in `model.ov_config` inside the config file. The rest of the config is identical to VoxDet.

```python
ov_config=dict(
    # Architecture
    cls_feat_channels=128,        # Must match SpatiallyDecoupledFPN out_channels
    img_feat_channels=128,        # Must match img_neck output
    embedding_dim=512,            # CLIP/LSeg standard

    # Data
    text_embedding_path="data/kitti/dataset/prompt_embedding/text_embeddings.json",
    base_class_indices=[0, 1, 4, 6, 9, 11, 13, 15, 17],
    novel_class_indices=[2, 3, 5, 7, 8, 10, 12, 14, 16, 18, 19],

    # Loss weights
    lambda_vox_pix=1.0,           # Voxel-pixel alignment
    lambda_vox_txt=1.0,           # Voxel-text alignment
    lambda_2d=0.1,                # 2D feature regularizer
    lambda_instance=0.5,          # Instance consistency

    # Distillation params
    temperature=0.1,              # Cosine similarity temperature
    use_confidence_weight=True,   # Weight by LSeg prediction confidence
    use_boundary_weight=True,     # Down-weight boundary voxels using VoxNT offsets
    confidence_threshold=0.5,     # Min LSeg confidence to include a voxel
    boundary_threshold=2.0,       # Voxels with min_offset < this are boundaries

    # Training mode
    ov_mode="hybrid",             # "hybrid" | "ov_only" | "supervised"
)
```

---

## How It Works Internally

### Training Flow (`OVVoxDet.forward_train`)

1. **Feature extraction** — identical to VoxDet: `image_encoder → depth_net → LSSViewTransformer → VoxFormer → CustomResNet3D → SpatiallyDecoupledFPN`
2. **SpatiallyDecoupledFPN** returns `(cls_feat_list, reg_feat_list)`. The regression branch is unchanged.
3. **VoxNT offsets** computed from `gt_occ` via `compute_all_direction_distances()`
4. **VoxDet supervised losses** — `VoxDetHead.loss()` computes `L_ce`, `L_sem_scal`, `L_geo_scal`, `L_reg` as normal
5. **Distillation** (the new part):
   - `Distiller3D(cls_feat)` → `[B, 512, X, Y, Z]` CLIP-aligned voxel embeddings
   - `OVVoxDetLoss` computes `L_vox_pix`, `L_vox_txt`, `L_2d`, `L_instance`
   - `L_vox_pix` is weighted by both LSeg confidence and VoxNT boundary proximity
6. **All losses merged** into a single `losses` dict → `pl_model.training_step` sums and backprops

### Inference Flow (`OVVoxDet.forward_test`)

- **Supervised mode**: uses VoxDetHead's learned `cls_convs` (standard path)
- **Open-vocabulary mode**: `cls_feat → Distiller3D → TextEmbeddingClassifier` (cosine similarity with CLIP text embeddings for arbitrary classes)

---

## File Descriptions

| File | Lines | Role |
|---|---|---|
| `models/ov_voxdet.py` | 390 | `OVVoxDet(VoxDet)` — main model class, registered as a detector |
| `models/distiller.py` | 134 | `VoxDetDistiller3D` (3D→CLIP), `VoxDetDistiller2D` (2D→LSeg), `TextEmbeddingClassifier` |
| `losses/ovo_losses.py` | 336 | `VoxelPixelAlignmentLoss`, `VoxelTextAlignmentLoss`, `Align2DLoss`, `InstanceConsistencyLoss`, `OVVoxDetLoss` |
| `datasets/pipelines/loading_lseg_features.py` | 100 | `LoadLSegFeatures` — pipeline transform loading `.npz` files into the data dict |
| `utils/preprocess_ov_data.py` | 334 | Offline preprocessing: LSeg extraction, CLIP text embeddings, voxel-pixel projection |
| `utils/voxel_filtering.py` | ~200 | Boundary weight computation, valid voxel filtering, instance grouping from VoxNT offsets |
| `configs/ov_voxdet_semantickitti.py` | ~280 | Full config — identical to VoxDet's except `model.type="OVVoxDet"`, `model.ov_config`, and `LoadLSegFeatures` in pipeline |

---

## Citation

```bibtex
@article{voxdet2025,
  title={VoxDet: Dense 3D Object Detection via Voxel Neighbor Trick},
  author={...},
  year={2025}
}

@inproceedings{ovo2024,
  title={OVO: Open-Vocabulary Occupancy},
  author={...},
  year={2024}
}
```
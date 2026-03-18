# Viewpoint Generalization in Single-Image Object Detection
### 3D-Consistent Synthetic Adaptation vs. 2D Augmentation

**A2S Lab** — Control and Automation Engineering, 2026

---

## Overview

This repository contains the full implementation of a controlled experimental study investigating whether LoRA adapters trained on **Zero123++-generated 3D-consistent synthetic views** produce superior viewpoint-invariant object detection compared to adapters trained on **2D-augmented data** from the same single reference image.

The core question: *does geometric consistency in synthetic training data transfer to viewpoint robustness at inference time?*

---

## Hypothesis

> LoRA adapters fine-tuned on Zero123++-generated multi-view synthetic data produce superior viewpoint-invariant detection compared to adapters fine-tuned on 2D-augmented data from the same single reference image. The primary expected improvement is in rear and rear-side viewpoint bins.

---

## Method

### Three Conditions

| Condition | Method | Training Data |
|-----------|--------|---------------|
| **A (Proposed)** | Zero123++ novel view synthesis | Single image → 96 synthetic views → 400 composited images |
| **B (Baseline)** | 2D augmentation | Single image → rotation, flip, scale, perspective warp → 400 images |
| **C (Reference)** | OWL-ViT zero-shot | No training — text-conditioned detection |

### Detection Protocol

```
test image
→ SAM automatic mask generation
→ region crops
→ DINOv2 base (frozen) + LoRA adapter
→ CLS token embedding
→ cosine similarity to object prototype
→ threshold + NMS
→ detections
```

### Backbone
- **DINOv2 base** (ViT-B/14), fully frozen
- **LoRA**: rank 16, alpha 32, inserted into query and value projections
- **Loss**: pure InfoNCE (τ = 0.07)
- **Training**: LR = 3×10⁻⁵, 30 epochs, effective batch size 16 (4 native + 4 accumulation steps), mixed precision

---

## Dataset

**Google Scanned Objects (GSO)** — 30 objects selected with full 360° azimuth coverage:
- 15 high-texture asymmetric objects (shoes, toys, tools)
- 15 low-texture symmetric objects (bottles, bowls, cylinders)

**Viewpoint bins** (ground-truth camera angles):
| Bin | Azimuth Range |
|-----|--------------|
| Frontal | 0°–45°, 315°–360° |
| Side | 45°–90°, 270°–315° |
| Rear-Side | 90°–135°, 225°–270° |
| Rear | 135°–225° |

---

## Setup

### Requirements
- Python 3.9
- PyTorch 2.4.0 + CUDA 12.1
- PyTorch3D 0.7.8

### Installation

```bash
conda create -n viewpoint_project python=3.9
conda activate viewpoint_project
conda install pytorch=2.4.0 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge --override-channels
conda install pytorch3d -c pytorch3d -c conda-forge --override-channels
pip install transformers==4.41.0 peft==0.9.0 diffusers==0.25.1 accelerate==0.27.0
pip install huggingface_hub==0.21.0
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### Data

Download Google Scanned Objects (IBRNet renderings, 7.5GB):
```bash
mkdir -p /path/to/data/gso_data
cd /path/to/data/gso_data
gdown https://drive.google.com/uc?id=1tKHhH-L1viCvTuBO1xg--B_ioK7JUrrE
unzip google_scanned_objects_renderings.zip
```

Download COCO val2017 (1GB, for background randomization):
```bash
mkdir -p /path/to/data/coco_data
cd /path/to/data/coco_data
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

Update paths in `config.py` to match your local data directories.

---

## Running the Experiment

### Step 1 — Extract reference images
```bash
python scripts/01_extract_alpha.py
```

### Step 2 — Generate Condition A training data
```bash
python scripts/02_generate_zero123.py
```

### Step 3 — Generate Condition B training data
```bash
python scripts/03_augmentation_2d.py
```

### Step 4 — Train LoRA adapters
```bash
python scripts/04_train_lora.py --condition condition_a
python scripts/04_train_lora.py --condition condition_b
```

### Step 5 — Evaluate
```bash
python scripts/05_evaluate.py
```

### Step 6 — OWL-ViT baseline
```bash
python scripts/06_owlvit_baseline.py
```

### Step 7 — Generate report
```bash
python scripts/07_generate_report.py --obj_id <obj_id>
```

### PoC mode (single object)
```bash
python scripts/04_train_lora.py --condition condition_a --obj_id <obj_id>
python scripts/04_train_lora.py --condition condition_b --obj_id <obj_id>
python scripts/05_evaluate.py --obj_id <obj_id>
python scripts/07_generate_report.py --obj_id <obj_id>
```

---

## Current Status

| Step | Status |
|------|--------|
| Environment setup | Complete |
| Dataset download and preparation | Complete |
| Object selection (30 objects) | Complete |
| Alpha extraction | Complete |
| Condition A data generation | Complete |
| Condition B data generation | Complete |
| LoRA training (full) | In progress |
| Threshold sweep | Pending |
| Full evaluation | Pending |
| OWL-ViT baseline | Pending |
| Final paper | Pending |

**PoC result (single object, unoptimized):**

| Viewpoint | Condition A | Condition B |
|-----------|-------------|-------------|
| Frontal | 0.2232 | 0.0025 |
| Side | 0.1011 | 0.0331 |
| Rear-Side | 0.0083 | 0.0005 |
| Rear | 0.0343 | 0.0020 |
| **Overall** | **0.0360** | **0.0023** |

Condition A outperforms Condition B across all viewpoint bins under unoptimized conditions.

---

## Planned Analyses

Beyond per-bin mAP, the full paper will include:

1. **Viewpoint Transfer Matrix** — 2D heatmap of detection score as a function of training azimuth vs test azimuth, comparing angular generalization bandwidth between conditions
2. **Embedding Geometry Analysis** — UMAP visualization of DINOv2 CLS embeddings colored by azimuth, before and after LoRA training, showing whether Zero123++ reshapes the feature space along a viewpoint manifold
3. **Texture Sensitivity Analysis** — correlation between object texture complexity and performance gain (Condition A − B), testing whether synthetic geometry depends on surface appearance cues

---

## Citation

If you use this code, please cite:

```bibtex
@misc{a2slab2026viewpoint,
  title={Viewpoint Generalization in Single-Image Object Detection:
         3D-Consistent Synthetic Adaptation vs. 2D Augmentation},
  author={A2S Lab},
  year={2026}
}
```

---

## Acknowledgements

- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus) — SUDO-AI
- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta AI
- [Segment Anything](https://github.com/facebookresearch/segment-anything) — Meta AI
- [Google Scanned Objects](https://arxiv.org/abs/2204.11918) — Google Research
- [PEFT](https://github.com/huggingface/peft) — HuggingFace
"""
config.py
=========
Central configuration for the viewpoint generalization experiment.
All paths, hyperparameters, and experiment settings are defined here.
Every other script imports from this file — do not hardcode values elsewhere.
"""

import os

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = "/home/enbiya/a2s_project"

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
GSO_ROOT = os.path.join(DATA_ROOT, "gso_data", "google_scanned_objects")
COCO_DIR = os.path.join(DATA_ROOT, "coco_data", "val2017")
MANIFESTS_DIR = os.path.join(DATA_ROOT, "gso_data", "manifests")
SPLITS_FILE = os.path.join(MANIFESTS_DIR, "splits.txt")
HIGH_TEXTURE_FILE = os.path.join(DATA_ROOT, "gso_data", "high_texture_objects.txt")
LOW_TEXTURE_FILE = os.path.join(DATA_ROOT, "gso_data", "low_texture_objects.txt")

TRAINING_DATA_ROOT = os.path.join(DATA_ROOT, "gso_data", "training_data")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")

REPOS_ROOT = os.path.join(PROJECT_ROOT, "repos")
ZERO123_REPO = os.path.join(REPOS_ROOT, "zero123plus")

# ---------------------------------------------------------------------------
# GSO Dataset
# ---------------------------------------------------------------------------

IMAGE_SIZE = 512  # GSO renders are 512x512

# ---------------------------------------------------------------------------
# Object Selection
# ---------------------------------------------------------------------------

def load_object_list(filepath):
    with open(filepath) as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def load_splits():
    """
    Returns dict:
        {
            'val_high': [...],
            'val_low': [...],
            'eval_high': [...],
            'eval_low': [...],
        }
    """
    splits = {'val_high': [], 'val_low': [], 'eval_high': [], 'eval_low': []}
    with open(SPLITS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            split_name, obj_id = line.split(',')
            splits[split_name].append(obj_id)
    return splits

# ---------------------------------------------------------------------------
# Zero123++ Generation
# ---------------------------------------------------------------------------

ZERO123_MODEL_ID = "sudo-ai/zero123plus-v1.2"
ZERO123_PIPELINE_ID = "sudo-ai/zero123plus-pipeline"
ZERO123_INFERENCE_STEPS = 36
ZERO123_NUM_CALLS = 16       # 16 calls x 6 views = 96 base views
ZERO123_VIEWS_PER_CALL = 6   # Zero123++ outputs 6 views per call as tiled image
ZERO123_TOTAL_VIEWS = ZERO123_NUM_CALLS * ZERO123_VIEWS_PER_CALL  # 96

# ---------------------------------------------------------------------------
# 2D Augmentation
# ---------------------------------------------------------------------------

AUG_ROTATION_DEGREES = (0, 360)
AUG_FLIP_PROBABILITY = 0.5
AUG_SCALE_RANGE = (0.8, 1.2)
AUG_COLOR_JITTER = {
    "brightness": 0.3,
    "contrast": 0.3,
    "saturation": 0.3,
    "hue": 0.0
}
AUG_PERSPECTIVE_DISTORTION = 0.3

# ---------------------------------------------------------------------------
# Background Randomization (shared by both conditions)
# ---------------------------------------------------------------------------

BG_BLUR_SIGMA = 2.0
BG_COLOR_JITTER_STRENGTH = 0.2
TARGET_TRAINING_IMAGES = 400  # target per object per condition

# ---------------------------------------------------------------------------
# DINOv2 + LoRA
# ---------------------------------------------------------------------------

DINOV2_MODEL_ID = "facebook/dinov2-base"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["query", "value"]  # attention projection layers

# Training
LEARNING_RATE = 3e-5
LR_SCHEDULER = "cosine"
NUM_EPOCHS = 30
NATIVE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH_SIZE = NATIVE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
MIXED_PRECISION = True

# InfoNCE loss
INFONCE_TEMPERATURE = 0.07
INFONCE_LAMBDA = 1.0  # pure InfoNCE, no additional loss terms

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Cosine similarity thresholds — derived separately per texture group
# Set to None until threshold sweep is complete
THRESHOLD_HIGH_TEXTURE = None
THRESHOLD_LOW_TEXTURE = None

NMS_IOU_THRESHOLD = 0.5
SAM_POINTS_PER_SIDE = 32  # SAM automatic mask generation density

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

VIEWPOINT_BINS = {
    "frontal":   [(0, 45), (315, 360)],
    "side":      [(45, 90), (270, 315)],
    "rear_side": [(90, 135), (225, 270)],
    "rear":      [(135, 225)],
}

MIN_TEST_IMAGES_PER_BIN = 10  # objects with fewer are excluded
MAP_IOU_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

CONDITIONS = ["condition_a", "condition_b", "condition_c"]
CONDITION_NAMES = {
    "condition_a": "Zero123++ (Proposed)",
    "condition_b": "2D Augmentation (Baseline)",
    "condition_c": "OWL-ViT (Reference)",
}

# ---------------------------------------------------------------------------
# Directory Creation Helper
# ---------------------------------------------------------------------------

def make_dirs():
    """Create all output directories if they do not exist."""
    dirs = [
        TRAINING_DATA_ROOT,
        RESULTS_ROOT,
        CHECKPOINTS_ROOT,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------------------------

def verify_paths():
    """Verify all critical data paths exist before running any experiment."""
    critical = [
        GSO_ROOT,
        COCO_DIR,
        MANIFESTS_DIR,
        SPLITS_FILE,
        HIGH_TEXTURE_FILE,
        LOW_TEXTURE_FILE,
    ]
    all_ok = True
    for path in critical:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {path}")
        if not exists:
            all_ok = False
    return all_ok

if __name__ == "__main__":
    print("Verifying paths...")
    ok = verify_paths()
    if ok:
        print("\nAll paths verified.")
        make_dirs()
        print("Output directories created.")
        splits = load_splits()
        for split_name, objs in splits.items():
            print(f"{split_name}: {len(objs)} objects")
    else:
        print("\nSome paths are missing. Fix before proceeding.")
"""
utils/dataset.py
================
Data loading utilities for the viewpoint generalization experiment.
Handles GSO object loading, manifest parsing, COCO background loading,
and alpha compositing.
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import csv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# GSO Object Loading
# ---------------------------------------------------------------------------

def get_i0_frame(obj_id):
    """
    Returns the frame index of I0 — the view closest to azimuth 0
    for the given object.
    """
    obj_path = os.path.join(config.GSO_ROOT, obj_id)
    pose_dir = os.path.join(obj_path, "pose")

    best_idx = None
    best_diff = float("inf")

    for i in range(250):
        pose_path = os.path.join(pose_dir, f"{str(i).zfill(6)}.txt")
        try:
            with open(pose_path) as f:
                values = [float(line.strip()) for line in f.readlines()]
            matrix = np.array(values).reshape(4, 4)
            cam_pos = matrix[:3, 3]
            az = np.degrees(np.arctan2(cam_pos[0], cam_pos[2]))
            if az < 0:
                az += 360
            diff = min(az, 360 - az)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        except Exception:
            continue

    return best_idx


def load_rgba(obj_id, frame_idx):
    """
    Loads RGB + mask for a given object and frame index.
    Returns PIL RGBA image with GSO mask as alpha channel.
    """
    obj_path = os.path.join(config.GSO_ROOT, obj_id)
    rgb = Image.open(
        os.path.join(obj_path, "rgb", f"{str(frame_idx).zfill(6)}.png")
    ).convert("RGB")
    mask = Image.open(
        os.path.join(obj_path, "mask", f"{str(frame_idx).zfill(6)}.png")
    ).convert("L")

    rgba = rgb.copy()
    rgba.putalpha(mask)
    return rgba


def load_i0_rgba(obj_id):
    """
    Loads the I0 reference image for a given object as RGBA.
    """
    frame_idx = get_i0_frame(obj_id)
    return load_rgba(obj_id, frame_idx), frame_idx


# ---------------------------------------------------------------------------
# Manifest Loading
# ---------------------------------------------------------------------------

def load_manifest(obj_id):
    """
    Loads the test manifest CSV for a given object.
    Returns list of dicts with keys:
        object_id, frame, azimuth, elevation, bin
    """
    manifest_path = os.path.join(
        config.MANIFESTS_DIR, f"{obj_id}_manifest.csv"
    )
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["azimuth"] = float(row["azimuth"])
            row["elevation"] = float(row["elevation"])
            rows.append(row)
    return rows


def get_test_frames_by_bin(obj_id):
    """
    Returns dict mapping bin name to list of frame indices.
    Example:
        {
            'frontal': ['000012', '000034', ...],
            'side': [...],
            'rear_side': [...],
            'rear': [...],
        }
    """
    manifest = load_manifest(obj_id)
    bins = {"frontal": [], "side": [], "rear_side": [], "rear": []}
    for row in manifest:
        b = row["bin"]
        if b in bins:
            bins[b].append(row["frame"])
    return bins


# ---------------------------------------------------------------------------
# COCO Background Loading
# ---------------------------------------------------------------------------

_coco_files = None

def get_coco_files():
    """
    Returns cached list of all COCO validation image paths.
    Loaded once and cached for subsequent calls.
    """
    global _coco_files
    if _coco_files is None:
        _coco_files = [
            os.path.join(config.COCO_DIR, f)
            for f in os.listdir(config.COCO_DIR)
            if f.endswith(".jpg")
        ]
    return _coco_files


def get_random_background(target_size=(512, 512), seed=None):
    """
    Loads a random COCO image crop as background.
    Applies gaussian blur to prevent network from learning
    COCO texture statistics.

    Args:
        target_size: (width, height) of output background
        seed: optional random seed for reproducibility

    Returns:
        PIL RGB image of size target_size
    """
    if seed is not None:
        random.seed(seed)

    coco_files = get_coco_files()
    bg_path = random.choice(coco_files)
    bg = Image.open(bg_path).convert("RGB")

    # random crop to target size
    w, h = bg.size
    tw, th = target_size
    if w < tw or h < th:
        bg = bg.resize(
            (max(w, tw), max(h, th)), Image.BILINEAR
        )
        w, h = bg.size

    left = random.randint(0, w - tw)
    top = random.randint(0, h - th)
    bg = bg.crop((left, top, left + tw, top + th))

    # gaussian blur to suppress COCO texture statistics
    bg = bg.filter(ImageFilter.GaussianBlur(radius=config.BG_BLUR_SIGMA))

    # slight color jitter
    strength = config.BG_COLOR_JITTER_STRENGTH
    bg = ImageEnhance.Brightness(bg).enhance(
        1.0 + random.uniform(-strength, strength)
    )
    bg = ImageEnhance.Contrast(bg).enhance(
        1.0 + random.uniform(-strength, strength)
    )
    bg = ImageEnhance.Color(bg).enhance(
        1.0 + random.uniform(-strength, strength)
    )

    return bg


# ---------------------------------------------------------------------------
# Alpha Compositing
# ---------------------------------------------------------------------------

def composite_on_background(rgba_image, background=None):
    """
    Composites an RGBA object image onto a background.
    If background is None, a random COCO background is used.

    Args:
        rgba_image: PIL RGBA image of the object
        background: PIL RGB image or None

    Returns:
        PIL RGB composite image
    """
    if background is None:
        background = get_random_background(
            target_size=rgba_image.size
        )
    else:
        background = background.convert("RGB")
        if background.size != rgba_image.size:
            background = background.resize(
                rgba_image.size, Image.BILINEAR
            )

    # paste object using alpha channel as mask
    background = background.copy()
    background.paste(rgba_image, mask=rgba_image.split()[3])
    return background


# ---------------------------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    splits = config.load_splits()
    test_obj = splits["eval_high"][0]
    print(f"Testing with object: {test_obj}")

    # test I0 loading
    rgba, frame_idx = load_i0_rgba(test_obj)
    print(f"I0 frame index: {frame_idx}")
    print(f"I0 RGBA size: {rgba.size}, mode: {rgba.mode}")

    # test manifest loading
    bins = get_test_frames_by_bin(test_obj)
    for bin_name, frames in bins.items():
        print(f"  {bin_name}: {len(frames)} test frames")

    # test background loading
    bg = get_random_background()
    print(f"Background size: {bg.size}, mode: {bg.mode}")

    # test compositing
    composite = composite_on_background(rgba)
    print(f"Composite size: {composite.size}, mode: {composite.mode}")

    # save test output
    out_path = "/home/enbiya/a2s_project/data/gso_data/dataset_test.png"
    composite.save(out_path)
    print(f"Test composite saved to {out_path}")
    print("dataset.py OK")



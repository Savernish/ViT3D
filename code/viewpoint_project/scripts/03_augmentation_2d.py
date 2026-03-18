"""
scripts/03_augmentation_2d.py
=============================
Generates synthetic training images for Condition B
using 2D augmentation of the I0 reference image.

For each object:
1. Loads I0 RGBA from training_data/<obj_id>/i0.png
2. Applies 2D augmentation pipeline:
   - Random rotation (0-360 degrees)
   - Horizontal flip
   - Scale jitter
   - Color jitter
   - Random crop
   - Random perspective warp
3. Composites each augmented view onto a random COCO background
4. Saves final composited training images

Image count is exactly matched to Condition A (400 images per object).

Output structure:
    training_data/
        <obj_id>/
            condition_b/
                final/      — composited training images (400 images)

Run:
    python scripts/03_augmentation_2d.py

Run single object (for testing):
    python scripts/03_augmentation_2d.py --obj_id <obj_id>
"""

import os
import sys
import argparse
import time
import random
import numpy as np
from PIL import Image, ImageEnhance

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataset import composite_on_background

config.make_dirs()

# fixed seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# 2D Augmentation Pipeline
# ---------------------------------------------------------------------------

def augment_rgba(rgba_image):
    """
    Applies 2D augmentation to an RGBA object image.
    Augmentations are applied to RGB and alpha channel together
    to maintain mask consistency.

    Args:
        rgba_image: PIL RGBA image

    Returns:
        PIL RGBA augmented image
    """
    # split into RGB and alpha
    rgb = rgba_image.convert("RGB")
    alpha = rgba_image.split()[3]

    # --- Random Rotation (0-360 degrees) ---
    angle = random.uniform(
        config.AUG_ROTATION_DEGREES[0],
        config.AUG_ROTATION_DEGREES[1]
    )
    rgb = TF.rotate(rgb, angle, fill=0)
    alpha = TF.rotate(alpha, angle, fill=0)

    # --- Horizontal Flip ---
    if random.random() < config.AUG_FLIP_PROBABILITY:
        rgb = TF.hflip(rgb)
        alpha = TF.hflip(alpha)

    # --- Scale Jitter ---
    scale = random.uniform(
        config.AUG_SCALE_RANGE[0],
        config.AUG_SCALE_RANGE[1]
    )
    new_w = int(rgba_image.width * scale)
    new_h = int(rgba_image.height * scale)
    rgb = TF.resize(rgb, (new_h, new_w), interpolation=Image.BILINEAR)
    alpha = TF.resize(alpha, (new_h, new_w), interpolation=Image.BILINEAR)

    # pad or crop back to original size
    rgb = _pad_or_crop(rgb, rgba_image.size)
    alpha = _pad_or_crop(alpha, rgba_image.size)

    # --- Color Jitter (RGB only — do not jitter alpha) ---
    cj = config.AUG_COLOR_JITTER
    color_jitter = T.ColorJitter(
        brightness=cj["brightness"],
        contrast=cj["contrast"],
        saturation=cj["saturation"],
        hue=cj["hue"]
    )
    rgb = color_jitter(rgb)

    # --- Random Perspective Warp ---
    # applied to both RGB and alpha to maintain consistency
    distortion = config.AUG_PERSPECTIVE_DISTORTION
    perspective_transform = T.RandomPerspective(
        distortion_scale=distortion,
        p=1.0,
        fill=0
    )
    # get same transform parameters for both
    width, height = rgb.size
    startpoints, endpoints = T.RandomPerspective.get_params(
        width, height, distortion
    )
    rgb = TF.perspective(rgb, startpoints, endpoints, fill=0)
    alpha = TF.perspective(alpha, startpoints, endpoints, fill=0)

    # --- Random Crop ---
    crop_scale = random.uniform(0.8, 1.0)
    crop_w = int(rgba_image.width * crop_scale)
    crop_h = int(rgba_image.height * crop_scale)
    left = random.randint(0, max(0, rgb.width - crop_w))
    top = random.randint(0, max(0, rgb.height - crop_h))
    rgb = TF.crop(rgb, top, left, crop_h, crop_w)
    alpha = TF.crop(alpha, top, left, crop_h, crop_w)

    # resize back to original size
    rgb = TF.resize(rgb, (rgba_image.height, rgba_image.width),
                    interpolation=Image.BILINEAR)
    alpha = TF.resize(alpha, (rgba_image.height, rgba_image.width),
                      interpolation=Image.NEAREST)

    # recombine into RGBA
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    return result


def _pad_or_crop(img, target_size):
    """
    Pads or center-crops image to target_size (width, height).
    """
    tw, th = target_size
    w, h = img.size

    if w == tw and h == th:
        return img

    # pad if smaller
    if w < tw or h < th:
        pad_left = (tw - w) // 2
        pad_top = (th - h) // 2
        result = Image.new(img.mode, (tw, th), 0)
        result.paste(img, (pad_left, pad_top))
        return result

    # crop if larger
    left = (w - tw) // 2
    top = (h - th) // 2
    return img.crop((left, top, left + tw, top + th))


# ---------------------------------------------------------------------------
# Per-Object Processing
# ---------------------------------------------------------------------------

def process_object(obj_id):
    """
    Runs full 2D augmentation pipeline for one object.

    Args:
        obj_id: str, GSO object folder name
    """
    out_dir = os.path.join(
        config.TRAINING_DATA_ROOT, obj_id, "condition_b"
    )
    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    # skip if already completed
    existing = len([f for f in os.listdir(final_dir)
                    if f.endswith(".png")])
    if existing >= config.TARGET_TRAINING_IMAGES:
        print(f"  Skipping {obj_id} — already has {existing} images")
        return existing

    # load I0 RGBA
    i0_path = os.path.join(config.TRAINING_DATA_ROOT, obj_id, "i0.png")
    rgba = Image.open(i0_path).convert("RGBA")

    # generate augmented images
    print(f"  Generating {config.TARGET_TRAINING_IMAGES} augmented images...")
    t0 = time.time()

    for idx in range(config.TARGET_TRAINING_IMAGES):
        augmented = augment_rgba(rgba)
        composited = composite_on_background(augmented)
        composited.save(
            os.path.join(final_dir, f"{str(idx).zfill(4)}.png")
        )

    elapsed = time.time() - t0
    print(f"  Generation complete in {elapsed:.1f}s")
    print(f"  Saved {config.TARGET_TRAINING_IMAGES} images to {final_dir}")
    return config.TARGET_TRAINING_IMAGES


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(obj_id=None):
    if obj_id:
        objects = [obj_id]
        print(f"Single object mode: {obj_id}")
    else:
        high_texture = config.load_object_list(config.HIGH_TEXTURE_FILE)
        low_texture = config.load_object_list(config.LOW_TEXTURE_FILE)
        objects = high_texture + low_texture
        print(f"Processing {len(objects)} objects for Condition B...")

    print(f"Target training images per object: "
          f"{config.TARGET_TRAINING_IMAGES}")
    print()

    total_start = time.time()
    success = 0
    failed = []

    for i, oid in enumerate(objects):
        print(f"[{i+1}/{len(objects)}] {oid}")
        try:
            n = process_object(oid)
            print(f"  Done — {n} images")
            success += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(oid)
        print()

    total_elapsed = time.time() - total_start
    print(f"Completed: {success}/{len(objects)} objects")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    if failed:
        print(f"Failed objects: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj_id",
        type=str,
        default=None,
        help="Process single object. If not set, processes all objects."
    )
    args = parser.parse_args()
    main(obj_id=args.obj_id)

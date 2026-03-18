"""
scripts/02_generate_zero123.py
==============================
Generates synthetic multi-view training images for Condition A
using Zero123++.

For each object:
1. Loads I0 RGBA from training_data/<obj_id>/i0.png
2. Runs Zero123++ for 16 inference calls (96 base views)
3. Splits each tiled output into 6 individual views
4. Composites each view onto a random COCO background
5. Replicates uniformly per azimuth to reach TARGET_TRAINING_IMAGES
6. Saves final composited training images

Output structure:
    training_data/
        <obj_id>/
            condition_a/
                raw/        — raw Zero123++ outputs (96 views)
                final/      — composited training images (400 images)

Run:
    python scripts/02_generate_zero123.py

Run single object (for testing):
    python scripts/02_generate_zero123.py --obj_id <obj_id>
"""

import os
import sys
import argparse
import time
import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataset import composite_on_background, get_coco_files

config.make_dirs()


# ---------------------------------------------------------------------------
# Pipeline Loading
# ---------------------------------------------------------------------------

_pipeline = None

def get_pipeline():
    """
    Loads Zero123++ pipeline once and caches it.
    Subsequent calls return the cached pipeline.
    """
    global _pipeline
    if _pipeline is None:
        print("Loading Zero123++ pipeline...")
        _pipeline = DiffusionPipeline.from_pretrained(
            config.ZERO123_MODEL_ID,
            custom_pipeline=config.ZERO123_PIPELINE_ID,
            torch_dtype=torch.float16
        )
        _pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            _pipeline.scheduler.config,
            timestep_spacing="trailing"
        )
        _pipeline.to("cuda:0")
        print("Pipeline loaded.")
    return _pipeline


# ---------------------------------------------------------------------------
# View Generation
# ---------------------------------------------------------------------------

def generate_views(rgba_image, num_calls=config.ZERO123_NUM_CALLS):
    """
    Generates multi-view images from a single RGBA input.

    Args:
        rgba_image: PIL RGBA image of the object
        num_calls: number of Zero123++ inference calls.
                   Each call produces 6 views as a tiled image.

    Returns:
        list of PIL RGB images — individual views
    """
    pipeline = get_pipeline()
    all_views = []

    for call_idx in range(num_calls):
        result = pipeline(
            rgba_image,
            num_inference_steps=config.ZERO123_INFERENCE_STEPS
        ).images[0]

        # split tiled 2x3 grid into 6 individual views
        w, h = result.size
        tile_w = w // 2
        tile_h = h // 3

        for row in range(3):
            for col in range(2):
                left = col * tile_w
                top = row * tile_h
                crop = result.crop(
                    (left, top, left + tile_w, top + tile_h)
                )
                all_views.append(crop.convert("RGB"))

        if (call_idx + 1) % 4 == 0:
            print(f"  Generated {len(all_views)}/{num_calls * 6} views...")

    return all_views


# ---------------------------------------------------------------------------
# Background Compositing with Uniform Replication
# ---------------------------------------------------------------------------

def composite_with_uniform_replication(views, target_count):
    """
    Composites views onto random COCO backgrounds.
    Replicates uniformly across views to reach target_count.
    Each replicated view gets a different random background.

    Args:
        views: list of PIL RGB images (raw Zero123++ outputs)
        target_count: int, desired total number of training images

    Returns:
        list of PIL RGB composited images
    """
    n_views = len(views)
    composited = []

    # compute how many times each view is replicated
    base_reps = target_count // n_views
    extra = target_count % n_views

    # assign replication counts uniformly
    reps = [base_reps + (1 if i < extra else 0) for i in range(n_views)]

    for view_idx, (view, rep) in enumerate(zip(views, reps)):
        for _ in range(rep):
            composited.append(composite_on_background(
                to_rgba_with_white_bg(view)
            ))

    return composited


def to_rgba_with_white_bg(rgb_image):
    """
    Converts a Zero123++ RGB output to RGBA by creating a simple
    alpha mask — removes the default gray background.
    Zero123++ outputs objects on a uniform gray background (128,128,128).
    We threshold to isolate the object.

    Args:
        rgb_image: PIL RGB image

    Returns:
        PIL RGBA image
    """
    import numpy as np
    arr = np.array(rgb_image)

    # gray background in Zero123++ is approximately (128,128,128)
    # create mask where pixels differ significantly from gray
    gray_val = 128
    diff = np.abs(arr.astype(int) - gray_val).max(axis=2)
    alpha = (diff > 15).astype(np.uint8) * 255

    rgba = Image.fromarray(arr).convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    return rgba


# ---------------------------------------------------------------------------
# Per-Object Processing
# ---------------------------------------------------------------------------

def process_object(obj_id):
    """
    Runs full Zero123++ generation pipeline for one object.

    Args:
        obj_id: str, GSO object folder name
    """
    out_dir = os.path.join(
        config.TRAINING_DATA_ROOT, obj_id, "condition_a"
    )
    raw_dir = os.path.join(out_dir, "raw")
    final_dir = os.path.join(out_dir, "final")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # skip if already completed
    existing = len([f for f in os.listdir(final_dir)
                    if f.endswith(".png")])
    if existing >= config.TARGET_TRAINING_IMAGES:
        print(f"  Skipping {obj_id} — already has {existing} images")
        return existing

    # load I0
    i0_path = os.path.join(config.TRAINING_DATA_ROOT, obj_id, "i0.png")
    rgba = Image.open(i0_path).convert("RGBA")

    # generate views
    print(f"  Generating {config.ZERO123_TOTAL_VIEWS} views...")
    t0 = time.time()
    views = generate_views(rgba)
    elapsed = time.time() - t0
    print(f"  Generation complete in {elapsed:.1f}s")

    # save raw views
    for idx, view in enumerate(views):
        view.save(os.path.join(raw_dir, f"{str(idx).zfill(4)}.png"))

    # composite with uniform replication
    print(f"  Compositing {config.TARGET_TRAINING_IMAGES} training images...")
    composited = composite_with_uniform_replication(
        views, config.TARGET_TRAINING_IMAGES
    )

    # save final training images
    for idx, img in enumerate(composited):
        img.save(os.path.join(final_dir, f"{str(idx).zfill(4)}.png"))

    print(f"  Saved {len(composited)} training images to {final_dir}")
    return len(composited)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(obj_id=None):
    if obj_id:
        # single object mode
        objects = [obj_id]
        print(f"Single object mode: {obj_id}")
    else:
        # all objects
        high_texture = config.load_object_list(config.HIGH_TEXTURE_FILE)
        low_texture = config.load_object_list(config.LOW_TEXTURE_FILE)
        objects = high_texture + low_texture
        print(f"Processing {len(objects)} objects for Condition A...")

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
        help="Process single object by ID. If not set, processes all objects."
    )
    args = parser.parse_args()
    main(obj_id=args.obj_id)

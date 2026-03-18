"""
scripts/01_extract_alpha.py
===========================
Extracts RGBA images for all 30 selected objects.
Combines GSO RGB renders with GSO masks to produce
clean object images with transparent backgrounds.

Output structure:
    training_data/
        <obj_id>/
            i0.png          — reference image (RGBA, frontal view)
            i0_frame.txt    — frame index of I0

Run:
    python scripts/01_extract_alpha.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataset import load_i0_rgba, load_rgba

config.make_dirs()


def extract_alpha_for_object(obj_id):
    """
    Extracts and saves I0 RGBA image for one object.

    Args:
        obj_id: str, GSO object folder name

    Returns:
        frame_idx: int, the frame index selected as I0
    """
    out_dir = os.path.join(config.TRAINING_DATA_ROOT, obj_id)
    os.makedirs(out_dir, exist_ok=True)

    # load I0
    rgba, frame_idx = load_i0_rgba(obj_id)

    # save RGBA
    i0_path = os.path.join(out_dir, "i0.png")
    rgba.save(i0_path)

    # save frame index for reference
    frame_txt_path = os.path.join(out_dir, "i0_frame.txt")
    with open(frame_txt_path, "w") as f:
        f.write(str(frame_idx))

    return frame_idx


def main():
    # load all 30 objects
    high_texture = config.load_object_list(config.HIGH_TEXTURE_FILE)
    low_texture = config.load_object_list(config.LOW_TEXTURE_FILE)
    all_objects = high_texture + low_texture

    print(f"Extracting I0 RGBA for {len(all_objects)} objects...")
    print(f"Output root: {config.TRAINING_DATA_ROOT}")
    print()

    success = 0
    failed = []

    for i, obj_id in enumerate(all_objects):
        try:
            frame_idx = extract_alpha_for_object(obj_id)
            texture = "HT" if obj_id in high_texture else "LT"
            print(f"[{i+1:02d}/{len(all_objects)}] [{texture}] {obj_id} "
                  f"— I0 frame: {frame_idx}")
            success += 1
        except Exception as e:
            print(f"[{i+1:02d}/{len(all_objects)}] FAILED {obj_id}: {e}")
            failed.append(obj_id)

    print()
    print(f"Completed: {success}/{len(all_objects)}")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("All objects processed successfully.")

    # verify outputs
    print()
    print("Verifying outputs...")
    missing = []
    for obj_id in all_objects:
        i0_path = os.path.join(config.TRAINING_DATA_ROOT, obj_id, "i0.png")
        if not os.path.exists(i0_path):
            missing.append(obj_id)

    if missing:
        print(f"Missing I0 files: {missing}")
    else:
        print("All I0 files present.")


if __name__ == "__main__":
    main()

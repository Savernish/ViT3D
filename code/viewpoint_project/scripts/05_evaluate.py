"""
scripts/05_evaluate.py
======================
Evaluates trained LoRA adapters on the GSO test set.

For each object and condition:
1. Loads trained LoRA checkpoint and prototype
2. Runs SAM automatic mask generation on each test image
3. Computes cosine similarity between region embeddings and prototype
4. Applies threshold and NMS to produce detections
5. Computes mAP@0.5 per viewpoint bin

Run single object PoC:
    python scripts/05_evaluate.py --obj_id 0X1fGvojr1Z

Run full evaluation:
    python scripts/05_evaluate.py
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataset import load_manifest, get_test_frames_by_bin
from utils.metrics import (
    compute_map_per_bin,
    aggregate_results,
    print_results_table,
    get_gt_box_from_mask,
    compute_iou
)

config.make_dirs()


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_trained_model(obj_id, condition, device):
    """
    Loads DINOv2 + trained LoRA adapter for one object/condition.

    Returns:
        model, prototype tensor
    """
    from transformers import Dinov2Model
    from peft import PeftModel

    checkpoint_dir = os.path.join(
        config.CHECKPOINTS_ROOT, obj_id, condition
    )
    prototype_path = os.path.join(checkpoint_dir, "prototype.pt")

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"No checkpoint at {checkpoint_dir}")
    if not os.path.exists(prototype_path):
        raise FileNotFoundError(f"No prototype at {prototype_path}")

    # load base model
    base_model = Dinov2Model.from_pretrained(config.DINOV2_MODEL_ID)

    # load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.to(device)
    model.eval()

    # load prototype
    prototype = torch.load(prototype_path, map_location=device)
    prototype = F.normalize(prototype, dim=0)

    return model, prototype


# ---------------------------------------------------------------------------
# SAM Region Proposals
# ---------------------------------------------------------------------------

_sam_predictor = None

def get_sam():
    """Loads SAM automatic mask generator once and caches it."""
    global _sam_predictor
    if _sam_predictor is None:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        # download SAM checkpoint if not present
        sam_checkpoint = os.path.join(
            config.REPOS_ROOT, "sam_vit_b_01ec64.pth"
        )
        if not os.path.exists(sam_checkpoint):
            print("Downloading SAM checkpoint...")
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, sam_checkpoint)
            print("SAM checkpoint downloaded.")

        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam.to("cuda")
        _sam_predictor = SamAutomaticMaskGenerator(
            sam,
            points_per_side=config.SAM_POINTS_PER_SIDE,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=500,
        )
    return _sam_predictor


def get_region_proposals(image_np):
    """
    Runs SAM automatic mask generation on an image.

    Args:
        image_np: numpy array (H, W, 3) uint8

    Returns:
        list of (x1, y1, x2, y2) bounding boxes
    """
    sam = get_sam()
    masks = sam.generate(image_np)

    boxes = []
    for mask_data in masks:
        bbox = mask_data["bbox"]  # SAM returns [x, y, w, h]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        boxes.append((int(x1), int(y1), int(x2), int(y2)))

    return boxes


# ---------------------------------------------------------------------------
# Embedding Extraction
# ---------------------------------------------------------------------------

def get_cls_embedding(model, pixel_values):
    """Extracts CLS token embedding from DINOv2."""
    outputs = model(pixel_values=pixel_values)
    return outputs.last_hidden_state[:, 0, :]


def embed_region(model, image_np, box, device):
    """
    Crops a region from an image and computes its CLS embedding.

    Args:
        model: DINOv2 + LoRA model
        image_np: numpy array (H, W, 3)
        box: (x1, y1, x2, y2)
        device: torch device

    Returns:
        Tensor (768,) normalized embedding
    """
    x1, y1, x2, y2 = box
    # ensure valid crop
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image_np.shape[1], x2)
    y2 = min(image_np.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = Image.fromarray(image_np[y1:y2, x1:x2])
    crop = crop.resize((224, 224), Image.BILINEAR)

    tensor = torch.tensor(
        np.array(crop), dtype=torch.float32
    ).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast():
            embedding = get_cls_embedding(model, tensor)
    embedding = F.normalize(embedding.squeeze(0), dim=0)
    return embedding


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect(model, prototype, image_np, threshold, device):
    """
    Detects object in image using region proposals + cosine similarity.

    Args:
        model: DINOv2 + LoRA
        prototype: Tensor (768,) normalized prototype
        image_np: numpy array (H, W, 3)
        threshold: float cosine similarity threshold
        device: torch device

    Returns:
        list of (confidence, x1, y1, x2, y2) detections before NMS
    """
    boxes = get_region_proposals(image_np)
    if not boxes:
        return []

    detections = []
    for box in boxes:
        embedding = embed_region(model, image_np, box, device)
        if embedding is None:
            continue
        similarity = torch.dot(embedding, prototype).item()
        if similarity >= threshold:
            detections.append((similarity, *box))

    # NMS
    detections = nms(detections, iou_threshold=config.NMS_IOU_THRESHOLD)
    return detections


def nms(detections, iou_threshold=0.5):
    """
    Non-maximum suppression.

    Args:
        detections: list of (confidence, x1, y1, x2, y2)
        iou_threshold: float

    Returns:
        filtered list of detections
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections
            if compute_iou(best[1:], d[1:]) < iou_threshold
        ]

    return kept


# ---------------------------------------------------------------------------
# Per-Object Evaluation
# ---------------------------------------------------------------------------

def evaluate_object(obj_id, condition, threshold, device):
    """
    Evaluates one object under one condition across all viewpoint bins.

    Returns:
        dict from compute_map_per_bin()
    """
    print(f"  Loading model for {obj_id}/{condition}...")
    model, prototype = load_trained_model(obj_id, condition, device)

    bins = get_test_frames_by_bin(obj_id)
    results_per_bin = {b: [] for b in bins}

    total_frames = sum(len(v) for v in bins.values())
    processed = 0

    for bin_name, frames in bins.items():
        for frame_idx in frames:
            # load test image
            img_path = os.path.join(
                config.GSO_ROOT, obj_id, "rgb", f"{frame_idx}.png"
            )
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)

            # get ground truth box
            gt_box = get_gt_box_from_mask(obj_id, frame_idx)

            # run detection
            detections = detect(
                model, prototype, image_np, threshold, device
            )

            results_per_bin[bin_name].append((detections, gt_box))
            processed += 1

            if processed % 50 == 0:
                print(f"  Processed {processed}/{total_frames} frames...")

    map_scores = compute_map_per_bin(results_per_bin)
    return map_scores


# ---------------------------------------------------------------------------
# Threshold Sweep
# ---------------------------------------------------------------------------

def sweep_threshold(obj_ids, condition, device, texture_group):
    """
    Sweeps cosine similarity threshold on validation objects.
    Returns the threshold that maximizes overall mAP.

    Args:
        obj_ids: list of validation object IDs
        condition: str
        device: torch device
        texture_group: str, for logging

    Returns:
        float best threshold
    """
    print(f"\nThreshold sweep — {texture_group} / {condition}")
    thresholds = np.arange(0.3, 0.95, 0.05)
    best_threshold = 0.5
    best_map = 0.0

    for threshold in thresholds:
        maps = []
        for obj_id in obj_ids:
            checkpoint_dir = os.path.join(
                config.CHECKPOINTS_ROOT, obj_id, condition
            )
            if not os.path.exists(checkpoint_dir):
                continue
            try:
                scores = evaluate_object(
                    obj_id, condition, float(threshold), device
                )
                maps.append(scores["overall"])
            except Exception as e:
                print(f"  Error on {obj_id}: {e}")
                continue

        if maps:
            avg_map = np.mean(maps)
            print(f"  threshold={threshold:.2f} mAP={avg_map:.4f}")
            if avg_map > best_map:
                best_map = avg_map
                best_threshold = float(threshold)

    print(f"  Best threshold: {best_threshold:.2f} (mAP={best_map:.4f})")
    return best_threshold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(obj_id=None, skip_threshold_sweep=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = config.load_splits()

    if obj_id:
        # PoC mode — single object, use fixed threshold
        print(f"\nPoC mode: evaluating {obj_id}")
        print("Using fixed threshold 0.5 for PoC (no sweep)")
        threshold_ht = 0.5
        threshold_lt = 0.5

        results = {}
        for condition in ["condition_a", "condition_b"]:
            checkpoint_dir = os.path.join(
                config.CHECKPOINTS_ROOT, obj_id, condition
            )
            if not os.path.exists(checkpoint_dir):
                print(f"No checkpoint for {obj_id}/{condition} — skipping")
                continue
            print(f"\nEvaluating {condition}...")

            # determine texture group
            high_objs = config.load_object_list(config.HIGH_TEXTURE_FILE)
            threshold = threshold_ht if obj_id in high_objs else threshold_lt

            scores = evaluate_object(obj_id, condition, threshold, device)
            results[condition] = scores

            print_results_table(
                aggregate_results([scores]),
                config.CONDITION_NAMES[condition]
            )

        # save PoC results
        out_path = os.path.join(
            config.RESULTS_ROOT, f"poc_{obj_id}.json"
        )
        os.makedirs(config.RESULTS_ROOT, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nPoC results saved to {out_path}")
        return

    # full evaluation
    # threshold sweep on validation split
    if not skip_threshold_sweep:
        threshold_ht_a = sweep_threshold(
            splits["val_high"], "condition_a", device, "high_texture"
        )
        threshold_lt_a = sweep_threshold(
            splits["val_low"], "condition_a", device, "low_texture"
        )
        threshold_ht_b = sweep_threshold(
            splits["val_high"], "condition_b", device, "high_texture"
        )
        threshold_lt_b = sweep_threshold(
            splits["val_low"], "condition_b", device, "low_texture"
        )

        # save thresholds
        thresholds = {
            "condition_a": {
                "high_texture": threshold_ht_a,
                "low_texture": threshold_lt_a
            },
            "condition_b": {
                "high_texture": threshold_ht_b,
                "low_texture": threshold_lt_b
            }
        }
        thresh_path = os.path.join(config.RESULTS_ROOT, "thresholds.json")
        os.makedirs(config.RESULTS_ROOT, exist_ok=True)
        with open(thresh_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        print(f"\nThresholds saved to {thresh_path}")
    else:
        thresh_path = os.path.join(config.RESULTS_ROOT, "thresholds.json")
        with open(thresh_path) as f:
            thresholds = json.load(f)

    # evaluate all objects
    eval_objects = splits["eval_high"] + splits["eval_low"]
    high_objs = config.load_object_list(config.HIGH_TEXTURE_FILE)

    all_results = {"condition_a": [], "condition_b": []}

    for condition in ["condition_a", "condition_b"]:
        print(f"\n{'='*50}")
        print(f"Evaluating {condition} on {len(eval_objects)} objects")
        print(f"{'='*50}")

        for i, oid in enumerate(eval_objects):
            texture = "high" if oid in high_objs else "low"
            threshold = thresholds[condition][f"{texture}_texture"]

            print(f"\n[{i+1}/{len(eval_objects)}] {oid} "
                  f"(threshold={threshold:.2f})")
            try:
                scores = evaluate_object(oid, condition, threshold, device)
                all_results[condition].append(scores)
                print(f"  overall mAP: {scores['overall']:.4f}")
            except Exception as e:
                print(f"  FAILED: {e}")

    # aggregate and print final results
    print("\n\nFINAL RESULTS")
    for condition in ["condition_a", "condition_b"]:
        if all_results[condition]:
            aggregated = aggregate_results(all_results[condition])
            print_results_table(
                aggregated,
                config.CONDITION_NAMES[condition]
            )

    # save results
    results_path = os.path.join(config.RESULTS_ROOT, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj_id",
        type=str,
        default=None,
        help="Evaluate single object in PoC mode."
    )
    parser.add_argument(
        "--skip_threshold_sweep",
        action="store_true",
        help="Skip threshold sweep and load from saved thresholds.json"
    )
    args = parser.parse_args()
    main(obj_id=args.obj_id,
         skip_threshold_sweep=args.skip_threshold_sweep)
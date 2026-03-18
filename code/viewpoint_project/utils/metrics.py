"""
utils/metrics.py
================
Evaluation metrics for the viewpoint generalization experiment.
Computes mAP@0.5 per viewpoint bin and overall.

Detection format follows COCO convention:
    prediction: (confidence_score, x1, y1, x2, y2)
    ground_truth: (x1, y1, x2, y2)

Since GSO provides full object renders (one object per image),
each test image has exactly one ground truth bounding box —
the tight bounding box around the object mask.
"""

import numpy as np
import os
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# IoU Computation
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """
    Computes Intersection over Union between two bounding boxes.

    Args:
        box_a: (x1, y1, x2, y2)
        box_b: (x1, y1, x2, y2)

    Returns:
        float IoU in [0, 1]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------------------------
# Ground Truth Box from GSO Mask
# ---------------------------------------------------------------------------

def get_gt_box_from_mask(obj_id, frame_idx):
    """
    Computes the tight bounding box around the object mask
    for a given GSO object and frame.

    Args:
        obj_id: str, GSO object folder name
        frame_idx: str or int, zero-padded frame index

    Returns:
        (x1, y1, x2, y2) bounding box in pixel coordinates
        or None if mask is empty
    """
    if isinstance(frame_idx, int):
        frame_idx = str(frame_idx).zfill(6)

    mask_path = os.path.join(
        config.GSO_ROOT, obj_id, "mask", f"{frame_idx}.png"
    )
    mask = np.array(Image.open(mask_path).convert("L"))
    rows = np.any(mask > 127, axis=1)
    cols = np.any(mask > 127, axis=0)

    if not rows.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return (int(x1), int(y1), int(x2), int(y2))


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------

def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Computes Average Precision (AP) at a given IoU threshold.

    Args:
        predictions: list of (confidence, x1, y1, x2, y2)
                     one entry per test image detection
        ground_truths: list of (x1, y1, x2, y2) or None
                       one entry per test image
                       None means no ground truth (skip image)

    Returns:
        float AP in [0, 1]
    """
    if not predictions:
        return 0.0

    # sort predictions by confidence descending
    predictions = sorted(predictions, key=lambda x: x[0], reverse=True)

    tp = []
    fp = []
    matched = set()

    for pred_idx, (conf, px1, py1, px2, py2) in enumerate(predictions):
        pred_box = (px1, py1, px2, py2)
        gt = ground_truths[pred_idx]

        if gt is None:
            fp.append(1)
            tp.append(0)
            continue

        iou = compute_iou(pred_box, gt)
        if iou >= iou_threshold and pred_idx not in matched:
            tp.append(1)
            fp.append(0)
            matched.add(pred_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    n_gt = sum(1 for gt in ground_truths if gt is not None)

    if n_gt == 0:
        return 0.0

    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # prepend sentinel values
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])

    # ensure precision is monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # compute area under precision-recall curve
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum(
        (recalls[indices] - recalls[indices - 1]) * precisions[indices]
    )
    return float(ap)


# ---------------------------------------------------------------------------
# Per-Bin mAP
# ---------------------------------------------------------------------------

def compute_map_per_bin(results_per_bin, iou_threshold=config.MAP_IOU_THRESHOLD):
    """
    Computes mAP per viewpoint bin and overall.

    Args:
        results_per_bin: dict mapping bin_name to list of
                         (predictions, ground_truth) tuples
                         where:
                           predictions = list of (conf, x1, y1, x2, y2)
                           ground_truth = (x1, y1, x2, y2) or None

    Returns:
        dict with keys: 'frontal', 'side', 'rear_side', 'rear', 'overall'
        values are float mAP scores
    """
    map_scores = {}
    all_predictions = []
    all_gts = []

    for bin_name in ["frontal", "side", "rear_side", "rear"]:
        if bin_name not in results_per_bin:
            map_scores[bin_name] = 0.0
            continue

        bin_preds = []
        bin_gts = []
        for preds, gt in results_per_bin[bin_name]:
            if preds:
                # take highest confidence detection per image
                best_pred = max(preds, key=lambda x: x[0])
                bin_preds.append(best_pred)
            else:
                # no detection — append zero confidence dummy
                bin_preds.append((0.0, 0, 0, 1, 1))
            bin_gts.append(gt)
            all_predictions.append(bin_preds[-1])
            all_gts.append(gt)

        ap = compute_ap(bin_preds, bin_gts, iou_threshold)
        map_scores[bin_name] = ap

    # overall mAP across all bins
    map_scores["overall"] = compute_ap(
        all_predictions, all_gts, iou_threshold
    )

    return map_scores


# ---------------------------------------------------------------------------
# Results Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(per_object_maps):
    """
    Aggregates per-object mAP scores into mean and std.

    Args:
        per_object_maps: list of dicts, each from compute_map_per_bin()

    Returns:
        dict with keys: bin names
        values: dict with 'mean' and 'std'
    """
    bins = ["frontal", "side", "rear_side", "rear", "overall"]
    aggregated = {}

    for bin_name in bins:
        scores = [m[bin_name] for m in per_object_maps if bin_name in m]
        if scores:
            aggregated[bin_name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "n": len(scores)
            }
        else:
            aggregated[bin_name] = {"mean": 0.0, "std": 0.0, "n": 0}

    return aggregated


def print_results_table(aggregated, condition_name):
    """
    Prints a formatted results table for a condition.

    Args:
        aggregated: output of aggregate_results()
        condition_name: str label for the condition
    """
    print(f"\n{'='*55}")
    print(f"Results: {condition_name}")
    print(f"{'='*55}")
    print(f"{'Viewpoint':<15} {'mAP@0.5':>10} {'±std':>10} {'n':>5}")
    print(f"{'-'*55}")
    for bin_name in ["frontal", "side", "rear_side", "rear", "overall"]:
        if bin_name in aggregated:
            m = aggregated[bin_name]["mean"]
            s = aggregated[bin_name]["std"]
            n = aggregated[bin_name]["n"]
            print(f"{bin_name:<15} {m:>10.4f} {s:>10.4f} {n:>5}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing metrics...")

    # test IoU
    box_a = (0, 0, 10, 10)
    box_b = (5, 5, 15, 15)
    iou = compute_iou(box_a, box_b)
    print(f"IoU test: {iou:.4f} (expected ~0.1429)")
    assert abs(iou - 0.1429) < 0.001

    # perfect overlap
    iou_perfect = compute_iou(box_a, box_a)
    print(f"Perfect IoU: {iou_perfect:.4f} (expected 1.0)")
    assert iou_perfect == 1.0

    # no overlap
    box_c = (20, 20, 30, 30)
    iou_none = compute_iou(box_a, box_c)
    print(f"No overlap IoU: {iou_none:.4f} (expected 0.0)")
    assert iou_none == 0.0

    # test GT box extraction
    splits = config.load_splits()
    test_obj = splits["eval_high"][0]
    gt_box = get_gt_box_from_mask(test_obj, 0)
    print(f"GT box for {test_obj} frame 0: {gt_box}")
    assert gt_box is not None

    # test AP computation
    predictions = [
        (0.9, 0, 0, 10, 10),
        (0.8, 5, 5, 15, 15),
        (0.7, 20, 20, 30, 30),
    ]
    ground_truths = [
        (0, 0, 10, 10),
        (5, 5, 15, 15),
        (20, 20, 30, 30),
    ]
    ap = compute_ap(predictions, ground_truths)
    print(f"AP perfect detections: {ap:.4f} (expected 1.0)")
    assert abs(ap - 1.0) < 0.001, f"Expected AP ~1.0, got {ap}"

    # test per-bin mAP
    results_per_bin = {
        "frontal": [([(0.9, 0, 0, 10, 10)], (0, 0, 10, 10))],
        "side": [([(0.8, 5, 5, 15, 15)], (5, 5, 15, 15))],
        "rear_side": [([(0.7, 0, 0, 10, 10)], (0, 0, 10, 10))],
        "rear": [([(0.6, 0, 0, 10, 10)], (0, 0, 10, 10))],
    }
    map_scores = compute_map_per_bin(results_per_bin)
    print(f"Per-bin mAP: {map_scores}")

    # test aggregation
    per_object = [map_scores, map_scores, map_scores]
    aggregated = aggregate_results(per_object)
    print_results_table(aggregated, "Test Condition")

    print("\nmetrics.py OK")

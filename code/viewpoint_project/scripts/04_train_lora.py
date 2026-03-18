"""
scripts/04_train_lora.py
========================
Trains LoRA adapters on DINOv2 for Conditions A and B.

For each object and condition:
1. Loads training images from condition_a/final or condition_b/final
2. Fine-tunes LoRA matrices on frozen DINOv2 backbone
3. Uses pure InfoNCE contrastive loss
4. Saves LoRA checkpoint after training

Training details:
- Backbone: DINOv2 base (frozen)
- LoRA: rank 16, alpha 32, inserted into query and value projections
- Batch size: 4 (effective 16 via gradient accumulation)
- Mixed precision: enabled
- Epochs: 50

Output structure:
    checkpoints/
        <obj_id>/
            condition_a/
                lora_weights.pt
                prototype.pt
                training_log.json
            condition_b/
                lora_weights.pt
                prototype.pt
                training_log.json

Run:
    python scripts/04_train_lora.py --condition condition_a
    python scripts/04_train_lora.py --condition condition_b

Run single object:
    python scripts/04_train_lora.py --condition condition_a --obj_id <obj_id>
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.losses import InfoNCELoss, validate_batch_labels

config.make_dirs()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrainingDataset(Dataset):
    """
    Loads training images for one object from one condition.
    Returns (image_tensor, object_label) pairs.
    """

    def __init__(self, obj_id, condition, obj_label):
        """
        Args:
            obj_id: str, GSO object folder name
            condition: str, 'condition_a' or 'condition_b'
            obj_label: int, integer label for this object
        """
        self.obj_label = obj_label
        final_dir = os.path.join(
            config.TRAINING_DATA_ROOT, obj_id, condition, "final"
        )
        self.image_paths = sorted([
            os.path.join(final_dir, f)
            for f in os.listdir(final_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        tensor = torch.tensor(
            np.array(img), dtype=torch.float32
        ).permute(2, 0, 1) / 255.0
        # normalize with ImageNet mean/std (DINOv2 standard)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, self.obj_label


class MultiObjectDataset(Dataset):
    """
    Combines training images from multiple objects into one dataset.
    Each batch will contain views from multiple objects,
    providing hard negatives for InfoNCE loss.
    """

    def __init__(self, obj_ids, condition):
        self.samples = []
        for label, obj_id in enumerate(obj_ids):
            final_dir = os.path.join(
                config.TRAINING_DATA_ROOT, obj_id, condition, "final"
            )
            paths = sorted([
                os.path.join(final_dir, f)
                for f in os.listdir(final_dir)
                if f.endswith(".png")
            ])
            for p in paths:
                self.samples.append((p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        tensor = torch.tensor(
            np.array(img), dtype=torch.float32
        ).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, label


# ---------------------------------------------------------------------------
# LoRA Setup
# ---------------------------------------------------------------------------

def setup_lora(model):
    """
    Injects LoRA matrices into DINOv2 query and value projections.
    Freezes all backbone parameters.
    Only LoRA matrices remain trainable.

    Args:
        model: HuggingFace DINOv2 model

    Returns:
        model with LoRA injected
        list of trainable parameter names
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # verify only LoRA parameters are trainable
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    print(f"  Trainable parameters: {len(trainable)}")
    print(f"  Frozen parameters: {len(frozen)}")

    # count parameter sizes
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable param count: {trainable_params:,} "
          f"({100*trainable_params/total_params:.2f}% of total)")

    return model


def get_cls_embedding(model, pixel_values):
    """
    Extracts CLS token embedding from DINOv2.

    Args:
        model: DINOv2 model (with or without LoRA)
        pixel_values: Tensor (B, 3, 224, 224)

    Returns:
        Tensor (B, 768) — CLS token embeddings
    """
    outputs = model(pixel_values=pixel_values)
    # CLS token is the first token in last_hidden_state
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding


# ---------------------------------------------------------------------------
# Prototype Computation
# ---------------------------------------------------------------------------

def compute_prototype(model, obj_id, condition, device):
    """
    Computes object prototype as mean CLS embedding
    across all training views.

    Args:
        model: trained DINOv2 + LoRA model
        obj_id: str
        condition: str
        device: torch device

    Returns:
        Tensor (768,) — prototype vector
    """
    model.eval()
    dataset = TrainingDataset(obj_id, condition, obj_label=0)
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=2)

    all_embeddings = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            with autocast():
                embeddings = get_cls_embedding(model, images)
            embeddings = F.normalize(embeddings, dim=1)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    prototype = all_embeddings.mean(dim=0)
    prototype = F.normalize(prototype, dim=0)
    return prototype


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_object(obj_id, condition, model_fresh, device):
    """
    Trains LoRA adapter for one object under one condition.

    Args:
        obj_id: str
        condition: str
        model_fresh: freshly loaded DINOv2 model (before LoRA injection)
        device: torch device

    Returns:
        dict with training log
    """
    from transformers import Dinov2Model
    import copy

    out_dir = os.path.join(config.CHECKPOINTS_ROOT, obj_id, condition)
    os.makedirs(out_dir, exist_ok=True)

    weights_path = os.path.join(out_dir, "lora_weights.pt")
    prototype_path = os.path.join(out_dir, "prototype.pt")
    log_path = os.path.join(out_dir, "training_log.json")

    # skip if already trained
    if os.path.exists(weights_path) and os.path.exists(prototype_path):
        print(f"  Skipping {obj_id}/{condition} — already trained")
        return None

    # inject LoRA into fresh model copy
    model = setup_lora(model_fresh)
    model = model.to(device)

    # build dataset — use all 30 objects for hard negatives
    # but only train adapter for this specific object
    # use single object dataset for simplicity
    # multi-object batching handled by loading multiple objects
    splits = config.load_splits()
    all_obj_ids = (
        splits["val_high"] + splits["val_low"] +
        splits["eval_high"] + splits["eval_low"]
    )

    # use subset of objects for hard negatives (memory constraint)
    # include target object plus 3 random others
    import random
    other_objs = [o for o in all_obj_ids if o != obj_id]
    selected_others = random.sample(other_objs, min(3, len(other_objs)))
    training_objects = [obj_id] + selected_others

    dataset = MultiObjectDataset(training_objects, condition)
    loader = DataLoader(
        dataset,
        batch_size=config.NATIVE_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE
    )

    # lr scheduler
    total_steps = config.NUM_EPOCHS * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    loss_fn = InfoNCELoss(temperature=config.INFONCE_TEMPERATURE)
    scaler = GradScaler()

    log = {"losses": [], "epochs": config.NUM_EPOCHS}
    t0 = time.time()

    model.train()
    optimizer.zero_grad()

    for epoch in range(config.NUM_EPOCHS):
        epoch_losses = []
        accum_step = 0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            if not validate_batch_labels(labels):
                continue

            with autocast():
                embeddings = get_cls_embedding(model, images)
                embeddings = F.normalize(embeddings, dim=1)
                loss = loss_fn(embeddings, labels)
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                accum_step = 0

            epoch_losses.append(
                loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            )

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        log["losses"].append(avg_loss)

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{config.NUM_EPOCHS} "
                  f"loss={avg_loss:.4f} "
                  f"time={elapsed:.0f}s")

    # save LoRA weights
    model.save_pretrained(out_dir)
    torch.save(model.state_dict(), weights_path)

    # compute and save prototype
    prototype = compute_prototype(model, obj_id, condition, device)
    torch.save(prototype, prototype_path)

    # save log
    log["total_time_seconds"] = time.time() - t0
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"  Training complete in {log['total_time_seconds']:.0f}s")
    print(f"  Final loss: {log['losses'][-1]:.4f}")
    print(f"  Saved to {out_dir}")

    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(condition, obj_id=None):
    from transformers import Dinov2Model
    import copy

    assert condition in ["condition_a", "condition_b"], \
        "condition must be 'condition_a' or 'condition_b'"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Condition: {condition}")

    # load base DINOv2 model once
    print("Loading DINOv2 base model...")
    base_model = Dinov2Model.from_pretrained(config.DINOV2_MODEL_ID)
    print("DINOv2 loaded.")
    print()

    if obj_id:
        objects = [obj_id]
        print(f"Single object mode: {obj_id}")
    else:
        splits = config.load_splits()
        objects = (
            splits["val_high"] + splits["val_low"] +
            splits["eval_high"] + splits["eval_low"]
        )
        print(f"Training {len(objects)} objects for {condition}...")

    total_start = time.time()
    success = 0
    failed = []

    for i, oid in enumerate(objects):
        print(f"\n[{i+1}/{len(objects)}] {oid} / {condition}")
        try:
            # fresh copy of base model for each object
            model_fresh = copy.deepcopy(base_model)
            train_object(oid, condition, model_fresh, device)
            success += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(oid)

    total_elapsed = time.time() - total_start
    print(f"\nCompleted: {success}/{len(objects)} objects")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["condition_a", "condition_b"],
        help="Which condition to train"
    )
    parser.add_argument(
        "--obj_id",
        type=str,
        default=None,
        help="Train single object. If not set, trains all objects."
    )
    args = parser.parse_args()
    main(condition=args.condition, obj_id=args.obj_id)

"""
utils/losses.py
===============
Loss functions for the viewpoint generalization experiment.
Uses pure InfoNCE contrastive loss as the sole training objective.

InfoNCE pulls embeddings of different views of the same object together
and pushes embeddings of different objects or backgrounds apart.

Reference: Oord et al., "Representation Learning with Contrastive
Predictive Coding", 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# InfoNCE Loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for self-supervised representation learning.

    For each anchor embedding in the batch:
        - Positive: a different synthetic view of the same object
        - Negatives: all other embeddings in the batch (different objects
          or background crops)

    The loss minimizes:
        -log( exp(sim(anchor, positive) / tau) /
              sum_k exp(sim(anchor, k) / tau) )

    where sim() is cosine similarity and tau is the temperature parameter.

    Args:
        temperature: float, controls hardness of negatives.
                     Lower = harder negatives. Default 0.07 (standard).
    """

    def __init__(self, temperature=config.INFONCE_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Computes InfoNCE loss over a batch of embeddings.

        Args:
            embeddings: Tensor of shape (N, D)
                        L2-normalized embedding vectors.
                        N = batch size, D = embedding dimension.
            labels: Tensor of shape (N,)
                    Integer object IDs. Views from the same object
                    share the same label. Different objects have
                    different labels.

        Returns:
            Scalar loss value.
        """
        # ensure embeddings are L2 normalized
        embeddings = F.normalize(embeddings, dim=1)

        # compute full NxN cosine similarity matrix
        # sim_matrix[i][j] = cosine similarity between embedding i and j
        sim_matrix = torch.matmul(embeddings, embeddings.T)  # (N, N)

        # scale by temperature
        sim_matrix = sim_matrix / self.temperature

        # build positive mask — True where labels match
        labels = labels.unsqueeze(1)  # (N, 1)
        positive_mask = (labels == labels.T).float()  # (N, N)

        # remove self-similarity from positive mask
        self_mask = torch.eye(
            embeddings.size(0), device=embeddings.device
        )
        positive_mask = positive_mask - self_mask

        # for numerical stability subtract row max before exp
        sim_matrix_stable = sim_matrix - sim_matrix.detach().max(
            dim=1, keepdim=True
        ).values

        # compute denominator — sum over all non-self entries
        exp_sim = torch.exp(sim_matrix_stable) * (1 - self_mask)
        denominator = exp_sim.sum(dim=1, keepdim=True)  # (N, 1)

        # compute log probability for each positive pair
        log_prob = sim_matrix_stable - torch.log(
            denominator + 1e-8
        )

        # average loss over all positive pairs
        # only compute loss where positive pairs exist
        num_positives = positive_mask.sum(dim=1)
        loss_per_anchor = -(positive_mask * log_prob).sum(dim=1) / (
            num_positives + 1e-8
        )

        # only include anchors that have at least one positive
        valid_anchors = num_positives > 0
        if valid_anchors.sum() == 0:
            # fallback — no valid positives in batch
            # this should not happen with correct batch construction
            return torch.tensor(0.0, requires_grad=True,
                                device=embeddings.device)

        loss = loss_per_anchor[valid_anchors].mean()
        return loss


# ---------------------------------------------------------------------------
# Batch Construction Validator
# ---------------------------------------------------------------------------

def validate_batch_labels(labels):
    """
    Validates that a batch contains at least one positive pair.
    A positive pair requires at least two embeddings with the same label.

    Args:
        labels: Tensor or list of integer object labels

    Returns:
        bool — True if batch is valid for InfoNCE training
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    unique, counts = torch.unique(labels, return_counts=True)
    return (counts > 1).any().item()


# ---------------------------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing InfoNCE loss...")

    torch.manual_seed(42)
    loss_fn = InfoNCELoss(temperature=0.07)

    # simulate batch: 3 objects, 4 views each = 12 embeddings
    N = 12
    D = 768  # DINOv2 base embedding dimension
    embeddings = F.normalize(torch.randn(N, D), dim=1)

    # labels: objects 0,1,2 each with 4 views
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    loss = loss_fn(embeddings, labels)
    print(f"Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"

    # test batch validation
    valid = validate_batch_labels(labels)
    print(f"Batch valid: {valid}")

    # test invalid batch (all different labels)
    invalid_labels = torch.arange(N)
    invalid = validate_batch_labels(invalid_labels)
    print(f"Invalid batch detected correctly: {not invalid}")

    # test that similar embeddings produce lower loss
    similar_embeddings = embeddings.clone()
    # make views of same object nearly identical
    for obj in range(3):
        base = embeddings[obj * 4].clone()
        for v in range(4):
            similar_embeddings[obj * 4 + v] = F.normalize(
                base + 0.01 * torch.randn(D), dim=0
            )

    loss_similar = loss_fn(similar_embeddings, labels)
    print(f"Loss with similar views: {loss_similar.item():.4f}")
    print(f"Loss with random views: {loss.item():.4f}")
    assert loss_similar.item() < loss.item(), \
        "Similar views should produce lower loss"

    print("losses.py OK")

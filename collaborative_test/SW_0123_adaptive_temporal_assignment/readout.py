"""Fixed argmax-slot primary readout registered for SW0123."""
from __future__ import annotations

import torch


def assignment_to_labels(probability: torch.Tensor, minimum_foreground_patches: int = 2):
    """Argmax slots; largest slot is BG with first-patch deterministic tie break."""
    if probability.ndim != 3 or probability.shape[1:] != (256, 11):
        raise ValueError("assignment probabilities must be [B,256,11]")
    if not torch.isfinite(probability).all():
        raise FloatingPointError("assignment probabilities must be finite")
    prediction = probability.argmax(dim=-1)
    labels = torch.zeros_like(prediction, dtype=torch.int64)
    for batch in range(prediction.shape[0]):
        counts = torch.bincount(prediction[batch], minlength=11)
        max_count = int(counts.max())
        largest = torch.nonzero(counts == max_count, as_tuple=False).flatten().tolist()
        background = min(largest, key=lambda slot: int(torch.nonzero(
            prediction[batch] == slot, as_tuple=False)[0, 0]))
        for slot in range(11):
            if slot == background or int(counts[slot]) < minimum_foreground_patches:
                continue
            labels[batch, prediction[batch] == slot] = slot + 1
    return labels.reshape(-1, 16, 16)

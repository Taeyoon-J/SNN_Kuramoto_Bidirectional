"""Pure NumPy helpers for the official Slot Attention transfer protocol."""
import numpy as np


def validate_slice(start, count, total):
    start, count, total = int(start), int(count), int(total)
    if start < 0 or count <= 0 or start + count > total:
        raise ValueError(
            f"Requested slice [{start}:{start + count}] is invalid for {total} images.")
    return start, start + count


def perimeter_background(slot_ids, num_slots=11):
    """Choose the most perimeter-assigned hard slot; ties go to smallest ID."""
    ids = np.asarray(slot_ids)
    if ids.ndim != 2 or min(ids.shape) < 2:
        raise ValueError("slot_ids must be a 2D mask with at least 2 pixels per side.")
    ids = ids.astype(np.int64, copy=False)
    if np.any(ids < 0) or np.any(ids >= int(num_slots)):
        raise ValueError("slot_ids contain a slot outside [0, num_slots).")
    border = np.concatenate((ids[0], ids[-1], ids[1:-1, 0], ids[1:-1, -1]))
    return int(np.bincount(border, minlength=int(num_slots)).argmax())


def remap_foreground(slot_ids, background_slot, num_slots=11):
    """Convert slot IDs to instance labels; remap selected background to zero."""
    ids = np.asarray(slot_ids)
    if ids.ndim != 2:
        raise ValueError("slot_ids must be a 2D hard mask.")
    ids = ids.astype(np.int64, copy=False)
    if np.any(ids < 0) or np.any(ids >= int(num_slots)):
        raise ValueError("slot_ids contain a slot outside [0, num_slots).")
    background_slot = int(background_slot)
    if not 0 <= background_slot < int(num_slots):
        raise ValueError("background_slot must be within [0, num_slots).")
    labels = ids + 1
    labels[ids == background_slot] = 0
    return labels

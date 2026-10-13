"""Deterministic, per-image registered horizontal-flip RGB augmentation."""
from __future__ import annotations

import torch


def make_views(clean_rgb: torch.Tensor, *, source_seed: int, image_id: int):
    """Return clean/transformed uint8 NHWC views and replayable augmentation data.

    Randomness is keyed by (source_seed, image_id), independent of batch order or
    accelerator RNG. Input RGB remains untouched.
    """
    if not torch.is_tensor(clean_rgb) or clean_rgb.dtype != torch.uint8:
        raise TypeError("clean_rgb must be uint8")
    if clean_rgb.device.type != "cpu":
        raise ValueError("view augmentation uses CPU uint8 cache tensors before device transfer")
    if clean_rgb.ndim != 3 or tuple(clean_rgb.shape) != (128, 128, 3):
        raise ValueError("clean_rgb must have shape [128,128,3]")
    seed = ((int(source_seed) & 0xFFFFFFFF) * 0x9E3779B1 + (int(image_id) & 0xFFFFFFFF)) & 0x7FFFFFFFFFFFFFFF
    rng = torch.Generator(device="cpu").manual_seed(seed)
    gain = 0.6 + 0.8 * torch.rand((3,), generator=rng, dtype=torch.float32)
    offset = -0.1 + 0.2 * torch.rand((3,), generator=rng, dtype=torch.float32)
    transformed = clean_rgb.flip(1).to(torch.float32) / 255.0
    transformed = (transformed * gain.view(1, 1, 3) + offset.view(1, 1, 3)).clamp(0.0, 1.0)
    transformed = torch.round(transformed * 255.0).to(torch.uint8)
    return clean_rgb, transformed, {
        "source_seed": int(source_seed), "image_id": int(image_id),
        "gain": gain.tolist(), "offset": offset.tolist(),
    }

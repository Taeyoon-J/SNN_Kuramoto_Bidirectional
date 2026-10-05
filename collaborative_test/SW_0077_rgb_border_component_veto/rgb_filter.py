"""GT-free whole-component veto using robust RGB statistics from image borders."""

import math

import torch
import torch.nn.functional as F


def border_color_component_filter(images, groups, distance_threshold, grid_size=16,
                                  scale_floor=4.0):
    """Drop complete CC groups whose median RGB distance resembles border background.

    `images` is [B,3,H,W] in raw 0..255 RGB space. Border patch means define a
    per-image robust diagonal color model (median and 1.4826*MAD); each supplied
    connected component is kept or dropped as a whole. Group members are never
    split, merged, relabelled, or reordered.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W]")
    if images.shape[0] != len(groups) or grid_size < 2:
        raise ValueError("image/group batch mismatch or invalid grid size")
    if not math.isfinite(float(distance_threshold)) or distance_threshold <= 0:
        raise ValueError("distance_threshold must be finite and positive")
    if not math.isfinite(float(scale_floor)) or scale_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    height, width = images.shape[-2:]
    if height % grid_size or width % grid_size:
        raise ValueError("image dimensions must divide evenly into the patch grid")

    patch_means = F.avg_pool2d(
        images.float(), kernel_size=(height // grid_size, width // grid_size),
        stride=(height // grid_size, width // grid_size),
    ).permute(0, 2, 3, 1).reshape(images.shape[0], grid_size * grid_size, 3)
    coords = torch.arange(grid_size, device=images.device)
    border = ((coords[:, None] == 0) | (coords[:, None] == grid_size - 1)
              | (coords[None, :] == 0) | (coords[None, :] == grid_size - 1)).reshape(-1)

    kept_groups, diagnostics = [], []
    for image_index, image_groups in enumerate(groups):
        border_values = patch_means[image_index, border]
        center = border_values.median(dim=0).values
        mad = (border_values - center).abs().median(dim=0).values
        scale = (1.4826 * mad).clamp_min(float(scale_floor))
        distance = torch.sqrt((((patch_means[image_index] - center) / scale) ** 2).sum(dim=-1))
        retained, group_scores, removed = [], [], []
        for group in image_groups:
            indices = torch.as_tensor(tuple(group), device=images.device, dtype=torch.long)
            if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= grid_size ** 2:
                raise ValueError("component contains an invalid patch index")
            score = float(distance[indices].median())
            group_scores.append(score)
            if score > float(distance_threshold):
                retained.append(group)
            else:
                removed.append(group)
        kept_groups.append(retained)
        diagnostics.append({
            "border_rgb_median": center.detach().cpu().tolist(),
            "border_rgb_robust_scale": scale.detach().cpu().tolist(),
            "component_median_distances": group_scores,
            "removed_group_count": len(removed),
            "removed_patch_count": sum(len(group) for group in removed),
        })
    return kept_groups, diagnostics

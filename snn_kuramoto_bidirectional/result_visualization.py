"""Utilities for visualizing single-sample S2Net object groups."""

from pathlib import Path

import torch
import torch.nn.functional as F


def object_groups_to_grid_masks(object_groups, grid_size):
    """Convert single-sample oscillator groups to binary row-major grid masks.

    Args:
        object_groups:
            List of groups from one classifier sample. Each group is a list,
            tuple, or 1D tensor of zero-based oscillator indices.
        grid_size:
            Positive integer or ``(grid_h, grid_w)`` pair.

    Returns:
        Float tensor shaped ``[K, grid_h, grid_w]`` containing only 0 and 1.
    """
    grid_h, grid_w = _parse_grid_size(grid_size)
    if object_groups is None:
        raise ValueError("object_groups must contain groups for one sample.")
    if not isinstance(object_groups, (list, tuple)):
        raise ValueError("object_groups must be a list or tuple for one sample.")

    num_oscillators = grid_h * grid_w
    masks = torch.zeros(
        len(object_groups),
        grid_h,
        grid_w,
        dtype=torch.float32,
    )
    for group_index, group in enumerate(object_groups):
        indices = torch.as_tensor(group, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError("Each object group must be a one-dimensional index collection.")
        if indices.numel() == 0:
            continue
        if bool((indices < 0).any()) or bool((indices >= num_oscillators).any()):
            raise ValueError(
                f"Oscillator indices must be between 0 and {num_oscillators - 1}."
            )
        rows = torch.div(indices, grid_w, rounding_mode="floor")
        columns = indices.remainder(grid_w)
        masks[group_index, rows, columns] = 1.0
    return masks


def expand_patch_masks(grid_masks, image_size):
    """Expand binary grid masks to image resolution with nearest neighbors."""
    if not torch.is_tensor(grid_masks) or grid_masks.ndim != 3:
        raise ValueError("grid_masks must have shape [K, grid_h, grid_w].")
    image_h, image_w = _parse_image_size(image_size)
    if grid_masks.size(1) <= 0 or grid_masks.size(2) <= 0:
        raise ValueError("grid mask dimensions must be positive.")
    if grid_masks.size(0) == 0:
        return grid_masks.new_empty((0, image_h, image_w))

    masks = F.interpolate(
        grid_masks.float().unsqueeze(1),
        size=(image_h, image_w),
        mode="nearest",
    ).squeeze(1)
    return masks


def visualize_object_groups(
    image,
    object_groups,
    grid_size,
    *,
    save_path=None,
    show=True,
):
    """Visualize binary masks and masked RGB images for one S2Net sample.

    ``image`` must be ``[3, H, W]``. ``object_groups`` must be the classifier
    output for one selected sample, not the outer batch list. Returned tensors
    are detached CPU tensors with masks shaped ``[K, H, W]`` and masked images
    shaped ``[K, 3, H, W]``.
    """
    import matplotlib.pyplot as plt

    display_image = _prepare_display_image(image)
    image_h, image_w = display_image.shape[-2:]
    grid_masks = object_groups_to_grid_masks(object_groups, grid_size)
    masks = expand_patch_masks(grid_masks, (image_h, image_w))
    masked_images = display_image.unsqueeze(0) * masks.unsqueeze(1)

    num_objects = grid_masks.size(0)
    if num_objects == 0:
        figure, axis = plt.subplots(figsize=(6, 3))
        axis.text(
            0.5,
            0.5,
            "No object groups detected",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axis.axis("off")
    else:
        figure, axes = plt.subplots(
            2,
            num_objects,
            figsize=(3.2 * num_objects, 6.0),
            squeeze=False,
        )
        for object_index in range(num_objects):
            axes[0, object_index].imshow(
                masks[object_index].numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            axes[0, object_index].set_title(f"Object {object_index + 1} Mask")
            axes[0, object_index].axis("off")

            axes[1, object_index].imshow(
                masked_images[object_index].permute(1, 2, 0).numpy()
            )
            axes[1, object_index].set_title(
                f"Object {object_index + 1} Masked Image"
            )
            axes[1, object_index].axis("off")
        figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()

    return {
        "figure": figure,
        "grid_masks": grid_masks,
        "masks": masks,
        "masked_images": masked_images,
    }


def _prepare_display_image(image):
    if not torch.is_tensor(image):
        raise ValueError("image must be a torch.Tensor shaped [3, H, W].")
    if image.ndim != 3 or image.size(0) != 3:
        raise ValueError("image must have shape [3, H, W] for one sample.")
    if image.size(1) <= 0 or image.size(2) <= 0:
        raise ValueError("image spatial dimensions must be positive.")

    display_image = image.detach().to(device="cpu", dtype=torch.float32).clone()
    if not image.is_floating_point():
        display_image = display_image / 255.0
    return display_image.clamp(0.0, 1.0)


def _parse_grid_size(grid_size):
    if isinstance(grid_size, int):
        grid_h = grid_w = int(grid_size)
    elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
        grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    else:
        raise ValueError("grid_size must be a positive int or a pair of ints.")
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid dimensions must be positive.")
    return grid_h, grid_w


def _parse_image_size(image_size):
    if isinstance(image_size, int):
        image_h = image_w = int(image_size)
    elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        image_h, image_w = int(image_size[0]), int(image_size[1])
    else:
        raise ValueError("image_size must be a positive int or a pair of ints.")
    if image_h <= 0 or image_w <= 0:
        raise ValueError("image dimensions must be positive.")
    return image_h, image_w

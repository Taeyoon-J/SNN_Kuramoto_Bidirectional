import torch

from .gamma_initializer import FeaturePatchGammaInitializer
from .input_layer_generator import CNNFeatureEncoder


@torch.no_grad()
def gamma_sampes(
    images,
    hparams,
    input_layer_path,
    device=None,
):
    """Create one flat collection of gamma vectors from a batch of images.

    Args:
        images:
            RGB image tensor shaped [num_images, 3, H, W].
        hparams:
            Hyperparameters used to construct the two pretrained encoders.
        input_layer_path:
            Checkpoint containing the trained CNNFeatureEncoder state_dict.
        device:
            Device on which inference is performed. Defaults to CUDA when
            available, otherwise CPU.

    Returns:
        Tensor shaped [num_images * num_feature_maps, num_regions]. Image and
        feature-map boundaries are intentionally flattened into one gamma
        sample dimension.
    """
    if not torch.is_tensor(images):
        images = torch.as_tensor(images, dtype=torch.float32)
    images = images.float()
    if images.dim() != 4:
        raise ValueError("images must have shape [num_images, 3, H, W].")

    device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    input_layer = CNNFeatureEncoder(
        num_kernels=hparams.num_feature_maps,
        kernel_size=hparams.kernel_size,
        in_channels=hparams.in_channels,
        bias=True,
    ).to(device)
    input_layer.load_state_dict(torch.load(input_layer_path, map_location=device))

    gamma_initializer = FeaturePatchGammaInitializer(
        grid_size=hparams.gamma_patch_grid_size,
        patch_size=hparams.gamma_patch_size,
        stride=hparams.gamma_patch_stride,
        reduction=hparams.gamma_patch_reduction,
    ).to(device)

    input_layer.eval()
    gamma_initializer.eval()

    feature_maps = input_layer(images.to(device))
    num_images, num_feature_maps, _, _ = feature_maps.shape
    gamma_seq = gamma_initializer(feature_maps)
    if gamma_seq.size(-1) != hparams.num_regions:
        raise ValueError(
            f"Patch gamma produced {gamma_seq.size(-1)} oscillators, "
            f"but hparams.num_regions is {hparams.num_regions}."
        )
    gamma_samples = gamma_seq.reshape(
        num_images * num_feature_maps,
        hparams.num_regions,
    )
    return gamma_samples.cpu()


def pearson_cor_sc(gamma_samples, hparams=None, eps=1e-8):
    """Build an absolute Pearson-correlation SC matrix from gamma samples.

    Args:
        gamma_samples:
            Tensor shaped [num_gamma_samples, num_regions]. Each column
            represents one brain region.
        eps:
            Small value used to safely handle a region with no variation.
        hparams:
            Optional hyperparameter object. When provided, the generated SC is
            stored in ``hparams.sc`` for later model construction.

    Returns:
        Symmetric tensor shaped [num_regions, num_regions], with values in
        [0, 1]. Positive and negative correlations both become connectivity
        strength through the absolute value.
    """
    if not torch.is_tensor(gamma_samples):
        gamma_samples = torch.as_tensor(gamma_samples, dtype=torch.float32)
    if not gamma_samples.is_floating_point():
        gamma_samples = gamma_samples.float()
    if gamma_samples.dim() != 2:
        raise ValueError(
            "gamma_samples must have shape [num_gamma_samples, num_regions]."
        )
    if gamma_samples.size(0) < 2:
        raise ValueError("At least two gamma samples are required.")

    centered = gamma_samples - gamma_samples.mean(dim=0, keepdim=True)
    column_norms = torch.linalg.vector_norm(centered, dim=0, keepdim=True)
    normalized = centered / column_norms.clamp_min(float(eps))

    correlation = normalized.transpose(0, 1) @ normalized
    sc = correlation.abs().clamp(0.0, 1.0)
    sc = sc.detach().cpu()
    if hparams is not None:
        hparams.sc = sc
    return sc

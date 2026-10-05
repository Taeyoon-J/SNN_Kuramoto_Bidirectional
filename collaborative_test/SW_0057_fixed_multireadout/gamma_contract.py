"""Fixed aligned gamma slice contract shared by evaluator and tests."""


EXPECTED_IDS = [1320, 1639]
EXPECTED_SHAPE = [320, 8, 256]


def aligned_gamma_prefix(gamma, manifest, gamma_sha256, global_start, count, steps, settle):
    """Validate saved manifest and tensor; return only the fixed prefix if preflighting."""
    if global_start != 1320 or count not in (4, 320) or (steps, settle) != (1024, 512):
        raise ValueError("Only fixed IDs1320-1639 long evaluation or its first-four-row preflight is allowed")
    if not isinstance(manifest, dict):
        raise ValueError("Gamma manifest must be a JSON object")
    if manifest.get("image_ids") != EXPECTED_IDS:
        raise ValueError(f"Manifest image_ids must be {EXPECTED_IDS}")
    if manifest.get("shape") != EXPECTED_SHAPE:
        raise ValueError(f"Manifest shape must be {EXPECTED_SHAPE}")
    if manifest.get("patch_grid_size") != 16 or manifest.get("dtype") != "torch.float32":
        raise ValueError("Manifest dtype/patch grid does not match aligned gamma contract")
    if not isinstance(gamma_sha256, str) or manifest.get("gamma_sha256") != gamma_sha256:
        raise ValueError("Manifest gamma checksum does not match the loaded gamma asset")
    if gamma.ndim != 3 or list(gamma.shape) != EXPECTED_SHAPE:
        raise ValueError(f"Expected exact aligned gamma shape {EXPECTED_SHAPE}, got {tuple(gamma.shape)}")
    return gamma[:count]

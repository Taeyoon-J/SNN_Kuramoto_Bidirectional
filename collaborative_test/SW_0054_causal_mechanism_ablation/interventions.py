"""Pure tensor interventions for SW0054 gate-causality controls."""
import torch


def validate_permutation(permutation, n, device):
    permutation = torch.as_tensor(permutation, dtype=torch.long, device=device)
    if permutation.shape != (n,) or not torch.equal(
        torch.sort(permutation).values, torch.arange(n, device=permutation.device)
    ):
        raise ValueError("permutation must contain every region index exactly once")
    return permutation


def apply_carrier_mask_intervention(carrier, local_mask, mode, permutation=None,
                                    mean_mask=None, mean_carrier=None):
    """Intervene on carrier or delayed mask separately; return drive, used carrier, mask.

    carrier is sin(current theta), [B,N,D]; local_mask is the delayed raw gate
    [B,N]. The dendritic drive is their product. Membrane gating follows the
    mode-specific rule documented in the experiment contract.
    """
    if carrier.ndim != 3 or local_mask.ndim != 2 or carrier.shape[:2] != local_mask.shape:
        raise ValueError("expected carrier [B,N,D] and local mask [B,N]")
    used_carrier, used_mask = carrier, local_mask
    if mode in ("gate_perm", "carrier_perm"):
        permutation = validate_permutation(permutation, carrier.shape[1], carrier.device)
        if mode == "gate_perm":
            used_mask = local_mask.index_select(1, permutation)
        else:
            used_carrier = carrier.index_select(1, permutation)
    elif mode == "gate_mean":
        if mean_mask is None or mean_mask.shape != local_mask.shape:
            raise ValueError("gate_mean requires per-image/per-region mean mask")
        used_mask = mean_mask
    elif mode == "carrier_mean":
        if mean_carrier is None or mean_carrier.shape != carrier.shape:
            raise ValueError("carrier_mean requires per-image/per-region mean carrier")
        used_carrier = mean_carrier
    elif mode != "normal":
        raise ValueError(f"unsupported gate intervention {mode}")
    return used_carrier * used_mask.unsqueeze(-1), used_carrier, used_mask


def temporal_means(carrier_history, mask_history):
    """Means for carrier [B,T,N,D] and delayed mask [B,T,N]."""
    if carrier_history.ndim != 4 or mask_history.ndim != 3:
        raise ValueError("expected carrier [B,T,N,D] and mask [B,T,N]")
    if carrier_history.shape[:3] != mask_history.shape:
        raise ValueError("carrier and mask histories must align in B,T,N")
    return carrier_history.mean(dim=1), mask_history.mean(dim=1)

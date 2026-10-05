"""Label-free horizontal reflection helpers for 16x16 graph outputs."""

import torch
import torch.nn.functional as F


def horizontal_flip_permutation(grid_size=16, device=None):
    if not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")
    rows = torch.arange(grid_size, device=device).view(grid_size, 1)
    cols = torch.arange(grid_size - 1, -1, -1, device=device).view(1, grid_size)
    return (rows * grid_size + cols).reshape(-1)


def unflip_adjacency(adjacency, grid_size=16):
    """Map graph coordinates from a horizontally reflected image to source coordinates."""
    if adjacency.ndim != 3 or adjacency.shape[-2:] != (grid_size**2, grid_size**2):
        raise ValueError("adjacency must have shape [B, grid_size**2, grid_size**2]")
    permutation = horizontal_flip_permutation(grid_size, adjacency.device)
    return adjacency.index_select(-2, permutation).index_select(-1, permutation)


def graph_equivariance_mse(original_graph, flipped_graph, grid_size=16):
    if original_graph.shape != flipped_graph.shape:
        raise ValueError("original and flipped graph shapes must match")
    aligned = unflip_adjacency(flipped_graph, grid_size)
    return F.mse_loss(aligned, original_graph.detach()), aligned


def freeze_graph_only(core, encoder):
    if getattr(core, "graph_generator", None) is None:
        raise ValueError("graph-only training requires core.graph_generator")
    core.requires_grad_(False)
    core.graph_generator.requires_grad_(True)
    encoder.requires_grad_(False)
    encoder.eval()
    params = [p for p in core.graph_generator.parameters() if p.requires_grad]
    if not params:
        raise ValueError("graph_generator has no trainable parameters")
    if any(p.requires_grad for n, p in core.named_parameters()
           if not n.startswith("graph_generator.")):
        raise AssertionError("non-graph core parameter left trainable")
    if any(p.requires_grad for p in encoder.parameters()):
        raise AssertionError("encoder parameter left trainable")
    return params


def grad_norm(loss, parameters, retain_graph=False):
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph,
                                allow_unused=True)
    squares = [g.detach().square().sum() for g in grads if g is not None]
    if not squares:
        return loss.new_zeros(())
    return torch.stack(squares).sum().sqrt()


def target_equivariance_weights(binding_grad_norm, equivariance_grad_norm):
    """Weights that make equivariance gradients 0.1x and 1x binding gradients."""
    b = float(binding_grad_norm)
    e = float(equivariance_grad_norm)
    if not (torch.isfinite(torch.tensor([b, e])).all() and b >= 0 and e > 0):
        raise ValueError("gradient norms must be finite, with positive equivariance norm")
    return {"weight_0p1x": 0.1 * b / e, "weight_1x": b / e}

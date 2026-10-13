"""Fixed-variance Gaussian image likelihood for live spatial assignments."""
from __future__ import annotations

import math

import torch
from torch import nn

GRID = 32
PATCHES = GRID * GRID
IMAGE_SIZE = 128
PATCH_PIXELS = 4
FOREGROUND_BASIS = 6
BACKGROUND_BASIS = 3


def calibrate_sigma2(train_rgb: torch.Tensor) -> float:
    """Mean RGB-channel population variance over exactly 64 clean TRAIN images."""
    if not torch.is_tensor(train_rgb) or tuple(train_rgb.shape) != (64, 128, 128, 3):
        raise ValueError("variance calibration requires exactly [64,128,128,3] images")
    if train_rgb.dtype != torch.uint8:
        raise TypeError("variance calibration accepts clean uint8 RGB only")
    pixels = train_rgb.detach().to(device="cpu", dtype=torch.float64) / 255.0
    per_image_channel = pixels.var(dim=(1, 2), unbiased=False)
    value = float(per_image_channel.mean().item())
    if not math.isfinite(value) or value <= 0:
        raise ValueError("calibrated RGB population variance must be finite and positive")
    return value


def _validate_inputs(gamma, assignment, rgb):
    if not torch.is_tensor(gamma) or gamma.ndim != 3 or gamma.shape[1] != PATCHES:
        raise ValueError("gamma must have shape [B,1024,D]")
    if not gamma.is_floating_point() or gamma.shape[0] < 1 or gamma.shape[2] < 1:
        raise TypeError("gamma must be a nonempty floating-point feature tensor")
    if not torch.isfinite(gamma).all():
        raise ValueError("gamma must be finite")
    if (not torch.is_tensor(assignment) or assignment.ndim != 3
            or assignment.shape[0] != gamma.shape[0] or assignment.shape[1] != PATCHES
            or not 1 <= assignment.shape[2] <= 11):
        raise ValueError("P must have shape [B,1024,K] with 1<=K<=11")
    if not assignment.is_floating_point():
        raise TypeError("P must be floating point")
    if assignment.dtype != gamma.dtype:
        raise TypeError("gamma and P must use the same floating-point dtype")
    if assignment.device != gamma.device:
        raise ValueError("gamma and P must be on the same device")
    if not torch.isfinite(assignment).all() or torch.any(assignment < 0):
        raise ValueError("P must be finite and nonnegative")
    if not torch.allclose(assignment.sum(dim=-1),
                          torch.ones_like(assignment[..., 0]), rtol=0, atol=1e-6):
        raise ValueError("P rows must sum to one")
    if (not torch.is_tensor(rgb) or tuple(rgb.shape) !=
            (gamma.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3)):
        raise ValueError("RGB target must have shape [B,128,128,3]")
    if rgb.dtype != torch.uint8:
        raise TypeError("RGB likelihood targets must be clean uint8 images")
    return gamma.shape[0], gamma.shape[2], assignment.shape[2]


def _coordinates(device, dtype):
    pixel_axis = (torch.arange(IMAGE_SIZE, device=device, dtype=dtype) + 0.5) * (2.0 / IMAGE_SIZE) - 1.0
    yy, xx = torch.meshgrid(pixel_axis, pixel_axis, indexing="ij")
    pixels = torch.stack((xx, yy), dim=-1)
    # Average the actual centers of the 4x4 pixels belonging to each patch.
    patch_centers = pixels.reshape(GRID, PATCH_PIXELS, GRID, PATCH_PIXELS, 2).mean(dim=(1, 3))
    return pixels, patch_centers


def foreground_basis(pixel_xy, center_xy):
    """Quadratic six-term basis at pixels relative to live patch centers."""
    relative = pixel_xy[None, :, :, None, :] - center_xy[:, None, None, :, :]
    x, y = relative.unbind(dim=-1)
    return torch.stack((torch.ones_like(x), x, y, x.square(), x * y, y.square()), dim=-1)


def background_basis(pixel_xy):
    x, y = pixel_xy.unbind(dim=-1)
    return torch.stack((torch.ones_like(x), x, y), dim=-1)


class SpatialMixtureDecoder(nn.Module):
    """Shared foreground quadratic RGB head and per-image affine background head."""
    def __init__(self, content_dim: int):
        super().__init__()
        if isinstance(content_dim, bool) or not isinstance(content_dim, int) or content_dim < 1:
            raise ValueError("content_dim must be a positive integer")
        self.content_dim = content_dim
        self.foreground = nn.Sequential(nn.Linear(content_dim, 32), nn.SiLU(),
                                        nn.Linear(32, 3 * FOREGROUND_BASIS))
        self.background = nn.Sequential(nn.Linear(content_dim, 16), nn.SiLU(),
                                        nn.Linear(16, 3 * BACKGROUND_BASIS))

    def component_means(self, gamma, assignment):
        if (not torch.is_tensor(gamma) or gamma.ndim != 3 or gamma.shape[1:] !=
                (PATCHES, self.content_dim)):
            raise ValueError("gamma content shape does not match decoder/native32 grid")
        if assignment.ndim != 3 or assignment.shape[:2] != gamma.shape[:2]:
            raise ValueError("P shape does not match gamma batch and native32 grid")
        if not gamma.is_floating_point() or not assignment.is_floating_point():
            raise TypeError("gamma and P must be floating point")
        if gamma.device != assignment.device or gamma.device != next(self.parameters()).device:
            raise ValueError("decoder, gamma, and P must share a device")
        batch, _, slots = assignment.shape
        content = gamma.detach()
        mass = assignment.sum(dim=1)
        safe_mass = mass.clamp_min(torch.finfo(assignment.dtype).tiny)
        pooled = torch.einsum("bnk,bnd->bkd", assignment, content) / safe_mass.unsqueeze(-1)
        _, patch_xy = _coordinates(gamma.device, gamma.dtype)
        centers = torch.einsum("bnk,nd->bkd", assignment, patch_xy.reshape(PATCHES, 2))
        centers = centers / safe_mass.to(centers.dtype).unsqueeze(-1)
        pixel_xy, _ = _coordinates(gamma.device, gamma.dtype)
        bg_coeff = self.background(pooled[:, 0]).reshape(batch, 3, BACKGROUND_BASIS)
        bg_basis = background_basis(pixel_xy)
        bg_means = torch.einsum("bcd,hwd->bhwc", bg_coeff, bg_basis).unsqueeze(3)
        if slots == 1:
            return bg_means
        fg_coeff = self.foreground(pooled[:, 1:]).reshape(
            batch, slots - 1, 3, FOREGROUND_BASIS)
        fg_basis = foreground_basis(pixel_xy, centers[:, 1:])
        fg_means = torch.einsum("bkcl,bhwkl->bhwkc", fg_coeff, fg_basis)
        return torch.cat((bg_means, fg_means), dim=3)


def make_decoder(content_dim: int, *, seed: int) -> SpatialMixtureDecoder:
    """Create deterministically without changing the caller's global CPU RNG."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("decoder initialization seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return SpatialMixtureDecoder(content_dim)


def gaussian_mixture_nll(gamma, assignment, rgb_uint8, decoder, *, sigma2: float):
    """Mean-pixel three-channel Gaussian mixture NLL with live assignment weights."""
    batch, _, slots = _validate_inputs(gamma, assignment, rgb_uint8)
    if not isinstance(decoder, SpatialMixtureDecoder) or decoder.content_dim != gamma.shape[2]:
        raise ValueError("decoder type/content dimension does not match gamma")
    if isinstance(sigma2, bool) or not isinstance(sigma2, (float, int)):
        raise TypeError("fixed sigma2 must be a scalar")
    sigma2 = float(sigma2)
    if not math.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("fixed sigma2 must be finite and positive")
    means = decoder.component_means(gamma, assignment)
    if means.shape != (batch, IMAGE_SIZE, IMAGE_SIZE, slots, 3):
        raise ValueError("decoder returned invalid component mean shape")
    p_pixels = (assignment.reshape(batch, GRID, GRID, slots)
                .repeat_interleave(PATCH_PIXELS, dim=1)
                .repeat_interleave(PATCH_PIXELS, dim=2))
    target = rgb_uint8.to(device=means.device, dtype=means.dtype) / 255.0
    residual = target.unsqueeze(3) - means
    log_density = -0.5 * (residual.square().sum(dim=-1) / sigma2
                           + 3.0 * math.log(2.0 * math.pi * sigma2))
    # Clamp before log to avoid constructing log(0), then mask zero-mass slots;
    # masked slots have no assignment or decoder gradient.
    tiny = torch.finfo(p_pixels.dtype).tiny
    log_weight = p_pixels.clamp_min(tiny).log().masked_fill(p_pixels <= 0, -torch.inf)
    log_mixture = torch.logsumexp(log_weight + log_density, dim=-1)
    if not torch.isfinite(log_mixture).all():
        raise FloatingPointError("Gaussian mixture produced a nonfinite per-pixel log density")
    return -log_mixture.mean()

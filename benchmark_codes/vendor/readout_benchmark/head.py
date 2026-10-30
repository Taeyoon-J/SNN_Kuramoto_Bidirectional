"""Trainable final masks from actual SW240 emitted activity, without GT at inference."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FEATURES = 144


def activity_features(trace, settle=512, bins=32):
    s = trace['component_spikes'][..., settle:].float()
    m = trace['component_membrane'][..., settle:].float()
    if s.ndim != 4 or s.shape[1] != 4 or m.shape != s.shape:
        raise ValueError('actual aligned [B,4,N,T] spike and membrane histories required')
    if s.shape[-1] < bins or s.shape[-1] % bins:
        raise ValueError('settled time history must divide into the declared bins')
    b, c, n, t = s.shape
    temporal = s.reshape(b, c, n, bins, t // bins).mean(-1).permute(0, 2, 1, 3).reshape(b, n, c*bins)
    stats = [x.permute(0, 2, 1) for x in
             (s.mean(-1), s.std(-1, unbiased=False), m.mean(-1), m.std(-1, unbiased=False))]
    features = torch.cat([temporal, *stats], dim=-1)
    if features.shape[-1] != FEATURES or not bool(torch.isfinite(features).all()):
        raise ValueError('finite 144-channel actual activity features required')
    return features


def candidate_masks(baseline, proposals, *, background_only=False, extra_limit=48, features=None):
    """No target input; preserve every original positive group without truncation."""
    base = torch.as_tensor(baseline, dtype=torch.long).flatten()
    prop = torch.as_tensor(proposals, device=base.device, dtype=torch.long).flatten()
    if base.shape != prop.shape or bool((base < 0).any() or (prop < 0).any()):
        raise ValueError('aligned nonnegative prediction grids required')
    reference = prop if background_only else base
    ids = torch.unique(reference)
    masks = [(reference == i) for i in ids.tolist() if i > 0]
    original_count = len(masks)
    if not background_only:
        extras = [(int((prop == i).sum()), int(i)) for i in torch.unique(prop).tolist() if i > 0]
        for _, i in sorted(extras, reverse=True):
            proposed = prop == i
            if masks:
                existing = torch.stack(masks)
                intersection = (existing & proposed).sum(1).float()
                union = (existing | proposed).sum(1).clamp_min(1)
                if bool((intersection / union > .95).any()):
                    continue
            masks.append(proposed)
            if len(masks) - original_count >= extra_limit:
                break
    if not masks:
        masks = [torch.ones_like(base, dtype=torch.bool)]
    if not background_only and features is not None:
        # Twelve spike-derived proposals, independent of GT object count.
        with torch.no_grad():
            z = features.detach().float()
            z = (z-z.mean(0)) / z.std(0, unbiased=False).clamp_min(.05)
            z = F.normalize(z, dim=-1)
            centers, nearest = [], torch.full((len(z),), float('inf'), device=z.device)
            for _ in range(min(12, len(z))):
                if centers:
                    nearest[torch.as_tensor(centers, device=z.device)] = -1
                index = int(nearest.argmax()) if centers else int((z-z.mean(0)).square().sum(1).argmax())
                centers.append(index)
                nearest = torch.minimum(nearest, (z-z[index]).square().sum(1))
            distances = torch.stack([(z-z[i]).square().sum(1) for i in centers])
            assignment = distances.argmin(0)
            for j, i in enumerate(centers):
                assignment[i] = j
            masks.extend(assignment == j for j in range(len(centers)))
    masks = torch.stack(masks)
    prior = torch.zeros((len(masks)+1, len(base)), device=base.device)
    if background_only:
        prior[1:] = 4.0 * masks.float()
    else:
        prior[0] = 4.0 * (base == 0)
        for j in range(original_count):
            prior[j+1] = 4.0 * masks[j].float()
    return masks, prior


class ActivityMaskHead(nn.Module):
    def __init__(self, *, background_only=False, seed=251000):
        super().__init__()
        self.background_only = bool(background_only)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.embedding = nn.Sequential(nn.LayerNorm(FEATURES), nn.Linear(FEATURES, 64), nn.SiLU(), nn.Linear(64, 32))
            self.background = nn.Sequential(nn.LayerNorm(FEATURES), nn.Linear(FEATURES, 32), nn.SiLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.background[-1].weight)
            nn.init.zeros_(self.background[-1].bias)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, features, baseline, proposals):
        if features.ndim != 2 or features.shape[1] != FEATURES or not bool(torch.isfinite(features).all()):
            raise ValueError('one image of finite [N,144] activity features required')
        base = torch.as_tensor(baseline, device=features.device, dtype=torch.long)
        prop = torch.as_tensor(proposals, device=features.device, dtype=torch.long)
        if base.numel() != len(features) or prop.shape != base.shape:
            raise ValueError('prediction and feature grids differ')
        masks, prior = candidate_masks(base, prop, background_only=self.background_only, features=features)
        bg = self.background(features).T
        if self.background_only:
            positive = prior[1:]
        else:
            z = F.normalize(self.embedding(features), dim=-1)
            seeds = F.normalize(masks.float() @ z / masks.sum(1, keepdim=True).clamp_min(1), dim=-1)
            positive = prior[1:] + self.residual_scale * (seeds @ z.T)
        return torch.cat([prior[:1] + bg, positive], dim=0)

    @torch.no_grad()
    def predict(self, features, baseline, proposals):
        logits = self(features, baseline, proposals)
        return logits.argmax(0).reshape(np.shape(baseline)).cpu().numpy().astype(np.int64)


def mask_objective(logits, labels, baseline, *, anchor_weight=.1, pair_weight=.25):
    """Hungarian final-mask CE/Dice + background + foreground-pair loss.

    GT participates only here, never in candidate generation or prediction.
    """
    from scipy.optimize import linear_sum_assignment

    truth = torch.as_tensor(labels, device=logits.device, dtype=torch.long).flatten()
    base = torch.as_tensor(baseline, device=logits.device, dtype=torch.long).flatten()
    if len(truth) != logits.shape[1] or base.shape != truth.shape or bool((truth < 0).any()):
        raise ValueError('aligned nonnegative TRAIN mask required')
    classes = torch.unique(truth)
    objects = classes[classes > 0]
    if len(objects):
        objects = torch.stack(sorted(objects, key=lambda k: int((truth == k).nonzero()[0])))
    if len(objects) > logits.shape[0] - 1:
        raise ValueError('inference candidate pool cannot represent TRAIN object count; do not use GT seeds')
    probability = logits.softmax(0)
    target = torch.zeros_like(truth)
    dice_terms = []
    if len(objects):
        masks = (truth[None] == objects[:, None]).float()
        intersection = probability[1:] @ masks.T
        denom = probability[1:].sum(1)[:, None] + masks.sum(1)[None]
        dice = (2 * intersection + 1e-6) / (denom + 1e-6)
        rows, cols = linear_sum_assignment(-dice.detach().cpu().numpy())
        if len(cols) != len(objects):
            raise ValueError('incomplete permutation-invariant assignment')
        for row, col in zip(rows, cols):
            target[truth == objects[col]] = int(row)+1
            dice_terms.append(1-dice[row, col])
    logp = logits.log_softmax(0)
    ce_terms = [-logp[target[truth == k], (truth == k).nonzero().flatten()].mean() for k in classes]
    ce = torch.stack(ce_terms).mean()
    bg_truth = (truth == 0).float()
    bg_probability = probability[0].clamp(1e-6, 1-1e-6)
    bg_terms = [F.binary_cross_entropy(bg_probability[truth == 0], bg_truth[truth == 0])] if bool((truth == 0).any()) else []
    if bool((truth > 0).any()):
        bg_terms.append(F.binary_cross_entropy(bg_probability[truth > 0], bg_truth[truth > 0]))
    background = torch.stack(bg_terms).mean()
    bg_dice = 1-(2*(probability[0]*bg_truth).sum()+1e-6)/(probability[0].sum()+bg_truth.sum()+1e-6)
    dice_loss = torch.stack([bg_dice, *dice_terms]).mean()
    # Deterministic balanced within-object and between-object pairs, no RNG leak.
    same_parts, different_parts = [], []
    conditional = probability[1:] / probability[1:].sum(0, keepdim=True).clamp_min(1e-6)
    for index, obj in enumerate(objects):
        a = (truth == obj).nonzero().flatten()
        if len(a) > 1:
            a = a[torch.linspace(0, len(a)-1, min(256, len(a)), device=a.device).long()]
            similarity = (conditional[:, a] * conditional[:, a.roll(1)]).sum(0).clamp(1e-6, 1-1e-6)
            same_parts.append(-similarity.log().mean())
        other = (truth == objects[(index+1) % len(objects)]).nonzero().flatten()
        if len(objects) > 1 and len(a) and len(other):
            b = other[torch.arange(len(a), device=a.device) % len(other)]
            similarity = (conditional[:, a] * conditional[:, b]).sum(0).clamp(1e-6, 1-1e-6)
            different_parts.append(-(1-similarity).log().mean())
    pair = torch.stack([*same_parts, *different_parts]).mean() if same_parts or different_parts else logits.sum()*0
    # Anchor at the original *partition*, independent of its integer ID names.
    # The reference partition is generated by the model, not by truth.
    unique = torch.unique(base)
    masses = []
    for k in unique:
        region = base == k
        p = probability[:, region].mean(1)
        masses.append(-(p.square().sum().clamp_min(1e-6)).log())
    anchor = torch.stack(masses).mean()
    total = ce + dice_loss + .25*background + pair_weight*pair + anchor_weight*anchor
    parts = dict(mask_ce=ce, mask_dice=dice_loss, background_bce=background, foreground_pair=pair, partition_anchor=anchor)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('nonfinite final mask objective')
    return total, parts

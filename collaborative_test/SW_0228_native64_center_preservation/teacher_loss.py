"""Preserve natural source centers while denoising; no invented unit margin."""
import math
import numpy as np
import torch
from collaborative_test.SW_0222_native64_frequency_teacher_denoising import teacher_loss as base

CENTER_WEIGHT = 16.
BASE_KEYS = ('gamma_reference', 'frequency_centers', 'class_weights')


def make_teacher(gamma, frequency, labels):
    targets = base.make_teacher(gamma, frequency, labels)
    targets['class_labels'] = torch.from_numpy(np.asarray(labels, dtype=np.int64).reshape(len(labels), -1).copy())
    return targets


def teacher_objective(gamma, frequency, targets, *, center_weight=CENTER_WEIGHT):
    if not math.isfinite(center_weight) or center_weight < 1:
        raise ValueError('finite source-center weight at least1 required')
    original, parts = base.teacher_objective(gamma, frequency, targets)
    labels = targets['class_labels'].to(device=frequency.device)
    if labels.dtype != torch.int64 or labels.shape != frequency.shape[:2] or bool((labels < 0).any()):
        raise ValueError('unaltered aligned TRAIN class labels required')
    centers = targets['frequency_centers'].to(device=frequency.device, dtype=frequency.dtype)
    image_losses = []
    for b in range(len(labels)):
        class_losses = []
        for label in torch.unique(labels[b]):
            take = labels[b] == label
            mean = frequency[b, take].mean(0)
            target = centers[b, take][0]
            class_losses.append((mean - target).square().mean())
        image_losses.append(torch.stack(class_losses).mean())
    center_loss = torch.stack(image_losses).mean()
    # The original denoising term equals within-class variance + center MSE.
    # Add only extra center preservation; retain its within-class coefficient1.
    loss = original + (center_weight - 1.) * center_loss
    return loss, dict(parts, source_center_MSE=center_loss,
                      within_object_variance=parts['frequency_denoising'] - center_loss)

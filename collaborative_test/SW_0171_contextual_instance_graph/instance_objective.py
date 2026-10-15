"""Leave-one-out, class-balanced spherical prototype CE for native graph z."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

EPS = 1e-12
TEMPERATURE = 0.1
_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def normalized_projection_output(raw: torch.Tensor) -> torch.Tensor:
    """Apply exactly the native graph's F.normalize(dim=-1, eps=1e-12)."""
    if not torch.is_tensor(raw) or raw.ndim != 3 or raw.shape[-1] < 1:
        raise ValueError("projection output must have shape [B,N,D]")
    norms = torch.linalg.vector_norm(raw, dim=-1)
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(norms).all()):
        raise FloatingPointError("graph projection produced a nonfinite embedding")
    if bool((norms <= EPS).any()):
        raise FloatingPointError("graph projection norm is at or below native epsilon")
    return F.normalize(raw, dim=-1, eps=EPS)


@contextmanager
def capture_native_projection(graph):
    """Capture and normalize the actual graph projection from one forward.

    Use around the real graph forward. The hook does not invoke the projection a
    second time and never detaches its output. It is removed even when forward or
    validation raises.
    """
    projection = getattr(graph, "projection", None)
    if not isinstance(projection, torch.nn.Module):
        raise TypeError("native graph must expose its projection module")
    captured = []

    def _hook(_module, _inputs, output):
        if captured:
            raise RuntimeError("native graph projection ran more than once in captured forward")
        captured.append(normalized_projection_output(output))

    handle = projection.register_forward_hook(_hook)
    try:
        yield captured
        if len(captured) != 1:
            raise RuntimeError("native graph projection was not captured exactly once")
    finally:
        handle.remove()


def _validate_inputs(z: torch.Tensor, labels) -> torch.Tensor:
    if not torch.is_tensor(z) or z.ndim != 3 or min(z.shape) < 1:
        raise ValueError("z must have nonempty shape [B,N,D]")
    labels = torch.as_tensor(labels, device=z.device)
    if labels.ndim != 2 or tuple(labels.shape) != tuple(z.shape[:2]):
        raise ValueError("instance labels must have shape [B,N] aligned with z")
    if labels.dtype not in _INTEGER_DTYPES:
        raise TypeError("instance labels must use an integer dtype")
    if bool((labels < 0).any()):
        raise ValueError("instance labels must be nonnegative; ID 0 is background")
    if not bool(torch.isfinite(z).all()):
        raise FloatingPointError("normalized graph embeddings contain nonfinite values")
    norms = torch.linalg.vector_norm(z, dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= EPS).any()):
        raise FloatingPointError("embedding norm is at or below native epsilon")
    unit = torch.ones_like(norms)
    if not torch.allclose(norms, unit, rtol=1e-5, atol=1e-6):
        raise ValueError("z must be the already normalized native graph projection")
    return labels.to(dtype=torch.int64)


def leave_one_out_instance_loss(z: torch.Tensor, labels, *, return_stats=False):
    """Mean per-image class-balanced LOO prototype CE, including background.

    ``z`` is the already normalized output captured from the actual graph
    projection. Singleton-class samples are excluded as queries, but their full
    prototypes remain among the negative classes for other queries.
    """
    labels = _validate_inputs(z, labels)
    per_image = []
    rows = []
    for batch_index in range(z.shape[0]):
        image_z = z[batch_index]
        image_labels = labels[batch_index]
        class_ids, counts = torch.unique(image_labels, sorted=True, return_counts=True)
        count_map = {int(class_id.item()): int(count.item())
                     for class_id, count in zip(class_ids, counts)}
        classes = [int(value.item()) for value in class_ids]
        if len(classes) < 2:
            per_image.append(image_z.sum() * 0.0)
            rows.append({"visible_classes": len(classes), "valid_query_classes": 0,
                         "excluded_singleton_queries": int(sum(v == 1 for v in count_map.values())),
                         "valid_queries": 0, "excluded_queries": int(image_z.shape[0]),
                         "excluded_query_classes": len(classes), "zero_loss": True,
                         "zero_loss_reason": "fewer_than_two_visible_classes"})
            continue

        class_sums = []
        full_prototypes = []
        for class_id in classes:
            members = image_z[image_labels == class_id]
            class_sum = members.sum(dim=0)
            mean = class_sum / count_map[class_id]
            norm = torch.linalg.vector_norm(mean)
            if not bool(torch.isfinite(norm)) or bool(norm <= EPS):
                raise FloatingPointError("full class prototype norm is at or below native epsilon")
            class_sums.append(class_sum)
            full_prototypes.append(F.normalize(mean, dim=0, eps=EPS))
        prototype_matrix = torch.stack(full_prototypes, dim=0)

        class_losses = []
        valid_queries = 0
        for class_index, class_id in enumerate(classes):
            count = count_map[class_id]
            if count < 2:
                continue
            queries = image_z[image_labels == class_id]
            loo_mean = (class_sums[class_index].unsqueeze(0) - queries) / (count - 1)
            loo_norm = torch.linalg.vector_norm(loo_mean, dim=-1)
            if (not bool(torch.isfinite(loo_norm).all())
                    or bool((loo_norm <= EPS).any())):
                raise FloatingPointError("leave-one-out prototype norm is at or below native epsilon")
            loo_prototype = F.normalize(loo_mean, dim=-1, eps=EPS)
            logits = queries @ prototype_matrix.transpose(0, 1)
            logits = torch.cat((logits[:, :class_index],
                                (queries * loo_prototype).sum(dim=-1, keepdim=True),
                                logits[:, class_index + 1:]), dim=1)
            targets = torch.full((queries.shape[0],), class_index,
                                 dtype=torch.int64, device=z.device)
            class_losses.append(F.cross_entropy(logits / TEMPERATURE, targets))
            valid_queries += int(queries.shape[0])

        if class_losses:
            per_image.append(torch.stack(class_losses).mean())
            zero_loss = False
            zero_loss_reason = None
        else:
            per_image.append(image_z.sum() * 0.0)
            zero_loss = True
            zero_loss_reason = "no_class_with_two_or_more_members"
        singleton_queries = int(sum(v for v in count_map.values() if v == 1))
        rows.append({"visible_classes": len(classes),
                     "valid_query_classes": len(class_losses),
                     "excluded_singleton_queries": singleton_queries,
                     "valid_queries": valid_queries,
                     "excluded_queries": int(image_z.shape[0]) - valid_queries,
                     "excluded_query_classes": sum(v == 1 for v in count_map.values()),
                     "zero_loss": zero_loss, "zero_loss_reason": zero_loss_reason})

    image_losses = torch.stack(per_image)
    loss = image_losses.mean()  # zero-loss images remain in the B4 image mean
    if return_stats:
        return loss, {"per_image_loss": [float(value.detach().cpu()) for value in image_losses],
                      "images": rows,
                      "excluded_singleton_queries": sum(r["excluded_singleton_queries"] for r in rows),
                      "excluded_queries": sum(r["excluded_queries"] for r in rows),
                      "excluded_query_classes": sum(r["excluded_query_classes"] for r in rows),
                      "zero_loss_images": sum(bool(r["zero_loss"]) for r in rows)}
    return loss

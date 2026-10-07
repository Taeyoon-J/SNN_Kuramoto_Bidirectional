"""Batch-8, validation-only stage-flow audit for frozen SW0095/97/98 cores."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import rankdata

ROOT = (Path(os.environ["SW0099_REPO_ROOT"]) if "SW0099_REPO_ROOT" in os.environ
        else Path(__file__).resolve().parents[2])
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test/SW_0094_aligned_joint_pilot"),
                str(ROOT / "collaborative_test")]
from SW_0094_aligned_joint_pilot.run import hparams, VAL_GAMMA, DATASET
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import (
    clevr_mask_patch, evaluate_patch_masks, spatial_components_to_patch_labels)
from snn_kuramoto_bidirectional.loss_function import phase_locking_value
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components

OUT = ROOT / "trained_models/SW0099_stage_flow_diagnosis"
IDS = list(range(1320, 1336))
DISTANCE_BINS = ((0, 2), (2, 5), (5, 10), (10, float("inf")))


def auc_stats(matrix, labels):
    """Foreground-only pair AUC and same-minus-different within fixed Euclidean bins."""
    matrix = matrix.detach().float().cpu()
    labels = labels.detach().cpu()
    n = labels.numel()
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    coords = torch.stack((yy.flatten(), xx.flatten()), dim=1)
    dist = torch.cdist(coords.float(), coords.float())
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
    fg = (labels[:, None] > 0) & (labels[None, :] > 0) & upper
    same = labels[:, None] == labels[None, :]

    def one(mask):
        target = same[mask].cpu().numpy()
        values = matrix[mask].detach().float().cpu().numpy()
        pos, neg = int(target.sum()), int((~target).sum())
        if not pos or not neg:
            return {"pairs": len(values), "same_pairs": pos, "different_pairs": neg,
                    "auc": None, "same_mean": None, "different_mean": None}
        ranks = rankdata(values, method="average")
        auc = (ranks[target].sum() - pos * (pos + 1) / 2.) / (pos * neg)
        return {"pairs": len(values), "same_pairs": pos, "different_pairs": neg,
                "auc": float(auc), "same_mean": float(values[target].mean()),
                "different_mean": float(values[~target].mean()),
                "same_minus_different": float(values[target].mean() - values[~target].mean())}

    result = {"overall": one(fg), "distance_bins_euclidean_patches": {}}
    for lo, hi in DISTANCE_BINS:
        selected = fg & (dist > lo) & (dist <= hi)
        result["distance_bins_euclidean_patches"][f"{lo:g}-{hi:g}"] = one(selected)
    return result


def edge_connectivity(matrix, labels):
    """GT-object induced connectivity after thresholding an undirected score graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    a = (matrix.detach().cpu().numpy() >= .5)
    a = np.logical_or(a, a.T)
    np.fill_diagonal(a, False)
    lab = labels.detach().cpu().numpy()
    output = []
    for obj in sorted(int(v) for v in np.unique(lab) if v > 0):
        nodes = np.flatnonzero(lab == obj)
        if len(nodes) < 2:
            continue
        count, cc = connected_components(csr_matrix(a[np.ix_(nodes, nodes)]), directed=False)
        sizes = np.bincount(cc, minlength=count)
        output.append({"object_id": obj, "patches": int(len(nodes)),
                       "largest_component_fraction": float(sizes.max() / len(nodes)),
                       "fragments": int(count)})
    return output


def read_labels(dataset, ids):
    masks = torch.from_numpy(dataset["mask"][ids])
    return clevr_mask_patch(masks, 8)["patch_labels"].reshape(len(ids), 256)


def validate_dendrite_fold_order():
    """Synthetic sentinel confirms hook [B*D,T,N] fold/unfold order."""
    bsz, comps, steps, nodes = 2, 4, 3, 5
    folded = torch.arange(bsz * comps * steps * nodes).reshape(bsz * comps, steps, nodes)
    restored = folded.reshape(bsz, comps, steps, nodes).permute(0, 1, 3, 2)
    for b in range(bsz):
        for d in range(comps):
            for n in range(nodes):
                for t in range(steps):
                    expected = ((b * comps + d) * steps + t) * nodes + n
                    if int(restored[b, d, n, t]) != expected:
                        raise AssertionError("dendritic component/time ordering was mixed")
    return True


def load_checkpoint(path, device):
    core = _core(device, path, 1024, "shared", 3, 1.5, 2., .5, 16., .35,
                 "factorized", "raw")
    core.eval()
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    return core


def sinusoidal_raw_gate(theta):
    """Return the raw-mode delayed scalar gate and its per-component drive."""
    delayed = torch.cat((theta[:, :1].expand(-1, 2, -1, -1), theta[:, :-2]), dim=1)
    mask = .5 * (1. + delayed.mean(dim=-1).sin())
    return mask, theta.sin() * mask.unsqueeze(-1)


def stage_signals(core, gamma):
    dendrite_rows = []
    handle = core.dendric_layer.register_forward_hook(
        lambda _m, _i, out: dendrite_rows.append(out.detach()))
    try:
        with torch.inference_mode():
            graph = core.graph_generator(gamma)
            _, spikes, membrane, theta = core(
                gamma, return_core_out=True, return_theta=True, num_time_steps=1024)
    finally:
        handle.remove()
    # Preserve the implementation's carrier*delayed-mask exactly: sin(theta[t])
    # multiplied by 0.5*(1+sin(mean(theta[t-2]))), gate_mode="raw".
    carrier = theta.sin()
    mask, gated = sinusoidal_raw_gate(theta)
    dendrite = torch.stack(dendrite_rows, dim=1)
    # Hook outputs are [B*D,N] per step; stack(dim=1) is [B*D,T,N].
    # Unfold B,D before moving time last. Do not reinterpret as T-major.
    dendrite = dendrite.reshape(gamma.shape[0], 4, 1024, 256).permute(0, 1, 3, 2)
    comp_mem = core.last_component_out
    comp_spikes = core.last_component_spikes
    if comp_mem is None or comp_spikes is None:
        raise RuntimeError("expected per-component membrane and spike histories")
    return graph, theta, carrier, mask, gated, dendrite, comp_mem, comp_spikes, spikes


def measure_batch(signals, labels):
    graph, theta, carrier, gate, gated, dendrite, comp_mem, comp_spikes, spikes = signals
    # Mean over oscillator components is the exact inference classifier signal.
    spike_aff = aligned_affinity_from_components(spikes, comp_spikes)
    # Membrane companion uses the same clamped-positive product Pearson rule.
    mem_aff = product_affinity(comp_mem)
    # Scalar stage histories use Pearson correlation; theta/carrier use the
    # product of per-component Pearson correlations before positive clipping.
    stages = {
        "graph_adjacency": graph,
        "phase_theta_product": phase_locking_value(theta, settle=512, combine="product"),
        "sinusoidal_carrier_product": product_affinity(carrier.permute(0, 3, 2, 1)),
        "delayed_gate": pearson_affinity(gate.permute(0, 2, 1), settle=512),
        "exact_gated_carrier_product": product_affinity(gated.permute(0, 3, 2, 1)),
        "dendritic_output_product": product_affinity(dendrite),
        "membrane_positive_product": mem_aff,
        "classifier_spike_positive_product": spike_aff,
    }
    rows = []
    preds = spike_synchrony_components(
        spikes.cpu(), synchrony_threshold=.5, min_group_size=2, settle=512,
        components=comp_spikes.cpu(), background="largest_component")
    pred_labels = spatial_components_to_patch_labels(preds, 16)
    for b in range(labels.shape[0]):
        stages_for_image = {}
        for name, matrix in stages.items():
            metric = auc_stats(matrix[b], labels[b])
            if name in ("membrane_positive_product", "classifier_spike_positive_product"):
                metric["threshold_0p5_edges"] = edge_connectivity(matrix[b], labels[b])
                labels_device = labels[b].to(matrix.device)
                upper = torch.triu(torch.ones_like(matrix[b], dtype=torch.bool), diagonal=1)
                fg = (labels_device[:, None] > 0) & (labels_device[None, :] > 0) & upper
                same = labels_device[:, None] == labels_device[None, :]
                edge = matrix[b] >= .5
                pos, neg = fg & same, fg & ~same
                metric["within_object_edge_recall_and_cross_object_false_edge_rate"] = {
                    "within_object_edge_recall": float(edge[pos].float().mean()) if pos.any() else None,
                    "cross_object_false_edge_rate": float(edge[neg].float().mean()) if neg.any() else None}
            stages_for_image[name] = metric
        target = labels[b]
        # Endpoint scores use the original patch-level metric implementation.
        score = evaluate_patch_masks(
            pred_labels[b:b+1], target.reshape(1, 16, 16))["per_image"]
        stages_for_image["endpoint"] = {
            "fg_ari": float(score["fg_ari"][0]),
            "foreground_iou": float(score["foreground_iou"][0]),
            "matched_object_iou": float(score["matched_object_iou"][0]),
            "predicted_groups": len(preds[b]),
            "target_groups": int(torch.unique(target[target > 0]).numel())}
        rows.append(stages_for_image)
    return rows


def pearson_affinity(x, eps=1e-8, settle=512):
    x = x[..., int(settle):]
    x = x.float() - x.float().mean(dim=-1, keepdim=True)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (x @ x.transpose(-1, -2)).clamp(-1., 1.)


def product_affinity(x, settle=512):
    # Input [B,D,N,T], output product across component Pearson affinities.
    per = [pearson_affinity(x[:, d], settle=settle) for d in range(x.shape[1])]
    return torch.stack(per).clamp_min(0.).prod(dim=0)


def aligned_affinity_from_components(activity, components):
    # Same helper and temporal settle used by SW0098 inference.
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    return spike_synchrony_affinity(activity, components, settle=512)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


CHECKPOINTS = [
    ("SW0095", i, ROOT / f"trained_models/SW0095_full70k_aligned_loss/seed{i}/core.pt")
    for i in range(3)
] + [
    ("SW0097", i, ROOT / f"trained_models/SW0097_graph_adaptation/seed{i}_positive_frozen/core.pt")
    for i in range(3)
] + [
    ("SW0098", i, ROOT / f"trained_models/SW0098_long_window/seed{i}/core.pt")
    for i in range(3)
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", help="single checkpoint for smoke or one queued run")
    p.add_argument("--condition", help="condition label")
    p.add_argument("--seed", type=int)
    p.add_argument("--ids", nargs=2, type=int, default=(1320, 1336))
    p.add_argument("--output", type=Path)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--order-check-only", action="store_true")
    args = p.parse_args()
    if args.order_check_only:
        torch.set_num_threads(2)
        validate_dendrite_fold_order()
        print("DENDRITE_FOLD_ORDER_CHECK_PASSED", flush=True)
        return
    device = "cpu" if args.cpu else "cuda"
    if not args.output:
        raise ValueError("explicit output required")
    if args.output.exists():
        raise FileExistsError(f"inspect existing diagnostic output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    validate_dendrite_fold_order()
    start, stop = map(int, args.ids)
    if start < 1320 or stop > 1336 or stop <= start:
        raise ValueError("only registered validation IDs1320-1335 are permitted")
    gamma_all = torch.load(VAL_GAMMA, map_location="cpu", weights_only=True)
    gamma_rows = list(range(start - 1320, stop - 1320))
    records = {"status": "running", "condition": args.condition, "seed": args.seed,
               "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha(args.checkpoint),
               "runner_sha256": sha(__file__),
               "ids": list(range(start, stop)), "batch_size": 8,
               "steps": 1024, "settle": 512, "threshold": .5,
               "diagnostic_only": True, "ground_truth_used_only_after_forward": True,
               "distance_bins_euclidean_patches": [[0, 2], [2, 5], [5, 10], [10, "inf"]],
               "batches": []}
    with h5py.File(DATASET, "r") as dataset:
        for offset in range(start, stop, 8):
            batch_ids = list(range(offset, min(offset + 8, stop)))
            if len(batch_ids) != 8:
                raise ValueError("diagnostic suite must preserve full batch8")
            rows = [v - 1320 for v in batch_ids]
            gamma = gamma_all[rows].to(device)
            core = load_checkpoint(args.checkpoint, device)
            signals = stage_signals(core, gamma)
            if signals[1].shape != (8, 1024, 256, 4):
                raise AssertionError(f"theta shape unexpected: {tuple(signals[1].shape)}")
            for tensor in signals:
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError("nonfinite stage tensor")
            # Read GT only after completing the forward and collecting all model stages.
            labels = read_labels(dataset, batch_ids)
            batch_rows = measure_batch(signals, labels)
            records["batches"].append({"ids": batch_ids,
                                         "rows": [{"image_id": iid, "stages": row}
                                                  for iid, row in zip(batch_ids, batch_rows)]})
            tmp = args.output.with_suffix(".tmp")
            tmp.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            tmp.replace(args.output)
            del core, signals, labels, gamma
            if device == "cuda":
                torch.cuda.empty_cache()
    records["status"] = "complete"
    records["completed_unix"] = time.time()
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    print("STAGE_FLOW_DIAGNOSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

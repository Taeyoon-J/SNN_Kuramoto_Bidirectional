"""One-pass q>=.50 common-neighbor pruning on frozen SW0097 controls only."""
import hashlib
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"), str(ROOT / "collaborative_test")]
from evaluate_fixed_split import _core
from SW_0094_aligned_joint_pilot.run import DATASET, VAL_GAMMA
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks, spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity, spike_synchrony_components

CONTROL = ROOT / "trained_models/SW0097_graph_adaptation"
OUT = ROOT / "collaborative_test/SW_0103_triangle_readout/results_archive/first16.json"
IDS = list(range(1320, 1336))
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
SLOT = {"fg_ari": .7749334667, "foreground_iou": .2035891312, "matched_object_iou": .2069369792}


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def groups_from_adj(adj):
    count, labels = connected_components(csr_matrix(adj), directed=False)
    components = [np.flatnonzero(labels == i).tolist() for i in range(count)]
    components = [x for x in components if len(x) >= 2]
    components.sort(key=len, reverse=True)
    if components:
        components = components[1:]  # existing largest-component background convention
    return components


def triangle_support(q):
    a = (q.detach().cpu().numpy() >= .50)
    a = np.logical_or(a, a.T)
    np.fill_diagonal(a, False)
    common = a.astype(np.int32) @ a.astype(np.int32)
    supported = a & (common >= 1)
    raw_cc_count, raw_cc = connected_components(csr_matrix(a), directed=False)
    raw_sizes = np.bincount(raw_cc, minlength=raw_cc_count)
    restored = []
    for u, v in zip(*np.where(np.triu(a, 1))):
        if raw_sizes[raw_cc[u]] == 2:
            supported[u, v] = supported[v, u] = True
            restored.append((int(u), int(v)))
    return a, supported, restored


def bridge_edges(adj):
    n = adj.shape[0]
    disc = [-1] * n
    low = [0] * n
    bridges = set()
    tick = [0]
    def visit(u, parent):
        disc[u] = low[u] = tick[0]; tick[0] += 1
        for v in np.flatnonzero(adj[u]):
            v = int(v)
            if disc[v] < 0:
                visit(v, u); low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add((min(u, v), max(u, v)))
            elif v != parent:
                low[u] = min(low[u], disc[v])
    for node in range(n):
        if disc[node] < 0:
            visit(node, -1)
    return bridges


def labels_from_groups(groups):
    out = torch.zeros(256, dtype=torch.int64)
    for idx, group in enumerate(groups, 1):
        out[torch.as_tensor(group, dtype=torch.long)] = idx
    return out


def categories_for_edges(edges, labels):
    out = {"same_instance_fg": 0, "cross_instance_fg": 0, "foreground_background": 0, "background_background": 0}
    for u, v in edges:
        a, b = int(labels[u]), int(labels[v])
        if a > 0 and b > 0:
            key = "same_instance_fg" if a == b else "cross_instance_fg"
        elif (a > 0) != (b > 0):
            key = "foreground_background"
        else:
            key = "background_background"
        out[key] += 1
    return out


def graph_object_fragments(adj, gt):
    records = []
    for obj in sorted(int(v) for v in np.unique(gt) if v > 0):
        nodes = np.flatnonzero(gt == obj)
        if len(nodes) < 2:
            continue
        ncc, comp = connected_components(csr_matrix(adj[np.ix_(nodes, nodes)]), directed=False)
        sizes = np.bincount(comp, minlength=ncc)
        records.append({"object_id": obj, "patches": int(len(nodes)), "fragments": int(ncc),
                        "largest_fraction": float(sizes.max() / len(nodes))})
    return records


def pair_statistics(pred, gt):
    fg = gt > 0
    ii, jj = np.triu_indices(len(gt), 1)
    fg_pair = fg[ii] & fg[jj]
    same = gt[ii] == gt[jj]
    assigned = (pred[ii] > 0) & (pred[ii] == pred[jj])
    same_sel = fg_pair & same
    cross_sel = fg_pair & ~same
    return {
        "same_instance_coassignment_recall": float(assigned[same_sel].mean()) if same_sel.any() else None,
        "cross_instance_coassignment_rate": float(assigned[cross_sel].mean()) if cross_sel.any() else None,
        "same_instance_pairs": int(same_sel.sum()), "cross_instance_pairs": int(cross_sel.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    gamma_all = torch.load(VAL_GAMMA, map_location="cpu", weights_only=True)
    if tuple(gamma_all.shape) != (320, 8, 256):
        raise AssertionError(f"validation gamma shape mismatch: {tuple(gamma_all.shape)}")
    result = {"status": "running", "purpose": "fixed one-pass triangle-supported readout diagnostic; frozen SW0097 source only",
              "ground_truth_read": False, "device": str(device), "ids": IDS, "batch": 8,
              "steps": 1024, "settle": 512, "classifier_threshold": .50,
              "rule": "A=(q>=.50), diag false, sym OR; keep A edges with at least one common neighbor; restore original edge only for each raw CC of exact size 2; one pass, no iteration or GT-based grouping",
              "gamma_sha256": file_sha(VAL_GAMMA), "started_unix": time.time(), "seeds": {}}
    predictions = {}
    # Construct both endpoint predictions for all three seeds before reading any GT.
    for seed in (range(1) if args.smoke_only else range(3)):
        ckpt = CONTROL / f"seed{seed}_positive_frozen/core.pt"
        core = _core(device, ckpt, 1024, "shared", 3, 1.5, 2., .5, 16., .35, "factorized", "raw")
        core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
        labels_raw, labels_new, adj_raw_rows, adj_new_rows = [], [], [], []
        raw_groups_count, new_groups_count, restored_by_image = [], [], []
        for start in ((0,) if args.smoke_only else (0, 8)):
            gamma = gamma_all[start:start + 8].to(device)
            with torch.inference_mode():
                _, spikes, _, _theta = core(gamma, return_core_out=True, return_theta=True, num_time_steps=1024)
                components = core.last_component_spikes
                q = spike_synchrony_affinity(spikes, components, settle=512)
                legacy = spike_synchrony_components(spikes.detach().cpu(), synchrony_threshold=.50, min_group_size=2,
                    settle=512, components=components.detach().cpu(), background="largest_component")
            for b in range(8):
                raw_adj, new_adj, restored = triangle_support(q[b])
                raw = groups_from_adj(raw_adj)
                new = groups_from_adj(new_adj)
                if [tuple(sorted(x)) for x in raw] != [tuple(sorted(x)) for x in legacy[b]]:
                    raise AssertionError(f"seed{seed} image{start+b}: raw .50 CC does not reproduce actual classifier")
                labels_raw.append(labels_from_groups(raw))
                labels_new.append(labels_from_groups(new))
                adj_raw_rows.append(raw_adj)
                adj_new_rows.append(new_adj)
                raw_groups_count.append(len(raw)); new_groups_count.append(len(new)); restored_by_image.append(restored)
            del gamma, spikes, components, q
        predictions[str(seed)] = {"checkpoint_sha256": file_sha(ckpt),
            "raw_labels": torch.stack(labels_raw), "new_labels": torch.stack(labels_new),
            "raw_adjacency": np.stack(adj_raw_rows), "triangle_adjacency": np.stack(adj_new_rows),
            "raw_groups": raw_groups_count, "new_groups": new_groups_count,
            "restored_edges": restored_by_image}
        del core
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.smoke_only:
        print(json.dumps({"status": "one_batch_preflight_passed", "ground_truth_read": False,
                          "prediction_batch_shape": [8, 256],
                          "raw_groups": predictions["0"]["raw_groups"],
                          "triangle_groups": predictions["0"]["new_groups"],
                          "restored_size2_edges": [len(x) for x in predictions["0"]["restored_edges"]]}, indent=2), flush=True)
        return

    # All six paired predictions now exist. Only now read labels and score.
    with h5py.File(DATASET, "r") as dataset:
        masks = torch.from_numpy(dataset["mask"][IDS])
        gt = clevr_mask_patch(masks, 8)["patch_labels"].reshape(16, 256).cpu()
    result["ground_truth_read"] = True
    result["seeds"] = {}
    for seed in range(3):
        item = predictions[str(seed)]
        raw = item["raw_labels"]
        new = item["new_labels"]
        scores_raw = evaluate_patch_masks(raw.reshape(16, 16, 16), gt.reshape(16, 16, 16))["per_image"]
        scores_new = evaluate_patch_masks(new.reshape(16, 16, 16), gt.reshape(16, 16, 16))["per_image"]
        per_image = []
        edge_removal = {k: 0 for k in ("same_instance_fg", "cross_instance_fg", "foreground_background", "background_background")}
        removed_bridges = dict(edge_removal)
        for i, iid in enumerate(IDS):
            g = gt[i].numpy(); raw_pred = raw[i].numpy(); new_pred = new[i].numpy()
            raw_adj, new_adj = item["raw_adjacency"][i], item["triangle_adjacency"][i]
            removed = set(zip(*np.where(np.triu(raw_adj & ~new_adj, 1))))
            bridges = bridge_edges(raw_adj)
            removed_bridge_edges = removed & bridges
            removed_cat = categories_for_edges(removed, g)
            bridge_cat = categories_for_edges(removed_bridge_edges, g)
            for key in edge_removal:
                edge_removal[key] += removed_cat[key]
                removed_bridges[key] += bridge_cat[key]
            pairs = pair_statistics(raw_pred, g), pair_statistics(new_pred, g)
            fg_loss_raw = int(((g > 0) & (raw_pred == 0)).sum())
            fg_loss_new = int(((g > 0) & (new_pred == 0)).sum())
            raw_frag = graph_object_fragments(item["raw_adjacency"][i], g)
            new_frag = graph_object_fragments(item["triangle_adjacency"][i], g)
            row = {"image_id": iid, "raw_metrics": {m: float(scores_raw[m][i]) for m in METRICS},
                   "triangle_metrics": {m: float(scores_new[m][i]) for m in METRICS},
                   "raw_groups": item["raw_groups"][i], "triangle_groups": item["new_groups"][i],
                   "new_zero_groups": item["raw_groups"][i] > 0 and item["new_groups"][i] == 0,
                   "foreground_patch_loss_raw": fg_loss_raw, "foreground_patch_loss_triangle": fg_loss_new,
                   "coassignment_raw": pairs[0], "coassignment_triangle": pairs[1],
                   "gt_object_fragments_raw": raw_frag, "gt_object_fragments_triangle": new_frag,
                   "restored_raw_size2_edges": len(item["restored_edges"][i])}
            row["removed_edge_classes"] = removed_cat
            row["removed_raw_graph_bridges_by_class"] = bridge_cat
            row["removed_edge_count"] = len(removed)
            row["removed_raw_graph_bridge_count"] = len(removed_bridge_edges)
            per_image.append(row)
        means_raw = {m: float(torch.nanmean(scores_raw[m])) for m in METRICS}
        means_new = {m: float(torch.nanmean(scores_new[m])) for m in METRICS}
        pair_raw = [r["coassignment_raw"] for r in per_image]
        pair_new = [r["coassignment_triangle"] for r in per_image]
        mean_cross_raw = float(np.mean([x["cross_instance_coassignment_rate"] for x in pair_raw]))
        mean_cross_new = float(np.mean([x["cross_instance_coassignment_rate"] for x in pair_new]))
        mean_same_raw = float(np.mean([x["same_instance_coassignment_recall"] for x in pair_raw]))
        mean_same_new = float(np.mean([x["same_instance_coassignment_recall"] for x in pair_new]))
        control_eval = json.loads((CONTROL / f"seed{seed}_positive_frozen/evaluation.json").read_text())
        scored = control_eval["sweep"][0]["scored_targets"]["our_hdf5"]
        legacy_match = {}
        for m in METRICS:
            archived = scored["per_image"][m][:16]
            computed = [float(scores_raw[m][i]) for i in range(16)]
            legacy_match[m] = {"max_abs_difference": max(abs(a-b) for a, b in zip(archived, computed)),
                               "exact_within_1e_9": all(abs(a-b) <= 1e-9 for a, b in zip(archived, computed))}
        result["seeds"][str(seed)] = {"checkpoint_sha256": item["checkpoint_sha256"],
            "raw_means": means_raw, "triangle_means": means_new,
            "metric_deltas": {m: means_new[m] - means_raw[m] for m in METRICS},
            "mean_cross_instance_coassignment_raw": mean_cross_raw,
            "mean_cross_instance_coassignment_triangle": mean_cross_new,
            "mean_same_instance_recall_raw": mean_same_raw,
            "mean_same_instance_recall_triangle": mean_same_new,
            "mean_same_instance_recall_drop": mean_same_raw - mean_same_new,
            "legacy_matches_archived_first16": legacy_match,
            "mean_foreground_patch_loss_raw": float(np.mean([r["foreground_patch_loss_raw"] for r in per_image])),
            "mean_foreground_patch_loss_triangle": float(np.mean([r["foreground_patch_loss_triangle"] for r in per_image])),
            "mean_raw_groups": float(np.mean(item["raw_groups"])),
            "mean_triangle_groups": float(np.mean(item["new_groups"])),
            "images_with_new_zero_groups_where_raw_nonzero": sum(r["new_zero_groups"] for r in per_image),
            "removed_edge_classes_total": edge_removal,
            "removed_raw_graph_bridges_by_class_total": removed_bridges,
            "removed_raw_graph_bridge_count_total": sum(removed_bridges.values()),
            "per_image": per_image}
    fg_deltas = [result["seeds"][str(s)]["metric_deltas"]["fg_ari"] for s in range(3)]
    means = {m: float(np.mean([result["seeds"][str(s)]["triangle_means"][m] for s in range(3)])) for m in METRICS}
    cross_raw = float(np.mean([result["seeds"][str(s)]["mean_cross_instance_coassignment_raw"] for s in range(3)]))
    cross_new = float(np.mean([result["seeds"][str(s)]["mean_cross_instance_coassignment_triangle"] for s in range(3)]))
    cross_decreases = cross_new < cross_raw
    same_drop = float(np.mean([result["seeds"][str(s)]["mean_same_instance_recall_drop"] for s in range(3)]))
    zero_ok = all(result["seeds"][str(s)]["images_with_new_zero_groups_where_raw_nonzero"] == 0 for s in range(3))
    prelim = (float(np.mean(fg_deltas)) > 0 and sum(x > 0 for x in fg_deltas) >= 2 and
              means["foreground_iou"] >= SLOT["foreground_iou"] + .05 and
              means["matched_object_iou"] >= SLOT["matched_object_iou"] + .05 and
              cross_decreases and same_drop <= .02 and zero_ok)
    result["first16_gate"] = {"mean_triangle_metrics": means,
        "mean_fg_ari_delta": float(np.mean(fg_deltas)), "seed_fg_ari_positive": sum(x > 0 for x in fg_deltas),
        "triangle_foreground_iou_over_slot": means["foreground_iou"] - SLOT["foreground_iou"],
        "triangle_object_iou_over_slot": means["matched_object_iou"] - SLOT["matched_object_iou"],
        "mean_cross_instance_coassignment_raw": cross_raw,
        "mean_cross_instance_coassignment_triangle": cross_new,
        "mean_cross_instance_coassignment_decreased": cross_decreases,
        "mean_same_instance_recall_drop": same_drop, "no_new_zero_groups": zero_ok,
        "pass": prelim, "full320_authorized_by_preregistered_gate": prelim}
    result["finished_unix"] = time.time()
    result["status"] = "complete"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(f"preserve existing {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "first16_gate": result["first16_gate"], "output": str(OUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

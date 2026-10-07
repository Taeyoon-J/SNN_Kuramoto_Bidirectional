"""Frozen-mask QA only; no training, threshold tuning, or loss implementation."""
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"), str(ROOT / "collaborative_test")]
from SW_0094_aligned_joint_pilot.run import DATASET, VAL_GAMMA, hparams
from SW_0099_stage_flow_diagnosis.diagnose import read_labels
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore

OUT = ROOT / "trained_models/SW0101_teacher_mask_qa"
SOURCE = ROOT / "trained_models/SW0095_full70k_aligned_loss"
IDS = list(range(1320, 1336))
BINS = (("near_0_2", 0.0, 2.0), ("mid_2_5", 2.0, 5.0))
TOP_K = 4
COS_POSITIVE = 0.8
COS_NEGATIVE = 0.0
MIN_PRECISION = 0.95
MIN_COVERAGE_IMAGES = 12
MIN_FG_RELEVANT_PAIRS = 50


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def graph_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    selected = {name: value for name, value in state.items()
                if name.startswith("graph_generator.")}
    if not selected:
        raise AssertionError(f"checkpoint has no graph_generator state: {path}")
    digest = hashlib.sha256()
    for name in sorted(selected):
        digest.update(name.encode())
        digest.update(tensor_hash(selected[name]).encode())
    return selected, digest.hexdigest()


def patch_distances(device="cpu"):
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    xy = torch.stack((yy.flatten(), xx.flatten()), dim=1).float().to(device)
    return torch.cdist(xy, xy)


def frozen_graph_and_projected_cosine(core, gamma):
    projected = []
    handle = core.graph_generator.projection.register_forward_hook(
        lambda _module, _inputs, output: projected.append(output.detach()))
    try:
        with torch.inference_mode():
            adjacency = core.graph_generator(gamma)
    finally:
        handle.remove()
    if len(projected) != 1:
        raise AssertionError(f"expected one actual projection call, found {len(projected)}")
    z = F.normalize(projected[0], dim=-1)
    cosine = torch.bmm(z, z.transpose(1, 2)).clamp(-1.0, 1.0)
    if adjacency.shape != (gamma.shape[0], 256, 256) or cosine.shape != adjacency.shape:
        raise AssertionError("graph/projected-cosine shape mismatch")
    if not torch.isfinite(adjacency).all() or not torch.isfinite(cosine).all():
        raise FloatingPointError("nonfinite graph or projected cosine")
    return adjacency.detach().cpu(), cosine.detach().cpu()


def mask_digest(mask):
    return hashlib.sha256(mask.to(torch.uint8).contiguous().numpy().tobytes()).hexdigest()


def build_frozen_masks(adjacency, cosine, ids):
    # Matrices use directed anchor rows. Eligibility and labels are GT-free.
    bsz, n, n2 = adjacency.shape
    if n != 256 or n2 != 256 or len(ids) != bsz:
        raise AssertionError("unexpected batch or node count")
    eye = torch.eye(n, dtype=torch.bool)
    no_diag = ~eye
    a = adjacency.detach().cpu()
    c = cosine.detach().cpu()
    masked_a = a.masked_fill(~no_diag.unsqueeze(0), float("-inf"))
    top_idx = masked_a.topk(TOP_K, dim=-1).indices
    top = torch.zeros((bsz, n, n), dtype=torch.bool)
    top.scatter_(-1, top_idx, True)
    mutual_top = top & top.transpose(1, 2)
    dist = patch_distances()
    per_bin = {name: [] for name, _, _ in BINS}
    masks = {name: {"positive": [], "negative": []} for name, _, _ in BINS}
    for name, low, high in BINS:
        spatial = (dist > low) & (dist <= high) & no_diag
        positive_base = mutual_top & (c >= COS_POSITIVE) & spatial.unsqueeze(0)
        negative_base = (a == 0) & (c <= COS_NEGATIVE) & spatial.unsqueeze(0)
        for b, image_id in enumerate(ids):
            pos = positive_base[b]
            neg = negative_base[b]
            eligible = (pos.sum(dim=-1) > 0) & (neg.sum(dim=-1) > 0)
            pos = pos & eligible[:, None]
            neg = neg & eligible[:, None]
            masks[name]["positive"].append(pos)
            masks[name]["negative"].append(neg)
            per_bin[name].append({
                "image_id": image_id,
                "eligible_anchors": int(eligible.sum()),
                "positive_directed_decisions": int(pos.sum()),
                "negative_directed_decisions": int(neg.sum()),
                "positive_mask_sha256": mask_digest(pos),
                "negative_mask_sha256": mask_digest(neg),
            })
    return masks, per_bin


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def pair_audit(positive, negative, labels):
    labels = labels.long().cpu()
    fg = labels > 0
    fg_pair = fg[:, None] & fg[None, :]
    fg_relevant = fg[:, None] | fg[None, :]
    same_fg = fg_pair & (labels[:, None] == labels[None, :])
    diff_fg = fg_pair & (labels[:, None] != labels[None, :])
    fg_bg = fg[:, None] ^ fg[None, :]
    bg_bg = (~fg)[:, None] & (~fg)[None, :]
    pos_rel = positive & fg_relevant
    neg_rel = negative & fg_relevant
    pos_correct = positive & same_fg
    neg_correct = negative & (diff_fg | fg_bg)
    return {
        "positive": {
            "fg_relevant_decisions": int(pos_rel.sum()),
            "correct_same_foreground_instance": int(pos_correct.sum()),
            "precision": ratio(int(pos_correct.sum()), int(pos_rel.sum())),
            "fg_fg_same_instance": int((positive & same_fg).sum()),
            "fg_fg_bridge_different_instances": int((positive & diff_fg).sum()),
            "fg_bg_bridge": int((positive & fg_bg).sum()),
            "bg_bg_excluded": int((positive & bg_bg).sum()),
        },
        "negative": {
            "fg_relevant_decisions": int(neg_rel.sum()),
            "correct_different_or_fg_bg": int(neg_correct.sum()),
            "precision": ratio(int(neg_correct.sum()), int(neg_rel.sum())),
            "fg_fg_different_instances": int((negative & diff_fg).sum()),
            "fg_bg_separated": int((negative & fg_bg).sum()),
            "fg_fg_same_instance_contamination": int((negative & same_fg).sum()),
            "bg_bg_excluded": int((negative & bg_bg).sum()),
        },
    }


def component_contamination(positive, labels):
    graph = positive | positive.T
    labels = labels.long().cpu()
    seen = set()
    components = []
    for node in torch.where(graph.any(dim=1))[0].tolist():
        if node in seen:
            continue
        stack, group = [node], []
        seen.add(node)
        while stack:
            current = stack.pop()
            group.append(current)
            for nxt in torch.where(graph[current])[0].tolist():
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        values = labels[group]
        objects = sorted(int(v) for v in torch.unique(values[values > 0]).tolist())
        has_fg = bool((values > 0).any())
        has_bg = bool((values == 0).any())
        components.append({"patches": len(group), "foreground_objects": objects,
                           "merges_multiple_foreground_instances": len(objects) > 1,
                           "connects_foreground_to_background": has_fg and has_bg})
    return {
        "positive_components_with_edges": len(components),
        "components_merging_foreground_instances": sum(
            item["merges_multiple_foreground_instances"] for item in components),
        "components_connecting_foreground_background": sum(
            item["connects_foreground_to_background"] for item in components),
        "components": components,
    }


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=False)
    mask_path = OUT / "masks_frozen_pre_gt.json"
    result_path = OUT / "qa_results.json"
    if mask_path.exists() or result_path.exists():
        raise FileExistsError("inspect existing SW0101 artifacts; no rerun/overwrite")

    source_paths = [SOURCE / f"seed{seed}/core.pt" for seed in range(3)]
    source_shas = {str(seed): sha256_file(path) for seed, path in enumerate(source_paths)}
    graph_states = {}
    graph_hashes = {}
    for seed, path in enumerate(source_paths):
        selected, graph_hashes[str(seed)] = graph_state(path)
        graph_states[str(seed)] = selected
    base_names = set(graph_states["0"])
    if any(set(graph_states[str(seed)]) != base_names for seed in range(1, 3)):
        raise AssertionError("SW0095 graph-generator state keys differ across seeds")
    graph_exact_equal = all(
        torch.equal(graph_states["0"][name], graph_states[str(seed)][name])
        for seed in (1, 2) for name in sorted(base_names))
    if not graph_exact_equal:
        raise AssertionError("SW0095 graph-generator tensors are not bitwise equal")

    hp = hparams("raw")
    core = S2NetCore(hp, device=device).to(device)
    source_state = torch.load(source_paths[0], map_location=device, weights_only=True)
    core.load_state_dict(source_state, strict=True)
    core.eval()
    del source_state, graph_states
    gamma_all = torch.load(VAL_GAMMA, map_location="cpu", weights_only=True).float()
    gamma = gamma_all[:16]
    if gamma.shape != (16, 8, 256):
        raise AssertionError(f"unexpected validation gamma shape {tuple(gamma.shape)}")

    frozen = {
        "status": "masks_frozen_pre_gt",
        "experiment": "SW0101_teacher_mask_qa",
        "ground_truth_read": False,
        "ids": IDS,
        "batch_size": 8,
        "source_checkpoint": str(source_paths[0]),
        "source_checkpoint_sha256": source_shas["0"],
        "all_sw0095_source_checkpoint_sha256": source_shas,
        "graph_state_sha256_by_seed": graph_hashes,
        "graph_state_bitwise_equal_all_seeds": graph_exact_equal,
        "graph_module_source_sha256": sha256_file(ROOT / "snn_kuramoto_bidirectional/graph_generator.py"),
        "gamma_cache_sha256": sha256_file(VAL_GAMMA),
        "qa_code_sha256": sha256_file(Path(__file__)),
        "feature_definition": "F.normalize(core.graph_generator.projection(gamma.transpose(1,2)), dim=-1); C = z @ z.T from a hook on the actual graph-generator projection call",
        "adjacency_definition": "A is the unchanged core.graph_generator(gamma) output",
        "mask_spec": {"top_k": TOP_K, "positive_cosine_min": COS_POSITIVE,
                      "negative_cosine_max": COS_NEGATIVE, "negative_adjacency_exactly_zero": True,
                      "bins_euclidean_patch_units": [[0, 2], [2, 5]],
                      "bin_bounds": "(low, high]", "diagonal_excluded": True,
                      "eligible_anchor": "anchor has at least one positive and at least one negative candidate in the same distance bin",
                      "decisions": "directed anchor-neighbor decisions from eligible anchors"},
        "pre_gt_per_image_bin": {},
        "positive_mask_sha256_all": {},
        "negative_mask_sha256_all": {},
    }
    pending_masks = {name: {"positive": [], "negative": []} for name, _, _ in BINS}
    for offset in (0, 8):
        batch_ids = IDS[offset:offset + 8]
        batch_gamma = gamma[offset:offset + 8].to(device)
        adjacency, cosine = frozen_graph_and_projected_cosine(core, batch_gamma)
        masks, pre_counts = build_frozen_masks(adjacency, cosine, batch_ids)
        for name, _, _ in BINS:
            frozen["pre_gt_per_image_bin"].setdefault(name, []).extend(pre_counts[name])
            pending_masks[name]["positive"].extend(masks[name]["positive"])
            pending_masks[name]["negative"].extend(masks[name]["negative"])
    for name, _, _ in BINS:
        frozen["positive_mask_sha256_all"][name] = hashlib.sha256(
            "".join(item["positive_mask_sha256"] for item in frozen["pre_gt_per_image_bin"][name]).encode()).hexdigest()
        frozen["negative_mask_sha256_all"][name] = hashlib.sha256(
            "".join(item["negative_mask_sha256"] for item in frozen["pre_gt_per_image_bin"][name]).encode()).hexdigest()
    write_json(mask_path, frozen)
    if not mask_path.is_file():
        raise RuntimeError("pre-GT mask freeze artifact was not written")

    # Target masks are intentionally opened only after the frozen mask artifact exists.
    with h5py.File(DATASET, "r") as dataset:
        labels = read_labels(dataset, IDS)
    if labels.shape != (16, 256):
        raise AssertionError(f"unexpected target-label shape {tuple(labels.shape)}")

    per_image = {name: [] for name, _, _ in BINS}
    for bin_name, _, _ in BINS:
        for index, image_id in enumerate(IDS):
            pos = pending_masks[bin_name]["positive"][index]
            neg = pending_masks[bin_name]["negative"][index]
            label = labels[index]
            scores = pair_audit(pos, neg, label)
            eligible = frozen["pre_gt_per_image_bin"][bin_name][index]["eligible_anchors"]
            per_image[bin_name].append({"image_id": image_id,
                                        "eligible_anchors": eligible,
                                        **scores})
    components = []
    for index, image_id in enumerate(IDS):
        positive_union = (pending_masks[BINS[0][0]]["positive"][index]
                          | pending_masks[BINS[1][0]]["positive"][index])
        components.append({"image_id": image_id,
                           **component_contamination(positive_union, labels[index])})

    by_bin = {}
    for bin_name, _, _ in BINS:
        rows = per_image[bin_name]
        eligible_images = sum(row["eligible_anchors"] > 0 for row in rows)
        pos_n = sum(row["positive"]["fg_relevant_decisions"] for row in rows)
        pos_correct = sum(row["positive"]["correct_same_foreground_instance"] for row in rows)
        neg_n = sum(row["negative"]["fg_relevant_decisions"] for row in rows)
        neg_correct = sum(row["negative"]["correct_different_or_fg_bg"] for row in rows)
        pos_precision, neg_precision = ratio(pos_correct, pos_n), ratio(neg_correct, neg_n)
        passed = (pos_precision is not None and neg_precision is not None
                  and pos_precision >= MIN_PRECISION and neg_precision >= MIN_PRECISION
                  and eligible_images >= MIN_COVERAGE_IMAGES
                  and pos_n >= MIN_FG_RELEVANT_PAIRS
                  and neg_n >= MIN_FG_RELEVANT_PAIRS)
        by_bin[bin_name] = {
            "eligible_images": eligible_images, "total_images": len(IDS),
            "coverage_fraction": eligible_images / len(IDS),
            "eligible_anchor_total": sum(row["eligible_anchors"] for row in rows),
            "positive_fg_relevant_decisions": pos_n,
            "positive_correct_same_foreground_instance": pos_correct,
            "positive_precision": pos_precision,
            "positive_fg_fg_bridge_different_instances": sum(row["positive"]["fg_fg_bridge_different_instances"] for row in rows),
            "positive_fg_bg_bridge": sum(row["positive"]["fg_bg_bridge"] for row in rows),
            "positive_bg_bg_excluded": sum(row["positive"]["bg_bg_excluded"] for row in rows),
            "negative_fg_relevant_decisions": neg_n,
            "negative_correct_different_or_fg_bg": neg_correct,
            "negative_precision": neg_precision,
            "negative_fg_fg_different_instances": sum(row["negative"]["fg_fg_different_instances"] for row in rows),
            "negative_fg_bg_separated": sum(row["negative"]["fg_bg_separated"] for row in rows),
            "negative_fg_fg_same_instance_contamination": sum(row["negative"]["fg_fg_same_instance_contamination"] for row in rows),
            "negative_bg_bg_excluded": sum(row["negative"]["bg_bg_excluded"] for row in rows),
            "pass": passed,
            "per_image": rows,
        }
    component_summary = {
        "positive_components_with_edges": sum(row["positive_components_with_edges"] for row in components),
        "components_merging_foreground_instances": sum(row["components_merging_foreground_instances"] for row in components),
        "components_connecting_foreground_background": sum(row["components_connecting_foreground_background"] for row in components),
        "per_image": components,
    }
    passed_all = all(item["pass"] for item in by_bin.values())
    result = {
        "status": "complete",
        "experiment": "SW0101_teacher_mask_qa",
        "mask_spec_frozen_before_ground_truth": True,
        "ground_truth_used_only_for_post_freeze_scoring": True,
        "ids": IDS,
        "mask_freeze_artifact": str(mask_path),
        "mask_freeze_sha256": sha256_file(mask_path),
        "source_checkpoint_sha256": source_shas["0"],
        "graph_state_bitwise_equal_all_sw0095_seeds": graph_exact_equal,
        "qa_code_sha256": sha256_file(Path(__file__)),
        "acceptance_gate": {"precision_min_each_class_each_bin": MIN_PRECISION,
                            "coverage_min_images_each_bin": MIN_COVERAGE_IMAGES,
                            "fg_relevant_pair_min_each_class_each_bin": MIN_FG_RELEVANT_PAIRS},
        "bins": by_bin,
        "positive_component_contamination": component_summary,
        "pass_teacher_recipe": passed_all,
        "decision": "proceed_to_new_design_review_only" if passed_all else
                    "stop_teacher_recipe_no_threshold_tuning_no_training",
        "completed_unix": time.time(),
    }
    write_json(result_path, result)
    print(json.dumps({"status": result["status"], "pass_teacher_recipe": passed_all,
                      "decision": result["decision"], "results": by_bin}, indent=2), flush=True)


if __name__ == "__main__":
    main()

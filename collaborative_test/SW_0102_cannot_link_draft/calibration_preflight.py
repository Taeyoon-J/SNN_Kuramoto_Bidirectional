"""No-update, no-GT gradient calibration on the first four matched train batches."""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
               str(ROOT / "collaborative_test")]
from SW_0094_aligned_joint_pilot.run import (GAMMA, hparams,
                                             aligned_affinity)
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv

from SW_0102_cannot_link_draft.forest_loss import (cannot_link_hinge,
                                                    same_component_pairs)

SOURCE = ROOT / "trained_models/SW0095_full70k_aligned_loss"
CONTROL = ROOT / "trained_models/SW0097_graph_adaptation"
OUT = ROOT / "trained_models/SW0102_cannot_link_draft/calibration"
BINS = (("near_0_2", 0.0, 2.0), ("mid_2_5", 2.0, 5.0))


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_equal(left, right):
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def expected_training_ids(seed):
    permutation = torch.randperm(70000,
        generator=torch.Generator().manual_seed(117 + seed))[:4096]
    return [int(index) if index < 1000 else int(index) + 640
            for index in permutation.tolist()]


def make_criterion():
    return UnsupervisedS2NetLoss(
        spike_rate_weight=0., spike_smooth_weight=0., spike_diversity_weight=0.,
        structural_weight=0., plv_bimodality_weight=6., plv_balance_weight=10.,
        plv_coherence_weight=.5, plv_collapse_weight=1., plv_target_density=.867,
        patch_grid_size=(16, 16))


def patch_distance():
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    coord = torch.stack((yy.flatten(), xx.flatten()), dim=1).float()
    return torch.cdist(coord, coord)


def actual_101_negative_masks(adjacency, projected):
    """GT-free exact SW0101 masks: near/mid, mutual top4 and eligibility."""
    if adjacency.ndim != 3 or adjacency.shape[1:] != (256, 256):
        raise ValueError("unexpected graph adjacency shape")
    z = F.normalize(projected, dim=-1)
    cosine = torch.bmm(z, z.transpose(1, 2)).clamp(-1.0, 1.0)
    batch, nodes, _ = adjacency.shape
    no_diag = ~torch.eye(nodes, dtype=torch.bool, device=adjacency.device)
    masked = adjacency.masked_fill(~no_diag.unsqueeze(0), float("-inf"))
    indices = masked.topk(4, dim=-1).indices
    top = torch.zeros_like(adjacency, dtype=torch.bool)
    top.scatter_(-1, indices, True)
    mutual_top = top & top.transpose(1, 2)
    distance = patch_distance().to(adjacency.device)
    masks = []
    counts = []
    for bin_name, low, high in BINS:
        spatial = ((distance > low) & (distance <= high) & no_diag)
        positive = mutual_top & (cosine >= .8) & spatial.unsqueeze(0)
        negative = (adjacency == 0) & (cosine <= 0) & spatial.unsqueeze(0)
        eligible = (positive.sum(-1) > 0) & (negative.sum(-1) > 0)
        negative = negative & eligible.unsqueeze(-1)
        masks.append(negative)
        counts.append({"bin": bin_name,
                       "eligible_anchors": eligible.sum(-1).detach().cpu().tolist(),
                       "negative_decisions": negative.sum((1, 2)).detach().cpu().tolist(),
                       "positive_candidates": positive.sum((1, 2)).detach().cpu().tolist(),
                       "negative_mask_sha256": hashlib.sha256(
                           negative.to(torch.uint8).detach().cpu().contiguous().numpy().tobytes()
                       ).hexdigest()})
    return masks, counts


def grad_norm(gradients):
    total = 0.0
    saw = False
    for gradient in gradients:
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite loss-specific gradient")
        saw = True
        total += float(gradient.detach().double().square().sum())
    return math.sqrt(total) if saw else 0.0


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(117)
    device = torch.device(args.device)
    gamma_path = ROOT / GAMMA.relative_to(ROOT)
    gamma_sha = file_sha(gamma_path)
    gamma_cache = torch.load(gamma_path, map_location="cpu", weights_only=True,
                             mmap=True)
    if tuple(gamma_cache.shape) != (70000, 8, 256):
        raise AssertionError(f"unexpected train gamma cache shape {tuple(gamma_cache.shape)}")

    result = {"status": "running", "experiment": "SW0102_cannot_link_draft",
              "purpose": "gradient calibration and readiness only; no optimizer or GT",
              "device": str(device), "gamma_cache": str(gamma_path),
              "gamma_cache_sha256": gamma_sha, "batch_size": 16,
              "train_steps": 64, "settle": 32, "first_batches_per_seed": 4,
              "seeds": {}, "started_unix": time.time()}
    all_ratios = []
    for seed in range(3):
        source_dir = SOURCE / f"seed{seed}"
        control_dir = CONTROL / f"seed{seed}_positive_frozen"
        source_path = source_dir / "core.pt"
        source_manifest = json.loads((source_dir / "manifest.json").read_text())
        control_manifest = json.loads((control_dir / "manifest.json").read_text())
        if not (source_dir / "COMPLETED").is_file() or not (control_dir / "COMPLETED").is_file():
            raise FileNotFoundError(f"completed SW0095/97 source/control missing for seed{seed}")
        source_sha = file_sha(source_path)
        expected_ids = expected_training_ids(seed)
        required = {"status": "complete", "source_model_seed": seed,
                    "seed": 117 + seed, "steps": 256, "batch": 16,
                    "train_steps": 64, "train_settle": 32,
                    "core_lr": 3e-5, "ground_truth_used_for_training": False}
        for key, value in required.items():
            if control_manifest.get(key) != value:
                raise AssertionError(f"SW0097 seed{seed} {key} mismatch")
        if source_manifest.get("status") != "complete" or source_manifest.get("source_model_seed") != seed:
            raise AssertionError(f"unexpected SW0095 seed{seed} source manifest")
        if control_manifest.get("source_sha256") != source_sha:
            raise AssertionError(f"SW0095/97 source checkpoint SHA mismatch at seed{seed}")
        if control_manifest.get("training_ids") != expected_ids:
            raise AssertionError(f"SW0097 seed{seed} training order is not registered permutation")

        core = S2NetCore(hparams("raw"), device=device).to(device)
        state = torch.load(source_path, map_location=device, weights_only=True)
        core.load_state_dict(state, strict=True)
        source_state = {key: value.detach().clone() for key, value in core.state_dict().items()}
        del state
        core.graph_generator.requires_grad_(False)
        core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
        core.train()
        params = [parameter for parameter in core.parameters() if parameter.requires_grad]
        if not params:
            raise AssertionError("no trainable core parameters")
        criterion = make_criterion()
        generator = torch.Generator().manual_seed(117 + seed)
        train_indices = torch.randperm(70000, generator=generator)[:4096]
        gradients = []
        seed_ratios = []
        same_cc_total = 0
        direct_total = 0
        indirect_total = 0
        batches = []

        for batch_index in range(4):
            batch_indices = train_indices[batch_index * 16:(batch_index + 1) * 16]
            gamma = gamma_cache[batch_indices].to(device)
            projected, adjacency = [], []
            graph_hook = core.graph_generator.register_forward_hook(
                lambda _module, _inputs, output: adjacency.append(output))
            projection_hook = core.graph_generator.projection.register_forward_hook(
                lambda _module, _inputs, output: projected.append(output))
            try:
                _, spikes, core_out, plv, theta = _forward_with_plv(
                    core, gamma, criterion, 32, "phase", "mean")
            finally:
                graph_hook.remove()
                projection_hook.remove()
            if len(adjacency) != 1 or len(projected) != 1:
                raise AssertionError("expected one graph/projection call in static-graph core")
            if core.last_component_spikes.shape[-1] != 64:
                raise AssertionError("unexpected component spike time dimension")
            masks, mask_counts = actual_101_negative_masks(adjacency[0], projected[0])
            q = aligned_affinity(core, settle=32)
            if not torch.isfinite(q).all():
                raise FloatingPointError("nonfinite classifier q")
            cut_loss, loss_counts = cannot_link_hinge(q, masks)
            if not torch.isfinite(cut_loss):
                raise FloatingPointError("nonfinite cannot-link loss")

            s = torch.maximum(q.detach(), q.detach().transpose(1, 2))
            samecc_per_bin, direct_per_bin, indirect_per_bin = [], [], []
            for negative in masks:
                same = same_component_pairs(q.detach(), negative, threshold=.50)
                off_diagonal = ~torch.eye(256, dtype=torch.bool, device=device)
                directed_neg = negative & off_diagonal.unsqueeze(0)
                direct = directed_neg & (s >= .50)
                indirect = directed_neg & same & (s <= .40)
                samecc_per_bin.append(int((directed_neg & same).sum()))
                direct_per_bin.append(int(direct.sum()))
                indirect_per_bin.append(int(indirect.sum()))
            same_cc = sum(samecc_per_bin)
            direct_count = sum(direct_per_bin)
            indirect_count = sum(indirect_per_bin)
            same_cc_total += same_cc
            direct_total += direct_count
            indirect_total += indirect_count

            old_spike = aligned_affinity(core, settle=32)
            weighted_old_spike_loss, _ = criterion(plv=old_spike)
            weighted_old_spike_loss = 5.0 * weighted_old_spike_loss
            if not torch.isfinite(weighted_old_spike_loss):
                raise FloatingPointError("nonfinite old weighted positive-spike loss")
            old_gradients = torch.autograd.grad(weighted_old_spike_loss, params,
                                                retain_graph=True, allow_unused=True)
            cut_gradients = torch.autograd.grad(cut_loss, params,
                                                retain_graph=False, allow_unused=True)
            old_norm = grad_norm(old_gradients)
            cut_norm = grad_norm(cut_gradients)
            valid_ratio = old_norm > 0 and cut_norm > 0 and same_cc > 0
            ratio = .25 * old_norm / cut_norm if valid_ratio else None
            if ratio is not None:
                seed_ratios.append(ratio)
                all_ratios.append(ratio)
            batches.append({"batch_index": batch_index,
                "gamma_cache_indices": batch_indices.tolist(),
                "training_dataset_ids": expected_ids[batch_index * 16:(batch_index + 1) * 16],
                "mask_counts_per_image_bin": mask_counts,
                "cut_loss": float(cut_loss.detach()),
                "loss_counts": loss_counts,
                "same_cc_negative_decisions_by_bin_at_q_ge_0_50": samecc_per_bin,
                "direct_q_ge_0_50_by_bin": direct_per_bin,
                "indirect_same_cc_q_le_0_40_by_bin": indirect_per_bin,
                "old_5x_positive_spike_grad_norm": old_norm,
                "cut_loss_grad_norm": cut_norm,
                "lambda_ratio_if_valid": ratio,
                "actual_mask_constructed_without_gt": True})
            del old_gradients, cut_gradients, q, gamma, theta, spikes, core_out, plv
            del weighted_old_spike_loss, cut_loss, old_spike
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()

        unchanged = tensor_state_equal(source_state, core.state_dict())
        if not unchanged:
            raise AssertionError("calibration pass mutated source checkpoint state; no update was intended")
        seed_result = {"source_core_sha256": source_sha,
                       "matched_control_source_sha256": control_manifest["source_sha256"],
                       "training_id_order_matches_control": True,
                       "registered_shuffle_seed": 117 + seed,
                       "source_model_seed": seed,
                       "source_state_unchanged_after_no_update_calibration": unchanged,
                       "same_cc_negative_decisions_at_q_ge_0_50": same_cc_total,
                       "direct_q_ge_0_50_decisions": direct_total,
                       "indirect_same_cc_q_le_0_40_decisions": indirect_total,
                       "four_batches": batches,
                       "seed_has_real_hard_negative": same_cc_total > 0,
                       "seed_has_valid_gradient_calibration": len(seed_ratios) > 0,
                       "seed_lambda_median": statistics.median(seed_ratios) if seed_ratios else None}
        result["seeds"][str(seed)] = seed_result
        if not seed_result["seed_has_real_hard_negative"]:
            result["status"] = "failed_same_cc_guard"
            result["decision"] = f"stop_before_training_seed{seed}_no_q_ge_0_50_negative_pair"
            break
        if not seed_result["seed_has_valid_gradient_calibration"]:
            result["status"] = "failed_gradient_guard"
            result["decision"] = f"stop_before_training_seed{seed}_no_valid_nonzero_gradient_ratio"
            break

    else:
        result["status"] = "calibration_complete"
        result["shared_lambda"] = statistics.median(all_ratios)
        result["decision"] = "ready_for_fullbatch_update_preflight_only"
    result["lambda_ratios_all_valid_batches"] = all_ratios
    result["finished_unix"] = time.time()
    result_path = args.output / "calibration.json"
    if result_path.exists():
        raise FileExistsError(f"preserve existing calibration result: {result_path}")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "shared_lambda": result.get("shared_lambda"),
                      "per_seed": {seed: {"samecc": row.get("same_cc_negative_decisions_at_q_ge_0_50"),
                                          "direct": row.get("direct_q_ge_0_50_decisions"),
                                          "indirect": row.get("indirect_same_cc_q_le_0_40_decisions"),
                                          "seed_lambda_median": row.get("seed_lambda_median"),
                                          "valid_batches": sum(item["lambda_ratio_if_valid"] is not None
                                                               for item in row.get("four_batches", []))}
                                   for seed, row in result["seeds"].items()}}, indent=2),
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

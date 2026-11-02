"""Paired SW0117 joint analytic-RGB replication for source seeds 1 and 2.

This leaves SW0117 immutable. Its completed seed-0 source/candidate outcomes and
calibrated RGB coefficient are historical references; this runner only creates
new paired control/candidate outputs for seeds 1 and 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
ARCHIVE = HERE / "results_archive"
OUT = ROOT / "trained_models/SW0122_joint_analytic_rgb_replication"
SEEDS = (1, 2)
ARMS = ("control", "analytic_candidate")
BATCH, UPDATES = 16, 256
TRAIN_STEPS, TRAIN_SETTLE = 64, 32
LAMBDA = 7.865416617457706
CORE_GRAPH_LR, ENCODER_LR, CLIP_NORM = 3e-5, 3e-6, 1.0
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb import run as rgb_base
from collaborative_test.SW_0115_analytic_partition_rgb.loss import batch_reconstruction_loss
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0117_joint_analytic_rgb import coordinator as sw117_coordinator

LAMBDA_PATH = sw117.LAMBDA_PATH
HISTORICAL_SUMMARY = HERE / "historical_seed0_reference.json"
EXPECTED_HISTORICAL_SUMMARY_SHA256 = (
    "a672b197c89608e6344bb77b8c8c5940a44f34f1bdc3ddfa38e4bb86da410479")
EXPECTED_SOURCE_TRAIN_MANIFEST_SHAS = {
    1: "241c9ea96fc3d3d2b168cb49d941e39c6751eca4436192e614760673545da38b",
    2: "37d88664e7f8d91ed14f1089bcdaaf3b10b952ab37b7a756aace54d7e1a80cef",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def require_vacant(path, what):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserving existing {what}: {path}")
    return path


def implementation_fingerprint():
    paths = [RUNNER, HERE / "coordinator.py", HERE / "protocol.json", HISTORICAL_SUMMARY,
             sw117.RUNNER, sw117.HERE / "protocol.json",
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py",
             ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py",
             ROOT / "snn_kuramoto_bidirectional/loss_function.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_s2net_core.py",
             ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}


def validate_historical_seed0():
    """Require, but never rewrite, the completed seed-0 recipe/reference."""
    if sha(HISTORICAL_SUMMARY) != EXPECTED_HISTORICAL_SUMMARY_SHA256:
        raise AssertionError("copied SW0117 historical summary changed from the reviewed artifact")
    summary = json.loads(HISTORICAL_SUMMARY.read_text(encoding="utf-8"))
    tasks = sw117_coordinator.task_plan()
    for arm in ARMS:
        for stage in ("preflight", "train", "evaluate"):
            task = next(t for t in tasks if t["arm"] == arm and t["stage"] == stage)
            if not sw117_coordinator.valid_result(task):
                raise AssertionError(f"historical SW0117 seed-0 {arm}/{stage} is not valid")
    lam_record = json.loads(LAMBDA_PATH.read_text(encoding="utf-8"))
    original_candidate_pf = sw117.ARCHIVE / "preflight_seed0_analytic_candidate.json"
    if not original_candidate_pf.is_file():
        raise FileNotFoundError(original_candidate_pf)
    _validate_seed0_lambda_records(summary, lam_record, sha(original_candidate_pf),
                                   sw117.implementation_fingerprint())
    return {"summary_sha256": sha(HISTORICAL_SUMMARY),
            "lambda_sha256": sha(LAMBDA_PATH),
            "seed0_candidate_manifest_sha256": sha(sw117.OUT / "seed0_analytic_candidate/manifest.json"),
            "seed0_candidate_evaluation_sha256": sha(sw117.OUT / "seed0_analytic_candidate/evaluation.json"),
            "seed0_control_manifest_sha256": sha(sw117.OUT / "seed0_control/manifest.json"),
            "seed0_control_evaluation_sha256": sha(sw117.OUT / "seed0_control/evaluation.json")}


def _validate_seed0_lambda_records(summary, lam_record, preflight_sha256, fingerprint):
    if (summary.get("status") != "completed_seed0_not_promoted"
            or summary.get("lambda_joint") != LAMBDA
            or lam_record.get("status") != "passed" or lam_record.get("seed") != 0
            or lam_record.get("lambda_joint") != LAMBDA
            or lam_record.get("preflight_sha256") != preflight_sha256
            or lam_record.get("implementation_fingerprint") != fingerprint):
        raise AssertionError("immutable SW0117 seed-0 coefficient/reference is not validated")
    return LAMBDA


def validate_source_reference(seed):
    if seed not in SEEDS:
        raise ValueError(f"SW0122 only accepts source seed 1 or 2, got {seed}")
    source, source_manifest, source_record, ids, rows = source_contract(seed)
    folder = source.parent
    eval_path = folder / "evaluation.json"
    manifest_path = source_manifest
    marker = folder / "COMPLETED"
    if not all(p.is_file() for p in (eval_path, manifest_path, marker)):
        raise FileNotFoundError(f"completed SW0097 seed-{seed} frozen source evaluation is required")
    if sha(manifest_path) != EXPECTED_SOURCE_TRAIN_MANIFEST_SHAS[seed]:
        raise AssertionError(f"SW0097 seed-{seed} source-training manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    metrics = _validate_source_evaluation_records(seed, source, ids, manifest, report)
    return {"source_core": source, "source_manifest": source_manifest,
            "source_manifest_sha256": sha(source_manifest), "source_core_sha256": sha(source),
            "source_evaluation": eval_path, "source_evaluation_sha256": sha(eval_path),
            "source_evaluation_manifest_sha256": sha(manifest_path),
            "source_metrics": metrics, "training_ids": ids, "rows": rows,
            "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()}


def _validate_source_evaluation_records(seed, source, ids, manifest, report):
    expected = {"fg_ari", "foreground_iou", "matched_object_iou"}
    if (manifest.get("status") != "complete" or manifest.get("arm") != "positive_frozen"
            or manifest.get("source_model_seed") != seed
            or manifest.get("unique_images_seen") != 4096 or manifest.get("steps") != 256
            or manifest.get("batch") != 16 or manifest.get("seed") != 117 + seed
            or manifest.get("training_ids") != np.asarray(ids).tolist()
            or manifest.get("ground_truth_used_for_training") is not False
            or report.get("checkpoint") != str(source)
            or report.get("ids") != [1320, 1639] or report.get("images") != 320
            or report.get("ground_truth_used_for_prediction") is not False):
        raise AssertionError(f"SW0097 seed-{seed} frozen source evaluation contract mismatch")
    score = report["sweep"][0]["scored_targets"]["our_hdf5"]
    if (set(score.get("metrics", {})) != expected
            or set(score.get("valid_count", {})) != expected
            or set(score.get("per_image", {})) != expected):
        raise AssertionError(f"SW0097 seed-{seed} source score schema mismatch")
    metrics = {}
    for name in sorted(expected):
        values = score["per_image"][name]
        mean = float(score["metrics"][name])
        if (score["valid_count"][name] != 320 or len(values) != 320
                or not all(math.isfinite(float(x)) for x in values)
                or not math.isfinite(mean)
                or abs(sum(map(float, values)) / 320.0 - mean) > 1e-12):
            raise AssertionError(f"invalid SW0097 seed-{seed} source metric {name}")
        metrics[name] = mean
    return metrics


def source_contract(seed):
    """Generalize SW0115's source-manifest checks without mutating its seed-0 API."""
    if seed not in SEEDS:
        raise ValueError(f"unsupported SW0122 seed {seed}")
    source, manifest_path, manifest = base.source_paths(seed)
    ids, rows = base.train_indices(seed)
    if (len(ids) != 4096 or len(np.unique(ids)) != 4096
            or manifest.get("status") != "complete"
            or manifest.get("unique_images_seen") != 4096
            or manifest.get("steps") != 256 or manifest.get("batch") != 16
            or manifest.get("seed") != 117 + seed
            or manifest.get("training_ids") != ids.tolist()
            or manifest.get("ground_truth_used_for_training") is not False
            or sha(manifest_path) != EXPECTED_SOURCE_TRAIN_MANIFEST_SHAS[seed]
            or sha(source) != base.EXPECTED_SOURCE_SHAS[seed]):
        raise AssertionError(f"registered SW0097 seed-{seed} source manifest/checkpoint mismatch")
    return source, manifest_path, manifest, ids, rows


def load_models(seed, device):
    source, source_manifest, manifest, ids, rows = source_contract(seed)
    if (sha(sw117.ENCODER_PATH) != base.EXPECTED_ENCODER_SHA256
            or sha(sw117.STATS_PATH) != base.EXPECTED_PREPROCESSING_SHA256):
        raise AssertionError("registered encoder/preprocessing SHA mismatch")
    core = base.make_core(device, TRAIN_STEPS)
    core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    core.requires_grad_(True)
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.size(0))]
    if core.kuramoto.spike_pulse_gain is not None or core.graph_generator.uses_feedback:
        raise AssertionError("source graph feedback/spike pulse differs from SW0117")
    encoder = sw117.load_input_encoder(str(sw117.ENCODER_PATH), num_kernels=8,
                                       kernel_size=3, channels=3, device=device)
    encoder.requires_grad_(True)
    patcher = sw117.FeaturePatchGammaInitializer(grid_size=16).to(device)
    stats = torch.load(sw117.STATS_PATH, map_location="cpu", weights_only=True)
    mean, std, clip = sw117.preprocessing_tensors(stats, device)
    return (core, encoder, patcher, mean, std, clip, source, source_manifest,
            manifest, ids, rows)


def _seed_lambda(arm):
    if arm == "control":
        return 0.0
    if arm != "analytic_candidate":
        raise ValueError(arm)
    validate_historical_seed0()
    return LAMBDA


def preflight(seed, arm, output, device="cuda"):
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("SW0122 only registers seeds 1/2 and the matched control/candidate arms")
    output = require_vacant(output, "SW0122 preflight")
    historical = validate_historical_seed0()
    source_ref = validate_source_reference(seed)
    lam = _seed_lambda(arm)
    fingerprint = implementation_fingerprint()
    (core, encoder, patcher, mean, std, clip, source, source_manifest, source_record,
     ids, rows) = load_models(seed, device)
    cached_gamma, _gamma_meta = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_meta, rgb_cache_sha = sw117.load_rgb_training_cache()
    families = sw117.eligible_params(core, encoder)
    initial_core, initial_encoder = sw117.clone_state(core), sw117.clone_state(encoder)
    reference = base.make_core(device, TRAIN_STEPS)
    reference.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    reference._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.size(0))]
    reference.train(); core.train(); core.graph_generator.train(); encoder.train()
    lossfn = sw117.criterion()
    records, real_values, scrambled_values = [], [], []
    permutation = torch.as_tensor(np.random.default_rng(11501).permutation(256),
                                  dtype=torch.long, device=device)
    torch.manual_seed(117 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117 + seed)
    for bi in range(4):
        start = bi * BATCH
        batch_rows = rows[start:start + BATCH]
        images = sw117.read_rgb(rgb_cache, batch_rows, device)
        patches = rgb_base.rgb_patch_means(images / 255.0)
        gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, images)
        cached = cached_gamma[torch.as_tensor(batch_rows, dtype=torch.long)].to(device)
        parts, parity = sw117.compare_initial_rollout(core, reference, gamma, cached, patches, lossfn)
        old, primary, positive, rloss, q, _labels, hard, per_image, *_ = parts
        stats, old_by, rgb_by = sw117.collect_joint_gradients(
            old, rloss, families, candidate=(arm == "analytic_candidate"))
        q_grad = (torch.autograd.grad(rloss, q, retain_graph=True, allow_unused=True)[0]
                  if arm == "analytic_candidate" else None)
        qnorm = 0.0 if q_grad is None else sw117.grad_norm((q_grad,))
        sw117.validate_gradient_families(stats, arm, qnorm)
        if arm == "analytic_candidate":
            for family in ("encoder", "graph", "core", "joint"):
                norm = float(stats[family]["rgb_gradient_norm"])
                if not math.isfinite(norm) or norm <= 0:
                    raise FloatingPointError(f"candidate RGB gradient is invalid for {family}")
        shuffled_h = [h.index_select(0, permutation) for h in hard]
        _scrambled, scrambled_per, _, _ = batch_reconstruction_loss(q, shuffled_h, patches)
        if not torch.isfinite(scrambled_per).all():
            raise FloatingPointError("nonfinite fixed count-preserving H scramble")
        real_values.extend(float(v.detach()) for v in per_image)
        scrambled_values.extend(float(v.detach()) for v in scrambled_per)
        records.append({"batch": bi, "global_ids": ids[start:start+BATCH].tolist(),
                        "cache_live_parity": parity,
                        "old_loss": float(old.detach()), "primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "rgb_loss": float(rloss.detach()),
                        "old_gradient_norms": {f: stats[f]["old_gradient_norm"]
                                               for f in ("encoder", "graph", "core", "joint")},
                        "rgb_gradient_norms": {f: stats[f]["rgb_gradient_norm"]
                                               for f in ("encoder", "graph", "core", "joint")},
                        "old_rgb_gradient_cosines": {f: stats[f]["gradient_cosine"]
                                                      for f in ("encoder", "graph", "core", "joint")},
                        "rgb_to_Q_gradient_norm": qnorm,
                        "group_counts": [int(h.shape[1]) for h in hard]})
        core.zero_grad(set_to_none=True); encoder.zero_grad(set_to_none=True)
    scramble_excess = rgb_base.mean_scramble_excess(real_values, scrambled_values)
    if not math.isfinite(scramble_excess) or scramble_excess <= 0:
        raise AssertionError("fixed count-preserving H row scramble did not increase RGB loss")
    if any(not torch.equal(initial_core[k], core.state_dict()[k]) for k in initial_core):
        raise AssertionError("preflight modified source core before throwaway update")
    if any(not torch.equal(initial_encoder[k], encoder.state_dict()[k]) for k in initial_encoder):
        raise AssertionError("preflight modified source encoder before throwaway update")

    # Independent disposable update proves the actual optimizer path without
    # changing the preflight model or any registered source artifact.
    (tc, te, tp, tm, ts, tclip, *_rest) = load_models(seed, device)
    tc.train(); tc.graph_generator.train(); te.train()
    tf = sw117.eligible_params(tc, te)
    opt = sw117.optimizer_for(tc, te)
    torch.manual_seed(117 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117 + seed)
    first = sw117.read_rgb(rgb_cache, rows[:BATCH], device)
    first_patches = rgb_base.rgb_patch_means(first / 255.0)
    first_gamma = sw117.encode_rgb(te, tp, tm, ts, tclip, first)
    first_cached = cached_gamma[torch.as_tensor(rows[:BATCH], dtype=torch.long)].to(device)
    throw, _ = sw117.compare_initial_rollout(tc, reference, first_gamma, first_cached,
                                             first_patches, lossfn)
    before_core, before_encoder = sw117.clone_state(tc), sw117.clone_state(te)
    opt.zero_grad(set_to_none=True)
    _, old_by, rgb_by = sw117.collect_joint_gradients(
        throw[0], throw[3], tf, candidate=(arm == "analytic_candidate"))
    sw117.apply_family_gradients(tf, old_by, rgb_by, lam)
    gradnorm = torch.nn.utils.clip_grad_norm_(tf["all"], CLIP_NORM)
    total = throw[0] + lam * throw[3]
    if not torch.isfinite(total) or not torch.isfinite(gradnorm) or float(gradnorm) <= 0:
        raise FloatingPointError("invalid disposable SW0122 B16 gradient/update")
    opt.step()
    changed = sw117.changed_trainable_groups(before_core, tc.state_dict(),
                                             before_encoder, te.state_dict())
    if not all(changed.values()):
        raise AssertionError(f"throwaway update did not change all trainable groups: {changed}")

    record = {"status": "passed", "experiment": "SW0122", "seed": seed, "arm": arm,
              "implementation_fingerprint": fingerprint,
              "historical_sw0117_seed0": historical,
              "source_core_sha256": source_ref["source_core_sha256"],
              "source_manifest_sha256": source_ref["source_manifest_sha256"],
              "source_evaluation_sha256": source_ref["source_evaluation_sha256"],
              "source_evaluation_manifest_sha256": source_ref["source_evaluation_manifest_sha256"],
              "source_metrics": source_ref["source_metrics"],
              "encoder_sha256": sha(sw117.ENCODER_PATH), "preprocessing_sha256": sha(sw117.STATS_PATH),
              "gamma_train_sha256": sha(base.GAMMA_TRAIN),
              "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
              "rgb_cache_sha256": rgb_cache_sha, "rgb_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
              "rgb_cache_source_size_bytes": rgb_meta.get("source_size_bytes"),
              "rgb_cache_source_mtime_ns": rgb_meta.get("source_mtime_ns"),
              "training_ids": ids.tolist(), "training_ids_sha256": source_ref["training_ids_sha256"],
              "matched_shuffle_seed": 117 + seed, "batch_size": BATCH,
              "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE,
              "ground_truth_used": False, "batches": records,
              "lambda_joint": lam, "lambda_artifact_sha256": sha(LAMBDA_PATH) if lam else None,
              "lambda_source": "immutable SW0117 seed0 calibration; no per-seed recalibration",
              "fixed_rows_scramble_rgb_loss_excess": scramble_excess,
              "throwaway_b16_adam_update": True,
              "throwaway_gradient_norm_preclip": float(gradnorm),
              "throwaway_parameter_groups_changed": changed}
    write(output, record)
    return record


def train(seed, arm, output, device="cuda", steps=UPDATES):
    if seed not in SEEDS or arm not in ARMS or steps != UPDATES:
        raise ValueError("SW0122 enables only matched seed1/2, 256-update arms")
    historical = validate_historical_seed0()
    source_ref = validate_source_reference(seed)
    lam = _seed_lambda(arm)
    output = require_vacant(output, "SW0122 training output")
    pf_path = ARCHIVE / f"preflight_seed{seed}_{arm}.json"
    if not pf_path.is_file():
        raise FileNotFoundError(pf_path)
    pf = json.loads(pf_path.read_text(encoding="utf-8"))
    if (pf.get("status") != "passed" or pf.get("experiment") != "SW0122"
            or pf.get("seed") != seed or pf.get("arm") != arm
            or pf.get("implementation_fingerprint") != implementation_fingerprint()
            or pf.get("source_evaluation_sha256") != source_ref["source_evaluation_sha256"]
            or pf.get("lambda_joint") != lam):
        raise AssertionError("current SW0122 preflight/source/immutable-lambda bindings are required")
    (core, encoder, patcher, mean, std, clip, source, source_manifest, source_record,
     ids, rows) = load_models(seed, device)
    gamma_cache, _gamma_meta = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_meta, rgb_cache_sha = sw117.load_rgb_training_cache()
    bindings = {"source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
                "encoder_sha256": sha(sw117.ENCODER_PATH), "preprocessing_sha256": sha(sw117.STATS_PATH),
                "gamma_train_sha256": sha(base.GAMMA_TRAIN),
                "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
                "rgb_cache_sha256": rgb_cache_sha,
                "rgb_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
                "training_ids_sha256": source_ref["training_ids_sha256"],
                "source_evaluation_sha256": source_ref["source_evaluation_sha256"],
                "implementation_fingerprint": implementation_fingerprint()}
    if any(pf.get(key) != value for key, value in bindings.items()):
        raise AssertionError("source/cache/order/implementation changed after SW0122 preflight")
    families = sw117.eligible_params(core, encoder)
    optimizer = sw117.optimizer_for(core, encoder)
    lossfn = sw117.criterion()
    torch.manual_seed(117 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117 + seed)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "training", "experiment": "SW0122", "seed": seed, "arm": arm,
                "source_core": str(source), "source_core_sha256": sha(source),
                "source_manifest_sha256": sha(source_manifest), "source_evaluation_sha256": source_ref["source_evaluation_sha256"],
                "preflight_sha256": sha(pf_path), "implementation_fingerprint": implementation_fingerprint(),
                "historical_sw0117_seed0": historical,
                "encoder_source_sha256": sha(sw117.ENCODER_PATH),
                "preprocessing_sha256": sha(sw117.STATS_PATH),
                "gamma_train_sha256": sha(base.GAMMA_TRAIN),
                "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
                "rgb_cache_sha256": rgb_cache_sha,
                "rgb_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
                "training_ids": ids.tolist(), "training_ids_sha256": source_ref["training_ids_sha256"],
                "matched_shuffle_seed": 117 + seed, "updates": steps, "batch_size": BATCH,
                "core_graph_lr": CORE_GRAPH_LR, "encoder_lr": ENCODER_LR,
                "clip_norm": CLIP_NORM, "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE,
                "lambda_joint": lam, "lambda_artifact_sha256": sha(LAMBDA_PATH) if lam else None,
                "objective": "phase_primary + 5*positive_actual_spike_product + immutable_lambda*analytic_partition_rgb_if_candidate",
                "ground_truth_used_for_training": False}
    write(output / "manifest.json", manifest)
    history = []
    core.train(); core.graph_generator.train(); encoder.train()
    for update in range(steps):
        batch_rows = rows[update * BATCH:(update + 1) * BATCH]
        images = sw117.read_rgb(rgb_cache, batch_rows, device)
        patches = rgb_base.rgb_patch_means(images / 255.0)
        gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, images)
        parts = sw117.objective_parts(core, gamma, patches, lossfn)
        old, primary, positive, rloss, _q, _labels, hard, per_image, *_ = parts
        total = old + lam * rloss if arm == "analytic_candidate" else old
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite SW0122 objective at update {update+1}")
        optimizer.zero_grad(set_to_none=True)
        stats, old_by, rgb_by = sw117.collect_joint_gradients(
            old, rloss, families, candidate=(arm == "analytic_candidate"))
        sw117.apply_family_gradients(families, old_by, rgb_by, lam)
        gradnorm = torch.nn.utils.clip_grad_norm_(families["all"], CLIP_NORM)
        if not torch.isfinite(gradnorm) or float(gradnorm) <= 0:
            raise FloatingPointError(f"invalid SW0122 joint gradient at update {update+1}")
        optimizer.step()
        history.append({"update": update + 1, "total": float(total.detach()),
                        "old": float(old.detach()), "primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "rgb": float(rloss.detach()), "lambda_joint": lam,
                        "gradient_norm_preclip": float(gradnorm),
                        "old_rgb_gradient_stats": stats,
                        "mean_groups": float(np.mean([h.shape[1] for h in hard])),
                        "mean_rgb_per_image": float(per_image.detach().mean())})
        if update == 0 or (update + 1) % 32 == 0:
            write(output / "progress.json", {"status": "training", "seed": seed,
                  "arm": arm, "update": update + 1, "total_updates": steps})
    if any(not torch.isfinite(p).all() for p in list(core.parameters()) + list(encoder.parameters())):
        raise FloatingPointError("nonfinite final SW0122 model parameter")
    torch.save(core.state_dict(), output / "core.pt")
    torch.save(encoder.state_dict(), output / "encoder.pt")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    write(output / "history.json", history)
    manifest.update(status="training_complete", completed=time.time(),
                    core_sha256=sha(output / "core.pt"), encoder_sha256=sha(output / "encoder.pt"),
                    optimizer_sha256=sha(output / "optimizer.pt"), history_sha256=sha(output / "history.json"))
    write(output / "manifest.json", manifest)
    (output / "TRAINING_COMPLETED").write_text("complete\n", encoding="utf-8")


def evaluate(seed, arm, checkpoint, output, device="cuda"):
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("SW0122 evaluation only accepts registered seed1/2 arms")
    checkpoint, output = Path(checkpoint), Path(output)
    sidecar_path = output.parent / "evaluation_manifest.json"
    if output.exists() or sidecar_path.exists():
        raise FileExistsError("preserve existing SW0122 evaluation/sidecar")
    folder = checkpoint.parent
    manifest_path, encoder_path = folder / "manifest.json", folder / "encoder.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "training_complete" or manifest.get("experiment") != "SW0122"
            or manifest.get("seed") != seed or manifest.get("arm") != arm
            or manifest.get("core_sha256") != sha(checkpoint)
            or manifest.get("encoder_sha256") != sha(encoder_path)
            or manifest.get("history_sha256") != sha(folder / "history.json")
            or manifest.get("optimizer_sha256") != sha(folder / "optimizer.pt")):
        raise AssertionError("evaluation checkpoint is not bound to completed SW0122 training")
    rgb_cache, _rgb_meta, rgb_cache_sha = sw117.validate_rgb_validation_cache()
    _, encoder, patcher, mean, std, clip, *_ = load_models(seed, device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True), strict=True)
    encoder.eval(); patcher.eval()
    gammas = []
    with torch.no_grad():
        for start in range(0, 320, 8):
            images = sw117.read_rgb(rgb_cache, np.arange(start, start + 8), device)
            gammas.append(sw117.encode_rgb(encoder, patcher, mean, std, clip, images).cpu())
    gamma = torch.cat(gammas, dim=0)
    gamma_path = output.parent / "trained_encoder_gamma_validation.pt"
    gamma_manifest_path = output.parent / "trained_encoder_gamma_validation_manifest.json"
    if gamma_path.exists() or gamma_manifest_path.exists():
        raise FileExistsError("preserve existing SW0122 trained-encoder gamma artifacts")
    tmp = gamma_path.with_suffix(".pt.tmp")
    torch.save(gamma, tmp); tmp.replace(gamma_path)
    write(gamma_manifest_path, {"status": "complete", "kind": "validation",
          "image_ids": [1320, 1639], "gamma_sha256": sha(gamma_path),
          "encoder_sha256": sha(encoder_path), "preprocessing_sha256": sha(sw117.STATS_PATH),
          "rgb_cache_sha256": rgb_cache_sha, "rgb_cache_manifest_sha256": sha(sw117.VAL_RGB_MANIFEST),
          "ground_truth_used_for_gamma": False})
    old_gamma, old_meta, old_validator = base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST, base.validate_gamma_cache
    def validate_trained(path, meta_path, validation=False):
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (not validation or Path(path) != gamma_path or Path(meta_path) != gamma_manifest_path
                or tuple(value.shape) != (320, 8, 256) or meta.get("gamma_sha256") != sha(gamma_path)
                or meta.get("encoder_sha256") != sha(encoder_path)
                or meta.get("image_ids") != [1320, 1639]):
            raise AssertionError("SW0122 trained-encoder evaluation gamma is invalid")
        return value, meta
    try:
        base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST = gamma_path, gamma_manifest_path
        base.validate_gamma_cache = validate_trained
        base.evaluate(seed, "control", checkpoint, output, device=device)
    finally:
        base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST, base.validate_gamma_cache = old_gamma, old_meta, old_validator
    report = json.loads(output.read_text(encoding="utf-8"))
    report.update(experiment="SW0122", seed=seed, arm=arm,
                  training_manifest_sha256=sha(manifest_path), checkpoint_sha256=sha(checkpoint),
                  encoder_checkpoint_sha256=sha(encoder_path),
                  trained_encoder_gamma_sha256=sha(gamma_path),
                  trained_encoder_gamma_manifest_sha256=sha(gamma_manifest_path),
                  evaluation_runner_sha256=sha(RUNNER), ground_truth_used_for_prediction=False)
    write(output, report)
    write(sidecar_path, {"experiment": "SW0122", "seed": seed, "arm": arm,
          "evaluation_file": output.name, "evaluation_sha256": sha(output),
          "training_manifest_sha256": sha(manifest_path), "checkpoint_sha256": sha(checkpoint),
          "encoder_checkpoint_sha256": sha(encoder_path), "evaluation_runner_sha256": sha(RUNNER),
          "shared_evaluator_sha256": sha(ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"),
          "trained_encoder_gamma_sha256": sha(gamma_path),
          "trained_encoder_gamma_manifest_sha256": sha(gamma_manifest_path),
          "evaluation_contract": {"ids": [1320, 1639], "images": 320, "batch_size": 8,
            "time_steps": 1024, "settle": 512, "membrane_vth": 0.06,
            "readout_threshold": 0.50, "min_group_size": 2,
            "background": "largest_component", "ground_truth_used_for_prediction": False}})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("validate-baselines", "preflight", "train", "evaluate"):
        sp = sub.add_parser(stage)
        sp.add_argument("--seed", type=int, choices=SEEDS)
        sp.add_argument("--arm", choices=ARMS)
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--output", type=Path)
        if stage == "evaluate":
            sp.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "validate-baselines":
        record = {"status": "passed", "experiment": "SW0122",
                  "historical_sw0117_seed0": validate_historical_seed0()}
        if args.output is not None:
            write(args.output, record)
    else:
        if args.seed not in SEEDS or args.arm not in ARMS or args.output is None:
            parser.error("preflight/train/evaluate require registered --seed, --arm, and --output")
        if args.stage == "preflight": record = preflight(args.seed, args.arm, args.output, args.device)
        elif args.stage == "train":
            train(args.seed, args.arm, args.output, args.device)
            record = {"status": "complete", "experiment": "SW0122", "stage": "train"}
        else:
            evaluate(args.seed, args.arm, args.checkpoint, args.output, args.device)
            record = {"status": "complete", "experiment": "SW0122", "stage": "evaluate"}
    print(json.dumps({"status": record.get("status", "passed"), "stage": args.stage,
                      "seed": args.seed, "arm": args.arm}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()

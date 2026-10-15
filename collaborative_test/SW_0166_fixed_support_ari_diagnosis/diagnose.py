from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "collaborative_test") not in sys.path:
    sys.path.insert(0, str(ROOT / "collaborative_test"))

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0165_foreground_partition_replication import ari_bound
from snn_kuramoto_bidirectional import evaluation as production_eval

SEEDS = (0, 1, 2)
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
COUNT = contract.COUNT
GRID = contract.GRID
BACKGROUND = 0
BOOTSTRAP_SEED = 166
BOOTSTRAP_REPLICATES = 10_000
SW0164_SEED1_SCORE_SHA256 = "978a9d97658837eb19f414a7fb7dbb35d86c6135cb1213e12e63e2837fd65df6"
DATASET = Path(sw130.source97.DATASET)
OUTPUT = HERE / "results_archive" / "diagnosis_20261014.json"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _counts(labels: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return counts


def _choose2(n):
    n = np.asarray(n, dtype=np.int64)
    return n * (n - 1) // 2


def _pair_summary(gt: np.ndarray, candidate: np.ndarray, support: np.ndarray) -> dict:
    gt = np.asarray(gt, dtype=np.int64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.int64).reshape(-1)
    support = np.asarray(support, dtype=bool).reshape(-1)
    if gt.shape != candidate.shape or gt.shape != support.shape:
        raise ValueError("pair-summary inputs must have equal flattened shapes")
    if gt.size == 0 or np.any(gt < 0) or np.any(candidate < 0):
        raise ValueError("pair-summary labels must be nonempty and nonnegative")
    gt_fg = gt != BACKGROUND
    covered = gt_fg & support
    missed = gt_fg & ~support
    gt_ids = np.unique(gt[gt_fg])
    covered_gt_counts = np.asarray([np.count_nonzero(covered & (gt == g)) for g in gt_ids], dtype=np.int64)
    missed_gt_counts = np.asarray([np.count_nonzero(missed & (gt == g)) for g in gt_ids], dtype=np.int64)
    same_gt_pairs = int(np.sum(_choose2(np.asarray([np.count_nonzero(gt == g) for g in gt_ids], dtype=np.int64))))
    total_fg_pairs = int(_choose2(np.count_nonzero(gt_fg)))
    different_gt_pairs = total_fg_pairs - same_gt_pairs

    cov_gt = gt[covered]
    cov_pred = candidate[covered]
    gt_values, row_inv, row_counts = np.unique(cov_gt, return_inverse=True, return_counts=True)
    pred_values, col_inv, col_counts = np.unique(cov_pred, return_inverse=True, return_counts=True)
    if cov_gt.size:
        joint_idx = row_inv * pred_values.size + col_inv
        joint = np.bincount(joint_idx, minlength=gt_values.size * pred_values.size).reshape(gt_values.size, pred_values.size)
    else:
        joint = np.zeros((gt_values.size, pred_values.size), dtype=np.int64)
    covered_same_gt_pairs = int(np.sum(_choose2(row_counts)))
    covered_same_pred_pairs = int(np.sum(_choose2(col_counts)))
    covered_same_both_pairs = int(np.sum(_choose2(joint)))
    covered_split = covered_same_gt_pairs - covered_same_both_pairs
    covered_merge = covered_same_pred_pairs - covered_same_both_pairs
    covered_diff_gt_pairs = int(_choose2(np.count_nonzero(covered)) - covered_same_gt_pairs)

    miss_total_pairs = int(_choose2(np.count_nonzero(missed)))
    missed_same_gt_pairs = int(np.sum(_choose2(missed_gt_counts)))
    forced_bg_merge = miss_total_pairs - missed_same_gt_pairs
    forced_cross_split = int(np.sum(covered_gt_counts * missed_gt_counts))
    cross_objects = int(np.count_nonzero((covered_gt_counts > 0) & (missed_gt_counts > 0)))

    def rate(num, den):
        return float(num / den) if den else None

    return {
        "foreground_patch_count": int(np.count_nonzero(gt_fg)),
        "foreground_pair_count": total_fg_pairs,
        "same_gt_pair_count": same_gt_pairs,
        "different_gt_pair_count": different_gt_pairs,
        "covered_patch_count": int(np.count_nonzero(covered)),
        "missed_patch_count": int(np.count_nonzero(missed)),
        "covered_gt_object_count": int(np.count_nonzero(covered_gt_counts)),
        "missed_gt_object_count": int(np.count_nonzero(missed_gt_counts)),
        "cross_support_gt_object_count": cross_objects,
        "covered_merge_pair_count": int(covered_merge),
        "covered_merge_pair_denominator": covered_diff_gt_pairs,
        "covered_merge_rate": rate(covered_merge, covered_diff_gt_pairs),
        "covered_split_pair_count": int(covered_split),
        "covered_split_pair_denominator": covered_same_gt_pairs,
        "covered_split_rate": rate(covered_split, covered_same_gt_pairs),
        "forced_bg_merge_pair_count": int(forced_bg_merge),
        "forced_bg_merge_pair_denominator": int(miss_total_pairs - missed_same_gt_pairs),
        "forced_bg_merge_rate": rate(forced_bg_merge, miss_total_pairs - missed_same_gt_pairs),
        "forced_cross_support_split_pair_count": forced_cross_split,
        "forced_cross_support_split_pair_denominator": forced_cross_split,
        "forced_cross_support_split_rate": (1.0 if forced_cross_split else None),
    }


def _split_positive_by_gt(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    if gt.shape != pred.shape:
        raise ValueError("GT/prediction shapes differ")
    out = np.zeros(pred.size, dtype=np.int64)
    g = gt.reshape(-1)
    p = pred.reshape(-1)
    pos = p > 0
    pairs = np.stack((p[pos], g[pos]), axis=1)
    if pairs.size:
        _, inverse = np.unique(pairs, axis=0, return_inverse=True)
        out[pos] = inverse + 1
    return out.reshape(pred.shape)


def _idealized_attachment(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    if gt.shape != pred.shape:
        raise ValueError("GT/prediction shapes differ")
    out = pred.copy()
    next_label = int(pred.max(initial=0)) + 1
    for obj in np.unique(gt):
        if int(obj) == BACKGROUND:
            continue
        objmask = gt == obj
        missed = objmask & (pred == BACKGROUND)
        if not missed.any():
            continue
        labels, counts = np.unique(pred[objmask & (pred > 0)], return_counts=True)
        if labels.size:
            # np.unique sorts labels: argmax deterministically chooses the smallest label on ties.
            chosen = int(labels[int(np.argmax(counts))])
        else:
            chosen = next_label
            next_label += 1
        out[missed] = chosen
    return out


def _production_fg_ari_one(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = np.asarray(gt, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    keep = gt != BACKGROUND
    return ari_bound.adjusted_rand_from_labels(gt[keep], pred[keep])


def diagnose_one(gt: np.ndarray, candidate: np.ndarray, source_t: np.ndarray) -> dict:
    gt = np.asarray(gt, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.int64)
    source_t = np.asarray(source_t, dtype=np.int64)
    if gt.shape != (GRID, GRID) or candidate.shape != gt.shape or source_t.shape != gt.shape:
        raise ValueError("each image must contain three 32x32 integer label maps")
    if min(int(gt.min()), int(candidate.min()), int(source_t.min())) < 0:
        raise ValueError("labels must be nonnegative")
    support = source_t > 0
    if not np.array_equal(candidate > 0, support):
        raise ValueError("candidate foreground membership differs from its Source-T support")
    actual = _production_fg_ari_one(gt, candidate)
    bound = ari_bound.fixed_support_ari_upper_bound(gt, support)
    after_split_labels = _split_positive_by_gt(gt, candidate)
    after_split = _production_fg_ari_one(gt, after_split_labels)
    attached = _idealized_attachment(gt, candidate)
    idealized = _production_fg_ari_one(gt, attached)
    if np.isfinite(actual) and np.isfinite(bound):
        if bound + 1e-12 < actual or after_split + 1e-12 < actual or after_split > bound + 1e-12:
            raise AssertionError("fixed-support bound/counterfactual ordering violated")
        merge_delta = after_split - actual
        split_delta = bound - after_split
        if abs((merge_delta + split_delta) - (bound - actual)) > 1e-12:
            raise AssertionError("registered sequential correction deltas do not telescope")
    else:
        merge_delta = split_delta = float("nan")
    pairs = _pair_summary(gt, candidate, support)
    return {
        "actual_fg_ari": actual,
        "after_split_mixed_positive_fg_ari": after_split,
        "fixed_support_ari_upper_bound": bound,
        "support_idealized_attachment_fg_ari": idealized,
        "merge_correction_delta": merge_delta,
        "split_correction_delta": split_delta,
        **pairs,
    }


def _bootstrap_contrast(values: np.ndarray, image_indices: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or image_indices.ndim != 2 or image_indices.shape[1] != values.size:
        raise ValueError("bootstrap input shape mismatch")
    valid = np.isfinite(values)
    if not valid.all():
        raise ValueError("bootstrap contrast must already use the common valid-image population")
    sampled = values[image_indices].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(x) for x in np.percentile(sampled, [2.5, 97.5])],
            "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "valid_count": int(valid.sum())}


def _metric_vector(labels: np.ndarray, gt: np.ndarray, name="fg_ari") -> np.ndarray:
    if name != "fg_ari":
        raise ValueError("SW0166 image-level metric is the production foreground ARI")
    labels=np.asarray(labels,dtype=np.int64); gt=np.asarray(gt,dtype=np.int64)
    if labels.shape!=(COUNT,GRID,GRID) or gt.shape!=labels.shape:
        raise ValueError("foreground ARI inputs must be the fixed320 native32 arrays")
    result=np.asarray([_production_fg_ari_one(gt[i],labels[i]) for i in range(COUNT)],dtype=np.float64)
    # When every image is valid, cross-check this direct per-image path against
    # the production scorer. Its batch wrapper intentionally rejects NaN rows;
    # the preregistered report still records the exact shared valid count.
    if np.isfinite(result).all():
        production=contract.production_metrics32(labels,gt)[name]
        if production["valid_count"]!=COUNT or not np.allclose(result,production["per_image"],rtol=0.0,atol=1e-12):
            raise AssertionError("direct FG-ARI differs from the production scorer")
    return result


def _all_image_diagnostics(gt: np.ndarray, candidate: np.ndarray, source_t: np.ndarray):
    rows = [diagnose_one(gt[i], candidate[i], source_t[i]) for i in range(COUNT)]
    result = {key: np.asarray([r[key] for r in rows], dtype=np.float64) for key in (
        "actual_fg_ari", "after_split_mixed_positive_fg_ari", "fixed_support_ari_upper_bound",
        "support_idealized_attachment_fg_ari", "merge_correction_delta", "split_correction_delta")}
    return rows, result


def _artifact_path(ref: dict) -> Path:
    for key in ("prediction_path", "npz_path", "archive_path"):
        if ref.get(key) is not None:
            return Path(ref[key])
    raise ValueError("validated prediction reference has no archive path")


def _reference_hashes(prepared: dict) -> dict:
    out = {}
    for name, ref in prepared["references"].items():
        if name not in ("source_T", "slot", "source_qcc"):
            continue
        p = _artifact_path(ref)
        proto = Path(ref["protocol_path"])
        out[name] = {"prediction_path": str(p.resolve()), "prediction_sha256": sha256_file(p),
                     "protocol_path": str(proto.resolve()), "protocol_sha256": sha256_file(proto)}
    return out


def implementation_fingerprint() -> dict:
    """Hash this diagnostic and every public validator/helper it relies on."""
    eval164, eval165, queue165 = _load_modules_and_prepare()
    from collaborative_test.SW_0165_foreground_partition_replication import preflight as preflight165
    from collaborative_test.SW_0164_foreground_preserving_partition import preflight as preflight164
    source97_module = sw130.source97
    paths = (
        HERE / "diagnose.py", HERE / "diagnosis_queue.py", HERE / "protocol.json", HERE / "DESIGN_KO.md",
        Path(sw130.__file__), Path(source97_module.__file__), Path(ari_bound.__file__), Path(contract.__file__),
        Path(production_eval.__file__),
        Path(eval164.__file__), Path(eval165.__file__),
        Path(preflight164.__file__), Path(preflight165.__file__), Path(queue165.__file__),
    )
    result = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            name = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            name = path.resolve().as_posix()
        result[name] = sha256_file(path)
    return result


def _score_file_and_sha(module, seed: int) -> tuple[Path, str]:
    path = Path(module.score_path(seed)) if hasattr(module, "score_path") else Path(module.SCORE_PATH)
    return path, sha256_file(path)


def _load_modules_and_prepare():
    from collaborative_test.SW_0164_foreground_preserving_partition import predict_evaluate as eval164
    from collaborative_test.SW_0165_foreground_partition_replication import predict_evaluate as eval165
    from collaborative_test.SW_0165_foreground_partition_replication import stage_queue as queue165
    return eval164, eval165, queue165


def _check165_stage(seed: int, evaluation, queue_module) -> dict:
    runtime = queue_module.RUNTIME_BASE / f"stage_queue_seed{seed}_20261014"
    state_path = runtime / "state.json"
    if not state_path.is_file():
        raise RuntimeError(f"SW0165 seed{seed} queue has no terminal state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or state.get("seed") != seed:
        raise RuntimeError(f"SW0165 seed{seed} is not complete")
    if state.get("queue_fingerprint") != queue_module.implementation_fingerprint() or state.get("evaluation_fingerprint") != evaluation.implementation_fingerprint():
        raise ValueError(f"SW0165 seed{seed} terminal state fingerprint mismatch")
    score_path = Path(evaluation.score_path(seed))
    score_sha = sha256_file(score_path)
    if state.get("score_sha256") != score_sha:
        raise ValueError(f"SW0165 seed{seed} score hash differs from terminal queue receipt")
    return {"state_path": str(state_path.resolve()), "state_sha256": sha256_file(state_path), "score_path": str(score_path.resolve()), "score_sha256": score_sha}


def _freeze_predictions():
    eval164, eval165, queue165 = _load_modules_and_prepare()
    frozen = {}
    # Phase 1: prepare and validate every prediction for every seed before any score validator runs.
    for seed in SEEDS:
        if seed == 1:
            prepared = eval164.prepare_prediction_cpu()
            refs = {}
            for arm in eval164.ARMS:
                refs[arm] = eval164.validate_prediction(arm, prepared=prepared)
            score_module = eval164
            score_path = Path(eval164.SCORE_PATH)
        else:
            _check165_stage(seed, eval165, queue165)
            prepared = eval165.prepare_prediction_cpu(seed=seed)
            refs = {}
            for arm in eval165.ARMS:
                refs[arm] = eval165.validate_prediction(arm, seed=seed, prepared=prepared)
            score_module = eval165
            score_path = Path(eval165.score_path(seed))
        source_t = np.asarray(prepared["references"]["source_T"]["labels"], dtype=np.int64)
        slot_labels = np.asarray(prepared["references"]["slot"]["labels"], dtype=np.int64)
        candidate = np.asarray(refs["foreground_induced"]["labels"], dtype=np.int64)
        control = np.asarray(refs["source_T_control"]["labels"], dtype=np.int64)
        for name, labels in (("candidate", candidate), ("control", control), ("Source-T", source_t), ("Slot", slot_labels)):
            if labels.shape != (COUNT, GRID, GRID) or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0:
                raise ValueError(f"seed{seed} {name} labels violate fixed320 schema")
        if not np.array_equal(candidate > 0, source_t > 0) or not np.array_equal(control, source_t):
            raise ValueError(f"seed{seed} candidate/control does not preserve fresh Source-T reference")
        frozen[seed] = {"prepared": prepared, "predictions": refs, "candidate": candidate, "control": control,
                        "source_T": source_t, "slot": slot_labels, "score_module": score_module,
                        "score_path": score_path, "references_sha256": _reference_hashes(prepared),
                        "queue_receipt": _check165_stage(seed, eval165, queue165) if seed != 1 else None}

    # Freeze/check score report bytes before any score validator can open GT.
    if sha256_file(frozen[1]["score_path"]) != SW0164_SEED1_SCORE_SHA256:
        raise ValueError("immutable SW0164 seed1 score changed before score validation")
    for seed in (0,2):
        if sha256_file(frozen[seed]["score_path"]) != frozen[seed]["queue_receipt"]["score_sha256"]:
            raise ValueError(f"seed{seed} SW0165 score differs from its terminal queue receipt")
    # Phase 2: all predictions and score bytes are frozen; public score validators may now recompute GT-last metrics.
    for seed in SEEDS:
        row = frozen[seed]
        module = row["score_module"]
        if seed == 1:
            checked = module.validate_score(row["score_path"], prepared=row["prepared"])
            expected_sha = SW0164_SEED1_SCORE_SHA256
        else:
            checked = module.validate_score(row["score_path"], seed=seed, prepared=row["prepared"])
            expected_sha = row["queue_receipt"]["score_sha256"]
        score_sha = sha256_file(row["score_path"])
        if score_sha != expected_sha or checked.get("score_sha256") != score_sha:
            raise ValueError(f"seed{seed} public score receipt hash mismatch")
        row["score_sha256"] = score_sha
        row["score_checked"] = checked
    # Rehash source/Slot assets and prediction outputs after all public validators, immediately before diagnostic GT.
    for seed in SEEDS:
        row = frozen[seed]
        now = _reference_hashes(row["prepared"])
        if now != row["references_sha256"]:
            raise ValueError(f"seed{seed} reference prediction changed after freezing")
        for arm, pred in row["predictions"].items():
            folder = Path(pred["directory"])
            pred_path = folder / "predictions.npz"
            protocol_path = folder / "protocol.json"
            if pred.get("prediction_sha256") != sha256_file(pred_path) or pred.get("protocol_sha256") != sha256_file(protocol_path):
                raise ValueError(f"seed{seed} {arm} artifact changed after freezing")
        if sha256_file(row["score_path"]) != row["score_sha256"]:
            raise ValueError(f"seed{seed} score report changed after validation")
        if seed == 1 and row["score_sha256"] != SW0164_SEED1_SCORE_SHA256:
            raise ValueError("immutable SW0164 seed1 score changed")
        if seed != 1:
            if _check165_stage(seed, eval165, queue165) != row["queue_receipt"]:
                raise ValueError(f"seed{seed} terminal SW0165 queue receipt changed")
    return frozen


def _read_canonical_gt(dataset: Path) -> tuple[np.ndarray, dict]:
    import h5py
    expected = Path(DATASET).resolve()
    if Path(dataset).resolve() != expected:
        raise ValueError("diagnostic only reads the canonical SW0130 validation dataset")
    st = dataset.stat()
    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= int(IMAGE_IDS[-1]):
            raise ValueError("canonical mask dataset does not contain fixed validation rows")
        raw = np.asarray(h5["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1]) + 1])
    gt = contract.modal_native32(raw)
    after = dataset.stat()
    if st.st_size != after.st_size or st.st_mtime_ns != after.st_mtime_ns:
        raise ValueError("canonical HDF5 identity changed during diagnostic GT read")
    return gt, {"path": str(expected), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                "image_ids": [int(IMAGE_IDS[0]), int(IMAGE_IDS[-1])], "count": COUNT,
                "mask_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()}


def _bootstrap_indices(valid_count: int = COUNT):
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    if type(valid_count) is not int or valid_count <= 0:
        raise ValueError("bootstrap needs at least one valid image")
    return rng.randint(0, valid_count, size=(BOOTSTRAP_REPLICATES, valid_count))


def _direction(ci):
    if ci[0] > 0:
        return "positive"
    if ci[1] < 0:
        return "negative"
    return "inconclusive"


def build_diagnostic(frozen: dict, gt: np.ndarray, *, dataset_info: dict) -> dict:
    if set(frozen) != set(SEEDS) or gt.shape != (COUNT, GRID, GRID):
        raise ValueError("diagnostic requires all fixed seeds and all fixed320 masks")
    per_seed = {}
    metrics = {}
    for seed in SEEDS:
        row = frozen[seed]
        cand = row["candidate"]
        src = row["source_T"]
        slot = row["slot"]
        actual = _metric_vector(cand, gt)
        source_v = _metric_vector(src, gt)
        slot_v = _metric_vector(slot, gt)
        rows, calc = _all_image_diagnostics(gt, cand, src)
        if not np.allclose(calc["actual_fg_ari"], actual, rtol=0.0, atol=1e-12, equal_nan=True):
            raise AssertionError("certified helper FG-ARI differs from production metric")
        safe_rows = []
        for source_row in rows:
            safe_rows.append({k:(None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else
                                 float(v) if isinstance(v,np.floating) else
                                 int(v) if isinstance(v,np.integer) else v)
                              for k,v in source_row.items()})
        per_seed[seed] = {
            "score_sha256": row["score_sha256"],
            "score_path": str(Path(row["score_path"]).resolve()),
            "prediction_sha256": {name: pred["prediction_sha256"] for name, pred in row["predictions"].items()},
            "candidate_fg_ari_mean": float(np.nanmean(actual)) if np.isfinite(actual).any() else None,
            "slot_fg_ari_mean": float(np.nanmean(slot_v)) if np.isfinite(slot_v).any() else None,
            "source_T_fg_ari_mean": float(np.nanmean(source_v)) if np.isfinite(source_v).any() else None,
            "valid_count": int(np.isfinite(actual).sum()),
            "valid_image_offsets": np.flatnonzero(np.isfinite(actual)).astype(int).tolist(),
            "per_image": safe_rows,
        }
        for key, values in calc.items():
            metrics.setdefault(key, {})[seed] = values
        metrics.setdefault("actual_fg_ari", {})[seed] = actual
        metrics.setdefault("source_T_fg_ari", {})[seed] = source_v
        metrics.setdefault("slot_fg_ari", {})[seed] = slot_v

    # Production FG-ARI has one common validity mask determined by GT foreground
    # count. Keep it explicit and never drop an invalid image silently.
    names = ("actual_fg_ari", "after_split_mixed_positive_fg_ari", "fixed_support_ari_upper_bound",
             "support_idealized_attachment_fg_ari", "source_T_fg_ari", "slot_fg_ari",
             "merge_correction_delta", "split_correction_delta")
    valid_masks=[np.isfinite(metrics["actual_fg_ari"][s]) for s in SEEDS]
    for name in names:
        for seed in SEEDS:
            if not np.array_equal(np.isfinite(metrics[name][seed]),valid_masks[0]):
                raise ValueError("production validity mask differs across metrics or seeds")
    valid_idx=np.flatnonzero(valid_masks[0])
    if valid_idx.size==0: raise ValueError("fixed320 has no valid foreground-ARI images")
    # Common fixed-seed bootstrap: average the same valid image indices across seeds.
    idx = _bootstrap_indices(int(valid_idx.size))
    contrasts = {}
    vectors = {name: np.mean(np.stack([metrics[name][s][valid_idx] for s in SEEDS]), axis=0) for name in names}
    for lhs, rhs in (("fixed_support_ari_upper_bound", "actual_fg_ari"),
                     ("fixed_support_ari_upper_bound", "constant_0p87"),
                     ("fixed_support_ari_upper_bound", "slot_fg_ari"),
                     ("actual_fg_ari", "constant_0p87"),
                     ("support_idealized_attachment_fg_ari", "constant_0p87"),
                     ("actual_fg_ari", "slot_fg_ari"),
                     ("support_idealized_attachment_fg_ari", "actual_fg_ari"),
                     ("merge_correction_delta", "zero"), ("split_correction_delta", "zero")):
        a = vectors[lhs]
        b = (np.full(valid_idx.size, 0.87) if rhs == "constant_0p87" else
             np.zeros(valid_idx.size) if rhs == "zero" else vectors[rhs])
        delta = a - b
        contrasts[f"{lhs}_minus_{rhs}"] = _bootstrap_contrast(delta, idx)

    fixed_means = {name: float(values.mean()) for name, values in vectors.items()}
    bound_mean = fixed_means["fixed_support_ari_upper_bound"]
    slot_mean = fixed_means["slot_fg_ari"]
    deterministic = {
        "fixed320_cannot_reach_0p87": bool(bound_mean < 0.87),
        "fixed320_cannot_strictly_beat_slot": bool(bound_mean <= slot_mean),
    }
    deterministic["fixed320_joint_goal_impossible_under_support"] = bool(
        deterministic["fixed320_cannot_reach_0p87"] or deterministic["fixed320_cannot_strictly_beat_slot"])
    generalization = {
        "bound_minus_0p87": _direction(contrasts["fixed_support_ari_upper_bound_minus_constant_0p87"]["ci95"]),
        "bound_minus_slot": _direction(contrasts["fixed_support_ari_upper_bound_minus_slot_fg_ari"]["ci95"]),
        "candidate_minus_slot": _direction(contrasts["actual_fg_ari_minus_slot_fg_ari"]["ci95"]),
    }
    raw_pair_names = ("foreground_pair_count", "same_gt_pair_count", "different_gt_pair_count",
                      "covered_merge_pair_count", "covered_split_pair_count", "forced_bg_merge_pair_count",
                      "forced_cross_support_split_pair_count")
    raw_counts = {name: {"sum_over_seed_image": int(sum(sum(int(r[name]) for r in per_seed[s]["per_image"]) for s in SEEDS)),
                         "mean_per_seed_image": float(np.mean([r[name] for s in SEEDS for r in per_seed[s]["per_image"]]))}
                  for name in raw_pair_names}
    return {
        "experiment": "SW0166_fixed_support_ari_diagnosis", "status": "complete",
        "seed_scope": list(SEEDS), "image_ids": [int(IMAGE_IDS[0]), int(IMAGE_IDS[-1])], "count": COUNT,
        "grid_size": [GRID, GRID], "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": True, "ground_truth_opened_after_all_predictions_and_scores_validated": True,
        "inputs": {str(s): {"score_sha256": frozen[s]["score_sha256"],
                            "references": frozen[s]["references_sha256"],
                            "prediction_artifacts": {name: {
                                "prediction_sha256": pred["prediction_sha256"],
                                "protocol_sha256": pred["protocol_sha256"],
                            } for name, pred in frozen[s]["predictions"].items()},
                            "queue_receipt": frozen[s]["queue_receipt"]} for s in SEEDS},
        "dataset": dataset_info, "per_seed": {str(k): v for k, v in per_seed.items()},
        "implementation_fingerprint": implementation_fingerprint(),
        "three_seed_fixed320_means": fixed_means, "paired_bootstrap": contrasts,
        "paired_bootstrap_spec": {"method": "common fixed-image resample across all three seeds, first average seeds at each image",
                                   "rng": "numpy RandomState", "seed": BOOTSTRAP_SEED,
                                   "replicates": BOOTSTRAP_REPLICATES, "valid_count": int(valid_idx.size),
                                   "valid_image_offsets": valid_idx.astype(int).tolist()},
        "fixed320_deterministic_feasibility": deterministic,
        "generalization_evidence": generalization,
        "raw_pair_counts": raw_counts,
        "interpretation_limits": [
            "The bound relaxes 4-connectivity and may be unattainable by the registered solver.",
            "The support-idealized attachment and sequential GT-conditioned splits are diagnostic counterfactuals, not predictions.",
            "Sequential correction deltas are order-dependent descriptions, not independent causal effects.",
            "This fixed validation diagnosis does not select seeds, change the recipe, or establish generalization."
        ],
    }


def run_diagnostic(*, output=OUTPUT, dataset=DATASET):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"create-only SW0166 output already exists: {output}")
    fingerprint_before = implementation_fingerprint()
    frozen = _freeze_predictions()
    # Make sure injected or resumed inputs still contain all public validation receipts.
    if set(frozen) != set(SEEDS) or any(not frozen[s].get("score_sha256") for s in SEEDS):
        raise ValueError("all three seeds must have completed public score validation")
    gt, dataset_info = _read_canonical_gt(Path(dataset))
    for seed in SEEDS:
        row = frozen[seed]
        if sha256_file(row["score_path"]) != row["score_sha256"]:
            raise ValueError(f"seed{seed} score changed during diagnostic GT read")
        if _reference_hashes(row["prepared"]) != row["references_sha256"]:
            raise ValueError(f"seed{seed} reference changed during diagnostic GT read")
        for pred in row["predictions"].values():
            folder = Path(pred["directory"])
            if (sha256_file(folder / "predictions.npz") != pred["prediction_sha256"]
                    or sha256_file(folder / "protocol.json") != pred["protocol_sha256"]):
                raise ValueError(f"seed{seed} candidate/control changed during diagnostic GT read")
    if implementation_fingerprint() != fingerprint_before:
        raise ValueError("SW0166 implementation changed during diagnostic")
    report = build_diagnostic(frozen, gt, dataset_info=dataset_info)
    report["implementation_fingerprint"] = fingerprint_before
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, output)
    except FileExistsError:
        raise
    finally:
        if temp.exists():
            temp.unlink()
    return report


if __name__ == "__main__":
    print(json.dumps(run_diagnostic(), allow_nan=False))

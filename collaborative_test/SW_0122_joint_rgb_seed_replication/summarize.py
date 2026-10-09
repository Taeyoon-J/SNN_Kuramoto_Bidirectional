"""Read-only, fail-closed summary for the registered SW0122 three-seed gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0122_joint_rgb_seed_replication import coordinator, run
from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117

HERE = Path(__file__).resolve().parent
SLOT_REFERENCE = HERE / "slot_reference.json"
EXPECTED_SLOT_REFERENCE_SHA256 = (
    "de49089f54c05e08574b654815a8fd4656cd0f306892f4ecb0c3da94ab579945")
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
ARMS = {"control": "control", "candidate": "analytic_candidate"}
DRAWS = 10_000
BOOTSTRAP_SEED = 122
EXPECTED_SLOT_IOU = {
    "foreground_iou": 0.20358913115224261,
    "matched_object_iou": 0.20693697915214362,
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _score(report: dict, *, seed: int, arm: str | None, expected_sha: str | None = None):
    if expected_sha and _sha(report["_path"]) != expected_sha:
        raise AssertionError(f"evaluation SHA mismatch for seed {seed} {arm or 'source'}")
    if report.get("ids") != [1320, 1639] or report.get("images") != 320:
        raise AssertionError(f"seed {seed} evaluation IDs/count mismatch")
    if report.get("ground_truth_used_for_prediction") is not False:
        raise AssertionError("GT was used for predictions")
    if arm is not None and (report.get("experiment") != "SW0122"
                            or report.get("seed") != seed or report.get("arm") != arm):
        raise AssertionError(f"seed {seed}/{arm} report identity mismatch")
    scored = report["sweep"][0]["scored_targets"]["our_hdf5"]
    if set(scored.get("metrics", {})) != set(METRICS):
        raise AssertionError("unexpected metric schema")
    if set(scored.get("valid_count", {})) != set(METRICS):
        raise AssertionError("unexpected valid-count schema")
    if set(scored.get("per_image", {})) != set(METRICS):
        raise AssertionError("unexpected per-image schema")
    per_image, means = {}, {}
    for metric in METRICS:
        vals = np.asarray(scored["per_image"][metric], dtype=np.float64)
        mean = float(scored["metrics"][metric])
        if (scored["valid_count"][metric] != 320 or vals.shape != (320,)
                or not np.isfinite(vals).all() or not math.isfinite(mean)
                or abs(float(vals.mean()) - mean) > 1e-12):
            raise AssertionError(f"invalid {metric} values for seed {seed}/{arm}")
        per_image[metric] = vals
        means[metric] = mean
    return {"metrics": means, "per_image": per_image, "evaluation_sha256": _sha(report["_path"])}


def _read_score(path: Path, **kwargs):
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["_path"] = path
    return _score(report, **kwargs)


def _assert_all_tasks_valid():
    tasks = coordinator.task_plan()
    if len(tasks) != 13 or len({task["task_id"] for task in tasks}) != 13:
        raise AssertionError("SW0122 registration no longer contains the expected 13 unique stages")
    invalid = [t["task_id"] for t in tasks if not coordinator.valid_result(t)]
    if invalid:
        raise RuntimeError("SW0122 results are not all terminal and valid: " + ", ".join(invalid))
    return {"registered_task_count": 13, "valid_task_ids": [t["task_id"] for t in tasks]}


def _training_artifacts(seed: int, arm: str):
    folder = coordinator.OUT / f"seed{seed}_{arm}"
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {name: folder / filename for name, filename in {
        "core": "core.pt", "encoder": "encoder.pt", "optimizer": "optimizer.pt",
        "history": "history.json"}.items()}
    actual = {name: _sha(path) for name, path in files.items()}
    for name, sha_name in (("core", "core_sha256"), ("encoder", "encoder_sha256"),
                           ("optimizer", "optimizer_sha256"), ("history", "history_sha256")):
        if manifest.get(sha_name) != actual[name]:
            raise AssertionError(f"SW0122 seed{seed}/{arm} {name} hash differs from training manifest")
    if (manifest.get("experiment") != "SW0122" or manifest.get("seed") != seed
            or manifest.get("arm") != arm or manifest.get("status") != "training_complete"
            or manifest.get("ground_truth_used_for_training") is not False
            or manifest.get("implementation_fingerprint") != run.implementation_fingerprint()):
        raise AssertionError(f"SW0122 seed{seed}/{arm} training manifest identity mismatch")
    history = json.loads(files["history"].read_text(encoding="utf-8"))
    if (len(history) != 256 or [r.get("update") for r in history] != list(range(1, 257))
            or any(not math.isfinite(float(r[key])) for r in history
                   for key in ("total", "old", "primary", "positive_actual_spike_product", "rgb",
                               "gradient_norm_preclip"))):
        raise AssertionError(f"SW0122 seed{seed}/{arm} history is incomplete/nonfinite")
    return {"manifest_sha256": _sha(manifest_path), "files": actual,
            "source_core_sha256": manifest.get("source_core_sha256"),
            "source_manifest_sha256": manifest.get("source_manifest_sha256"),
            "source_evaluation_sha256": manifest.get("source_evaluation_sha256"),
            "training_ids_sha256": manifest.get("training_ids_sha256"),
            "matched_shuffle_seed": manifest.get("matched_shuffle_seed"),
            "updates": len(history)}


def _verify_training_source_binding(artifact: dict, source_ref: dict):
    pairs = (("source_core_sha256", "source_core_sha256"),
             ("source_manifest_sha256", "source_manifest_sha256"),
             ("source_evaluation_sha256", "source_evaluation_sha256"),
             ("training_ids_sha256", "training_ids_sha256"))
    for train_key, source_key in pairs:
        if artifact.get(train_key) != source_ref.get(source_key):
            raise AssertionError(f"training manifest is not bound to frozen source {source_key}")


def _load_seed_scores():
    historical = json.loads(run.HISTORICAL_SUMMARY.read_text(encoding="utf-8"))
    if _sha(run.HISTORICAL_SUMMARY) != run.EXPECTED_HISTORICAL_SUMMARY_SHA256:
        raise AssertionError("historical SW0117 seed-0 summary SHA mismatch")
    historical_validated = run.validate_historical_seed0()

    scores = {"source": {}, "control": {}, "candidate": {}}
    seed0_base = base.source_paths(0)[0].parent / "evaluation.json"
    expected_source_sha = historical["arms"]["source"]["evaluation_sha256"]
    scores["source"][0] = _read_score(seed0_base, seed=0, arm=None,
                                      expected_sha=expected_source_sha)
    expected_arm_keys = {"control": "control", "candidate": "candidate"}
    for label, historical_key in expected_arm_keys.items():
        arm = ARMS[label]
        path = sw117.OUT / f"seed0_{arm}" / "evaluation.json"
        ref = historical["arms"][historical_key]
        value = _read_score(path, seed=0, arm=None, expected_sha=ref["evaluation_sha256"])
        for metric in METRICS:
            if abs(value["metrics"][metric] - float(ref["metrics"][metric])) > 1e-12:
                raise AssertionError(f"historical seed-0 {label}/{metric} disagrees with frozen summary")
        scores[label][0] = value

    source_reference_shas, trained_artifacts = {}, {}
    for seed in (1, 2):
        source_ref = run.validate_source_reference(seed)
        source_value = _read_score(source_ref["source_evaluation"], seed=seed, arm=None,
                                   expected_sha=source_ref["source_evaluation_sha256"])
        scores["source"][seed] = source_value
        source_reference_shas[str(seed)] = {
            "core_sha256": source_ref["source_core_sha256"],
            "manifest_sha256": source_ref["source_manifest_sha256"],
            "evaluation_sha256": source_value["evaluation_sha256"],
            "training_ids_sha256": source_ref["training_ids_sha256"],
        }
        for label, arm in ARMS.items():
            artifact = _training_artifacts(seed, arm)
            _verify_training_source_binding(artifact, source_ref)
            trained_artifacts[f"seed{seed}_{label}"] = artifact
            task = next(t for t in coordinator.task_plan()
                        if t["stage"] == "evaluate" and t["seed"] == seed and t["arm"] == arm)
            value = _read_score(coordinator.artifact_path(task), seed=seed, arm=arm)
            scores[label][seed] = value
    return scores, historical_validated, source_reference_shas, trained_artifacts


def _bootstrap_pair(left, right):
    """Same paired image resamples are shared across the three seed rows."""
    delta_by_seed = np.stack([left[s]["per_image"]["fg_ari"]
                              - right[s]["per_image"]["fg_ari"] for s in (0, 1, 2)])
    per_image_three_seed_mean = delta_by_seed.mean(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 320, size=(DRAWS, 320))
    draws = per_image_three_seed_mean[indices].mean(axis=1)
    return {"mean_delta": float(per_image_three_seed_mean.mean()),
            "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "draws": DRAWS, "seed": BOOTSTRAP_SEED,
            "unit": "paired image; each draw index is applied to all three seeds"}


def summarize():
    task_evidence = _assert_all_tasks_valid()
    scores, historical_evidence, source_artifacts, trained_artifacts = _load_seed_scores()
    means = {}
    for label in ("source", "control", "candidate"):
        means[label] = {metric: float(np.mean([scores[label][s]["metrics"][metric]
                                               for s in (0, 1, 2)]))
                        for metric in METRICS}
    if _sha(SLOT_REFERENCE) != EXPECTED_SLOT_REFERENCE_SHA256:
        raise AssertionError("bundled matched-Slot reference SHA mismatch")
    slot_doc = json.loads(SLOT_REFERENCE.read_text(encoding="utf-8"))
    slot = slot_doc["metrics"]
    if (slot_doc.get("experiment") != "SW0122"
            or slot_doc.get("source_experiment") != "SW0095_full70k_aligned_loss"
            or slot_doc.get("source_summary_sha256") != "85f5f36c9900e31841909d7d7e50855f5bb24a3f2d567e05846b67266923056e"
            or slot_doc.get("training_pool", {}).get("unique_images") != 70000
            or slot_doc.get("training_pool", {}).get("segments_inclusive") != [[0, 999], [1640, 70639]]
            or slot_doc.get("evaluation", {}).get("ids_inclusive") != [1320, 1639]
            or slot_doc.get("evaluation", {}).get("count") != 320):
        raise AssertionError("bundled matched-Slot protocol/source identity mismatch")
    if any(not math.isfinite(float(slot[m])) for m in METRICS):
        raise AssertionError("invalid registered matched-slot means")
    if any(float(slot[m]) != expected for m, expected in EXPECTED_SLOT_IOU.items()):
        raise AssertionError("matched Slot IoU baseline differs from the preregistered values")

    source_gains = [scores["candidate"][s]["metrics"]["fg_ari"]
                    - scores["source"][s]["metrics"]["fg_ari"] for s in (0, 1, 2)]
    ci_control = _bootstrap_pair(scores["candidate"], scores["control"])
    ci_source = _bootstrap_pair(scores["candidate"], scores["source"])
    gates = {
        "candidate_mean_fg_ari_exceeds_source": means["candidate"]["fg_ari"] > means["source"]["fg_ari"],
        "candidate_mean_fg_ari_exceeds_control": means["candidate"]["fg_ari"] > means["control"]["fg_ari"],
        "at_least_two_seed_fg_gains_vs_source": sum(gain > 0 for gain in source_gains) >= 2,
        "candidate_minus_control_ci_lower_positive": ci_control["ci95"][0] > 0,
        "candidate_minus_source_ci_lower_positive": ci_source["ci95"][0] > 0,
        "foreground_iou_exceeds_slot_plus_0_05": means["candidate"]["foreground_iou"] > float(slot["foreground_iou"]) + 0.05,
        "matched_object_iou_exceeds_slot_plus_0_05": means["candidate"]["matched_object_iou"] > float(slot["matched_object_iou"]) + 0.05,
    }
    return {
        "experiment": "SW0122", "status": "complete", "promotion_gate_passed": all(gates.values()),
        "task_validation": task_evidence,
        "historical_sw0117_seed0": historical_evidence,
        "source_artifacts_seed1_2": source_artifacts,
        "sw0122_training_artifacts": trained_artifacts,
        "evaluation_sha256": {label: {str(s): scores[label][s]["evaluation_sha256"]
                                      for s in (0, 1, 2)}
                              for label in ("source", "control", "candidate")},
        "per_seed_metrics": {label: {str(s): scores[label][s]["metrics"] for s in (0, 1, 2)}
                             for label in ("source", "control", "candidate")},
        "three_seed_means": means,
        "candidate_minus_source_fg_ari_by_seed": {str(s): source_gains[s] for s in (0, 1, 2)},
        "paired_bootstrap_candidate_minus_control_fg_ari": ci_control,
        "paired_bootstrap_candidate_minus_source_fg_ari": ci_source,
        "matched_slot_means": {m: float(slot[m]) for m in METRICS},
        "matched_slot_reference_sha256": _sha(SLOT_REFERENCE),
        "matched_slot_source_summary_sha256": slot_doc["source_summary_sha256"],
        "gate_checks": gates,
        "interpretation": "This fixed 256-update replication uses historical SW0117 seed 0 and newly trained seeds 1/2. It does not change SW0117's original promotion decision.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional new JSON output; existing files are never overwritten")
    args = parser.parse_args()
    result = summarize()
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"preserving existing summary: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"status": "complete", "promotion_gate_passed": result["promotion_gate_passed"],
                      "three_seed_means": result["three_seed_means"],
                      "gate_checks": result["gate_checks"],
                      "output": str(args.output) if args.output else None}, allow_nan=False))


if __name__ == "__main__":
    main()

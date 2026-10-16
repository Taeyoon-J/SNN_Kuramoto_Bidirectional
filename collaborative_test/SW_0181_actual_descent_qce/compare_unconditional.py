"""Supplemental paired comparison with the completed SW0178/179 QCE control."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

if "--worker" not in __import__("sys").argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (0, 1, 2)
ENDPOINTS = ("conditional", "standalone")
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
COUNT = 320
BOOTSTRAP_SEED = 178
BOOTSTRAP_REPLICATES = 10_000
SW0179_REPORT = ROOT / "collaborative_test" / "SW_0179_qce_seed_replication" \
    / "results_archive" / "evaluation_seedmean_20261016.json"
SW0179_REPORT_SHA256 = "758e48f83b44dc31efeae73154c13dbcd8ffa6132a85c7055279a0f4b2649c3d"
SW0181_SCORE = HERE / "results_archive" / "evaluation_seedmean_20261016.json"
OUTPUT = HERE / "results_archive" / "unconditional_qce_comparison_20261016.json"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _vectors(report, group):
    result = {}
    rows = report.get("scores", {}).get(group)
    if not isinstance(rows, dict): raise ValueError(f"missing metric group {group}")
    for metric in METRICS:
        row = rows.get(metric, {})
        per_seed = row.get("per_seed_per_image")
        if not isinstance(per_seed, dict) or set(per_seed) != {str(seed) for seed in SEEDS}:
            raise ValueError(f"{group}/{metric} lacks all three per-image seed vectors")
        matrix = np.stack([np.asarray(per_seed[str(seed)], dtype=np.float64) for seed in SEEDS])
        if matrix.shape != (3, COUNT) or not np.isfinite(matrix).all():
            raise ValueError(f"{group}/{metric} per-image vectors are malformed")
        mean_by_seed = row.get("per_seed_mean", {})
        if set(mean_by_seed) != {str(seed) for seed in SEEDS}:
            raise ValueError(f"{group}/{metric} per-seed means are incomplete")
        for idx, seed in enumerate(SEEDS):
            if not np.isclose(float(matrix[idx].mean()), float(mean_by_seed[str(seed)]),
                              rtol=0, atol=1e-15):
                raise ValueError(f"{group}/{metric} seed{seed} vector mean mismatch")
        result[metric] = matrix.mean(axis=0)
    return result


def _validate_report(report, *, expected_experiment, required_sha=None):
    if report.get("status") != "complete" or report.get("seed_scope") != list(SEEDS):
        raise ValueError("comparison report is not a completed three-seed fixed320 score")
    if required_sha is not None and required_sha != SW0179_REPORT_SHA256:
        raise ValueError("unconditional reference report is not the pinned SW0179 artifact")
    candidate_group = "candidate_standalone"
    for endpoint in ENDPOINTS:
        group = f"candidate_{endpoint}"
        _vectors(report, group)
    return True


def _paired(candidate, control, sample_indices):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    if delta.shape != (COUNT,) or not np.isfinite(delta).all():
        raise ValueError("paired comparison requires a finite shared fixed320 vector")
    ci = np.percentile(delta[sample_indices].mean(axis=1), [2.5, 97.5])
    return {"mean": float(delta.mean()), "ci95": [float(ci[0]), float(ci[1])],
        "bootstrap_seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "paired image after per-image mean across three seeds"}


def compare_reports(candidate_report, control_report, *, control_sha256):
    _validate_report(candidate_report, expected_experiment="SW0181_actual_descent_qce_three_seed_score")
    _validate_report(control_report, expected_experiment="SW0179_qce_seed_replication_three_seed_score",
                     required_sha=control_sha256)
    if candidate_report.get("experiment") != "SW0181_actual_descent_qce_three_seed_score":
        raise ValueError("candidate report experiment identity mismatch")
    if control_report.get("experiment") != "SW0179_qce_seed_replication_three_seed_score":
        raise ValueError("matched unconditional-control report identity mismatch")
    indices = np.random.RandomState(BOOTSTRAP_SEED).randint(0, COUNT,
        size=(BOOTSTRAP_REPLICATES, COUNT))
    results = {}
    for endpoint in ENDPOINTS:
        c = _vectors(candidate_report, f"candidate_{endpoint}")
        b = _vectors(control_report, f"candidate_{endpoint}")
        results[endpoint] = {metric: _paired(c[metric], b[metric], indices)
            | {"candidate_mean": float(c[metric].mean()), "control_mean": float(b[metric].mean())}
            for metric in METRICS}
    return {"experiment": "SW0181_vs_completed_unconditional_QCE_supplement",
        "status": "complete", "seed_scope": list(SEEDS), "endpoint_scope": list(ENDPOINTS),
        "metric_scope": list(METRICS), "candidate_experiment": candidate_report["experiment"],
        "control_experiment": control_report["experiment"], "paired_results": results,
        "bootstrap": {"rng": "numpy RandomState", "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES, "resampling_unit": "320 fixed images"},
        "validation_data_read": False}


def run(*, score_path=SW0181_SCORE, control_path=SW0179_REPORT, output=OUTPUT):
    score_path, control_path, output = Path(score_path), Path(control_path), Path(output)
    if output.exists(): raise FileExistsError("preserving existing unconditional comparison")
    if sha256_file(control_path) != SW0179_REPORT_SHA256:
        raise ValueError("pinned SW0179 unconditional QCE report changed")
    candidate = json.loads(score_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    candidate_hash, control_hash = sha256_file(score_path), sha256_file(control_path)
    report = compare_reports(candidate, control, control_sha256=control_hash)
    report["candidate_score_sha256"] = candidate_hash
    report["control_score_sha256"] = control_hash
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, output)
    finally:
        if temp.exists(): temp.unlink()
    return {**report, "output_sha256": sha256_file(output)}


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "output_sha256": result["output_sha256"]}))

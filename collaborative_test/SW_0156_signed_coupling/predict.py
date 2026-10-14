"""Fixed320 inference for the frozen SW0156 signed-coupling recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
PRED_ROOT = ARCHIVE / "predictions"
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0135_native32_spike_binding import foundation
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator
from collaborative_test.SW_0156_signed_coupling import screen, source_screen_queue
from collaborative_test.SW_0156_signed_coupling.signed_coupling import install_instance_signed_interaction

SEED = 1
ARMS = ("source_T", "signed_candidate", "matched_control")
STEPS, SETTLE = 1024, 512


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint():
    paths = (HERE / "protocol.json", HERE / "signed_coupling.py", HERE / "screen.py",
             HERE / "source_screen_queue.py", HERE / "predict.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
             ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
             ROOT / "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
             ROOT / "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0156 prediction dependency missing")
    result = dict(foundation.implementation_fingerprint())
    result.update(sw143.implementation_fingerprint())
    result.update({path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths})
    return result


def prediction_dir(arm, root=PRED_ROOT):
    if arm not in ARMS:
        raise ValueError(f"unregistered SW0156 prediction arm: {arm}")
    return Path(root) / arm / "seed1"


def _write_npz_once(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _write_json_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _ledger_header(arm, implementation, screen_record, screen_sha, t_reference):
    return {
        "experiment": "SW0156_native32_signed_coupling",
        "arm": arm, "seed": SEED, "implementation": implementation,
        "source_screen_sha256": screen_sha,
        "source_lineage": screen_record["source_lineage"],
        "source_reference_prediction_sha256": t_reference["prediction_sha256"],
        "source_reference_protocol_sha256": t_reference["protocol_sha256"],
        "expected_source_model_state_sha256": screen_record["source_model_state_before_sha256"],
    }


def _read_ledger(target, expected_header, ids):
    """Read committed whole-image rows; uncommitted/unknown files fail closed."""
    target = Path(target)
    header_path, ledger_path = target / "prediction_header.json", target / "prediction_ledger.jsonl"
    if not target.exists():
        target.mkdir(parents=True)
        _write_json_once(header_path, expected_header)
        with ledger_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"record_type": "header", "header": expected_header},
                                    sort_keys=True, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        (target / "rows").mkdir()
        return {}
    if not target.is_dir() or not header_path.is_file() or not ledger_path.is_file():
        raise ValueError("existing prediction output is not a recognized resumable ledger")
    if json.loads(header_path.read_text(encoding="utf-8")) != expected_header:
        raise ValueError("prediction resume header lineage/fingerprint mismatch")
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("prediction ledger lacks a committed header")
    first = json.loads(lines[0])
    if first != {"record_type": "header", "header": expected_header}:
        raise ValueError("prediction ledger header mismatch")
    rows = {}
    row_dir = target / "rows"
    if not row_dir.is_dir():
        raise ValueError("prediction ledger row directory missing")
    for expected_index, line in enumerate(lines[1:]):
        row = json.loads(line)
        if (row.get("record_type") != "image" or row.get("index") != expected_index
                or not 0 <= expected_index < len(ids)
                or row.get("image_id") != int(ids[expected_index])):
            raise ValueError("prediction ledger indices or image IDs are invalid")
        name = f"{expected_index:03d}.npz"
        path = row_dir / name
        if row.get("filename") != name or not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ValueError("prediction ledger row is missing or hash-mismatched")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {"image_id", "qcc_labels"}:
                raise ValueError("prediction ledger row archive schema mismatch")
            image_id = int(np.asarray(saved["image_id"]).item())
            labels = np.asarray(saved["qcc_labels"])
        if (image_id != int(ids[expected_index]) or labels.shape != (32, 32)
                or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
            raise ValueError("prediction ledger row values are invalid")
        rows[expected_index] = labels.astype(np.int64, copy=False)
    permitted = {"prediction_header.json", "prediction_ledger.jsonl", "rows",
                 "predictions.npz", "protocol.json"}
    if any(item.name not in permitted for item in target.iterdir()):
        raise ValueError("unregistered files in prediction output directory")
    known_rows = {f"{index:03d}.npz" for index in rows}
    if any(path.name not in known_rows for path in row_dir.iterdir()):
        raise ValueError("uncommitted prediction row file preserved; manual review required")
    return rows


def _commit_prediction_row(target, index, image_id, labels):
    row_dir = Path(target) / "rows"
    row_dir.mkdir(exist_ok=True)
    name = f"{index:03d}.npz"
    path = row_dir / name
    _write_npz_once(path, image_id=np.asarray(image_id, dtype=np.int64),
                    qcc_labels=np.asarray(labels, dtype=np.int64))
    entry = {"record_type": "image", "index": int(index), "image_id": int(image_id),
             "filename": name, "sha256": sha256_file(path)}
    ledger = Path(target) / "prediction_ledger.jsonl"
    with ledger.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _validated_screen():
    path = source_screen_queue.OUTPUT
    if not source_screen_queue.validate_report(path):
        raise ValueError("a current, validated GT-free SW0156 source screen is required")
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def validate_prediction(directory, *, arm, expected_fingerprint=None):
    directory = Path(directory)
    npz = directory / "predictions.npz"
    protocol_path = directory / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if expected_fingerprint is None:
        expected_fingerprint = fingerprint()
    expected_spec = {
        "source_T": {"source_arm": "T", "beta": 0.0, "base_scale": 1.0},
        "signed_candidate": {"source_arm": "T", "beta": 0.5, "base_scale": 1.0},
        "matched_control": {"source_arm": "T", "beta": 0.0, "base_scale": 0.5},
    }[arm]
    if (protocol.get("experiment") != "SW0156_native32_signed_coupling"
            or protocol.get("status") != "complete" or protocol.get("arm") != arm
            or protocol.get("seed") != SEED or protocol.get("steps") != STEPS
            or protocol.get("settle") != SETTLE or protocol.get("grid_size") != [32, 32]
            or protocol.get("count") != 320 or protocol.get("image_ids") != [1320, 1639]
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or protocol.get("implementation") != expected_fingerprint
            or protocol.get("arm_spec") != expected_spec
            or protocol.get("prediction_sha256") != sha256_file(npz)):
        raise ValueError(f"invalid SW0156 {arm} prediction protocol")
    screen_record, screen_sha = _validated_screen()
    t_reference = sw143.validate_arm_prediction(sw143._arm_dir("T", SEED), seed=SEED)
    if (protocol.get("source_screen_sha256") != screen_sha
            or protocol.get("source_lineage") != screen_record.get("source_lineage")
            or protocol.get("source_reference_prediction_sha256") != t_reference["prediction_sha256"]
            or protocol.get("source_reference_protocol_sha256") != t_reference["protocol_sha256"]
            or protocol.get("source_model_state_before_sha256") !=
                screen_record.get("source_model_state_before_sha256")
            or protocol.get("source_model_state_after_sha256") !=
                screen_record.get("source_model_state_before_sha256")
            or protocol.get("validation_rgb_assets") !=
                screen_record.get("source_lineage", {}).get("rgb_asset_validation")
            or protocol.get("committed_image_rows") != 320
            or protocol.get("per_image_resume_ledger") != (arm != "source_T")):
        raise ValueError("prediction is not bound to the validated source screen")
    with np.load(npz, allow_pickle=False) as archive:
        if set(archive.files) != {"image_ids", "qcc_labels"}:
            raise ValueError("SW0156 prediction NPZ schema mismatch")
        ids = contract.require_fixed_image_ids(archive["image_ids"])
        labels = np.asarray(archive["qcc_labels"])
    if labels.shape != (320, 32, 32) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("SW0156 labels violate direct native32 QCC contract")
    return {"ids": ids, "labels": labels.astype(np.int64, copy=False),
            "npz_path": npz, "protocol_path": protocol_path,
            "prediction_sha256": protocol["prediction_sha256"],
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "protocol": protocol}


@torch.no_grad()
def predict_arm(arm, *, device="cuda:0", output_root=PRED_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0156 arm")
    target = prediction_dir(arm, output_root)
    screen_record, screen_sha = _validated_screen()
    t = sw143.validate_arm_prediction(sw143._arm_dir("T", SEED), seed=SEED)
    if t["protocol"].get("source") != screen_record.get("source_lineage"):
        raise ValueError("SW0143-T and source-screen source lineage differ")
    ids = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
    if not np.array_equal(t["ids"], ids):
        raise ValueError("SW0143-T fixed320 IDs changed")
    spec = {
        "source_T": {"source_arm": "T", "beta": 0.0, "base_scale": 1.0},
        "signed_candidate": {"source_arm": "T", "beta": 0.5, "base_scale": 1.0},
        "matched_control": {"source_arm": "T", "beta": 0.0, "base_scale": 0.5},
    }[arm]
    if (target / "protocol.json").is_file():
        existing = validate_prediction(target, arm=arm)
        return {"status": "prediction_complete", "arm": arm,
                "prediction_sha256": existing["prediction_sha256"], "reused": True}
    if arm == "source_T":
        if target.exists():
            raise FileExistsError("preserve partial source-reference copy; no automatic rewrite")
        labels = t["labels"].copy()
        state_before = state_after = screen_record["source_model_state_before_sha256"]
        reused_source = True
    else:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SW0156 fixed320 inference requires CUDA")
        torch.cuda.set_device(dev)
        torch.cuda.init()
        found = foundation.load_native32_foundation(SEED, dev, verify_rgb_assets=True)
        found.encoder.eval(); found.wrapped.eval()
        source_lineage = sw143._native_source_record(found)
        if source_lineage != screen_record["source_lineage"]:
            raise ValueError("current native32 source differs from validated screen")
        if found.wrapped.core.theta_init != "gamma" or found.wrapped.core.theta_init_noise != 0:
            raise ValueError("registered fixed inference requires deterministic gamma theta initialization")
        graph = found.wrapped.core.graph_generator
        geodesic_operator.install_operator(graph, "T")
        layer = found.wrapped.core.kuramoto
        before = screen._digest_state(found.encoder, found.patcher, found.wrapped)
        if before != screen_record.get("source_model_state_before_sha256"):
            raise ValueError("current model weights differ from the validated source-screen state")
        restore = install_instance_signed_interaction(
            graph, layer, beta=spec["beta"], base_scale=spec["base_scale"])
        cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
        if cache.dtype != np.uint8 or cache.shape != (320, 128, 128, 3):
            restore()
            raise ValueError("fixed320 validation RGB cache schema mismatch")
        implementation = fingerprint()
        expected_header = _ledger_header(arm, implementation, screen_record,
                                         screen_sha, t)
        committed = _read_ledger(target, expected_header, ids)
        labels = np.empty((320, 32, 32), dtype=np.int64)
        try:
            for index in range(320):
                if index in committed:
                    labels[index] = committed[index]
                    continue
                rgb = torch.from_numpy(np.asarray(cache[index:index + 1]).copy()).to(dev)
                gamma = found.encode(rgb)
                _unused_primary, qcc = sw136._predict_one(found, None, gamma, primary=False)
                qcc = np.asarray(qcc)
                if (qcc.shape != (1, 32, 32) or not np.issubdtype(qcc.dtype, np.integer)
                        or np.any(qcc < 0)):
                    raise ValueError(f"invalid native32 QCC labels at validation row {index}")
                labels[index] = qcc[0]
                _commit_prediction_row(target, index, int(ids[index]), labels[index])
                if index % 40 == 39:
                    print(json.dumps({"stage": "predict", "arm": arm,
                                      "predicted": index + 1, "count": 320}), flush=True)
        finally:
            restore()
        after = screen._digest_state(found.encoder, found.patcher, found.wrapped)
        if before != after or after != screen_record.get("source_model_state_before_sha256"):
            raise AssertionError("source model changed during SW0156 inference")
        state_before, state_after = before, after
        reused_source = False
    if not np.array_equal(ids, t["ids"]):
        raise ValueError("SW0156 prediction IDs changed")
    implementation = fingerprint()
    npz_path = target / "predictions.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as existing:
            if (set(existing.files) != {"image_ids", "qcc_labels"}
                    or not np.array_equal(existing["image_ids"], ids)
                    or not np.array_equal(existing["qcc_labels"], labels)):
                raise ValueError("existing assembled prediction differs from committed image ledger")
    else:
        _write_npz_once(npz_path, image_ids=ids, qcc_labels=labels)
    protocol = {
        "experiment": "SW0156_native32_signed_coupling", "status": "complete",
        "seed": SEED, "arm": arm, "arm_spec": spec,
        "image_ids": [1320, 1639], "count": 320, "grid_size": [32, 32],
        "prediction_shape": [320, 32, 32], "steps": STEPS, "settle": SETTLE,
        "readout": "unchanged SW0136 production actual-spike native32 QCC",
        "implementation": implementation, "source_lineage": screen_record["source_lineage"],
        "source_screen_sha256": screen_sha,
        "source_reference_prediction_sha256": t["prediction_sha256"],
        "source_reference_protocol_sha256": t["protocol_sha256"],
        "validation_rgb_assets": screen_record["source_lineage"]["rgb_asset_validation"],
        "committed_image_rows": 320,
        "per_image_resume_ledger": arm != "source_T",
        "source_model_state_before_sha256": state_before,
        "source_model_state_after_sha256": state_after,
        "source_reference_reused": reused_source,
        "prediction_sha256": sha256_file(target / "predictions.npz"),
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
    }
    _write_json_once(target / "protocol.json", protocol)
    return {"status": "prediction_complete", "arm": arm,
            "prediction_sha256": protocol["prediction_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=PRED_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(predict_arm(args.arm, device=args.device,
                                 output_root=args.output_root), allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

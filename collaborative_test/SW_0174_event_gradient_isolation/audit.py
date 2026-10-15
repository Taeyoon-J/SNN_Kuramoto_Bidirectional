"""SW0174 scoped event-gradient isolation around the frozen SW0173 audit."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np  # Load NumPy/OpenMP before torch on Windows.
import torch

from collaborative_test.SW_0173_gradient_fidelity import audit as native_audit
from collaborative_test.SW_0173_gradient_fidelity import audit_queue as native_queue
from collaborative_test.SW_0173_gradient_fidelity import gradient_fidelity as gf
from collaborative_test.SW_0173_gradient_fidelity import resource_probe as probe
from collaborative_test.SW_0174_event_gradient_isolation import event_gradient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "results_archive" / "event_gradient_audit_20261015" / "report.json"
REFERENCE_SHA256 = "d3163bf1ef8bf963ed18d72ccfdbc4dbac1655b47348bf1afb4115c89da01657"
REFERENCE_RAW = native_queue.RUNTIME / "report.json"
REFERENCE_RECORDS = (native_audit.HERE / "results_archive" / "completed_audit_20261015"
                     / "server_records.json")
SCALES = (1.0 / 16.0, 1.0)


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint():
    files = {
        "collaborative_test/SW_0174_event_gradient_isolation/DESIGN_KO.md": HERE / "DESIGN_KO.md",
        "collaborative_test/SW_0174_event_gradient_isolation/event_gradient.py": HERE / "event_gradient.py",
        "collaborative_test/SW_0174_event_gradient_isolation/audit.py": HERE / "audit.py",
        "collaborative_test/SW_0174_event_gradient_isolation/queue.py": HERE / "queue.py",
        "collaborative_test/SW_0174_event_gradient_isolation/test_event_gradient.py": HERE / "test_event_gradient.py",
        "collaborative_test/SW_0173_gradient_fidelity/results_archive/completed_audit_20261015/server_records.json": REFERENCE_RECORDS,
    }
    hashes = {}
    for rel, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"SW0174 fingerprint dependency missing: {rel}")
        hashes[rel] = _sha_file(path)
    return {"experiment": "SW0174_event_gradient_isolation",
            "files": hashes, "sw0173_fingerprint": native_audit.fingerprint(),
            "sw0173_reference_report_sha256": REFERENCE_SHA256,
            "optimizer_updates": 0, "validation_data_read": False,
            "reserve_data_read": False}


def _validate_reference_payload(report, records, expected_fp, source_proof):
    state = records.get("state", {})
    if (state.get("status") != "complete_gradient_audit" or state.get("returncode") != 0
            or state.get("report_sha256") != REFERENCE_SHA256
            or records.get("report_file_sha256") != REFERENCE_SHA256):
        raise ValueError("frozen SW0173 comparator receipt is not the registered completed report")
    if records.get("report") != report:
        raise ValueError("SW0173 report bytes differ from the preserved supervisor record")
    if (report.get("status") != "complete_gradient_audit"
            or report.get("optimizer_updates") != 0
            or report.get("validation_data_read") is not False
            or report.get("reserve_data_read") is not False
            or report.get("model_state_restored") is not True
            or report.get("source_model_unchanged") is not True):
        raise ValueError("SW0173 reference is not a valid no-update native comparator")
    if report.get("implementation_fingerprint") != expected_fp:
        raise ValueError("frozen SW0173 comparator fingerprint differs from current sealed sources")
    if report.get("source_proof") != source_proof:
        raise ValueError("frozen SW0173 comparator used different admitted source/input bindings")
    if report.get("same_rng_parity_rows") is None or report.get("source_baseline") is None:
        raise ValueError("frozen SW0173 comparator lacks the required fixed TRAIN baseline rows")
    return report


def _load_reference(prepared):
    raw = REFERENCE_RAW.read_bytes()
    digest = _sha_bytes(raw)
    if digest != REFERENCE_SHA256:
        raise ValueError("preserved SW0173 report does not match its registered SHA256")
    report = json.loads(raw.decode("utf-8"))
    records = json.loads(REFERENCE_RECORDS.read_text(encoding="utf-8"))
    return _validate_reference_payload(report, records, native_audit.fingerprint(),
                                       prepared["source_proof"]), digest


def prepare_cpu():
    prepared = native_audit.prepare_cpu()
    if prepared.get("cuda_initialized") is not False or prepared.get("validation_data_read") is not False:
        raise ValueError("SW0174 CPU preparation exceeded frozen SW0173 admission scope")
    reference, reference_sha = _load_reference(prepared)
    prepared["sw0173_reference_report_sha256"] = reference_sha
    return prepared


def _rng_equal(left, right):
    if not torch.equal(left["cpu"], right["cpu"]):
        return False
    if (left["cuda"] is None) != (right["cuda"] is None):
        return False
    if left["cuda"] is not None:
        return len(left["cuda"]) == len(right["cuda"]) and all(
            torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))
    return True


def _paired_forward(forward, scope, args, trace_fields):
    """Run original then stopped on identical RNG; return stopped graph result."""
    before = gf._rng_snapshot()
    scope.install_original()
    try:
        native = forward(*args)
        native_snapshot = event_gradient.snapshot_forward(native, trace_fields)
        native_after = gf._rng_snapshot()
    finally:
        scope.install_intervention()
    del native
    gf._rng_restore(before)
    isolated = forward(*args)
    isolated_snapshot = event_gradient.snapshot_forward(isolated, trace_fields)
    isolated_after = gf._rng_snapshot()
    if not _rng_equal(native_after, isolated_after):
        raise ValueError("native and stopped-event paired forwards consumed different RNG")
    checks = event_gradient.compare_forward_snapshots(native_snapshot, isolated_snapshot)
    if not checks["all_forward_values_exact"]:
        raise ValueError("event-gradient intervention changed same-RNG forward values")
    return isolated, native_snapshot, isolated_snapshot, checks


def _run_scoped_intervention(rollout, forward_owner, run):
    """Run a callback under the event hook and restore aliases on every exit."""
    original_event_alias = rollout.act_fun_adp
    original_forward = forward_owner._forward
    scope = None
    try:
        with event_gradient.detach_event_outputs(rollout) as scope:
            result = run(scope)
        event_calls = scope.calls
    finally:
        forward_owner._forward = original_forward
        if rollout.act_fun_adp is not original_event_alias:
            rollout.act_fun_adp = original_event_alias
        if (rollout.act_fun_adp is not original_event_alias
                or forward_owner._forward is not original_forward):
            raise RuntimeError("event intervention failed to restore original callable aliases")
    return result, event_calls


def _readout_signature(value, arrays, position, gain, solver, contract, sw136):
    _gamma, trace, adjacency, _projection, q = value
    labels, metrics, merges, qcc = native_audit._readouts(
        q[0], adjacency[0], trace, arrays, position, gain, solver, contract, sw136)
    return {"qcc_labels_sha256": probe._tensor_sha(torch.from_numpy(qcc)),
            "readout_labels_sha256": {name: probe._tensor_sha(torch.from_numpy(item))
                                      for name, item in labels.items()},
            "metrics": metrics, "merge_counts": merges}


def _assert_native_matches_reference(reference, position, native_snapshot, signature,
                                     audit_module, arrays, gain, solver, contract, sw136,
                                     native_value):
    parity = {int(row["position"]): row for row in reference["same_rng_parity_rows"]}
    if position not in parity:
        raise ValueError("frozen SW0173 comparator lacks a paired trace row")
    hashes = {name: probe._tensor_sha(value) for name, value in native_snapshot["trace"].items()}
    if hashes != parity[position]["full_trace_sha256"]:
        raise ValueError("native intervention baseline trace differs from pinned SW0173 report")
    baseline = {int(row["position"]): row for arm in ("cohort", "transfer")
                for row in reference["source_baseline"][arm]}[position]
    if baseline["image_id"] != int(arrays["image_ids"][position]):
        raise ValueError("SW0173 comparator image binding changed")
    actual = _readout_signature(native_value, arrays, position, gain, solver, contract, sw136)
    for key in ("qcc_labels_sha256", "readout_labels_sha256", "metrics", "merge_counts"):
        if actual[key] != baseline[key]:
            raise ValueError(f"native source readout {key} differs from pinned SW0173 baseline")
    return {"native_trace_hashes_match_frozen_SW0173": True,
            "native_source_readouts_match_frozen_SW0173": True,
            "source_readout_signature": actual}


def _primary_decision(native_report, reference):
    native_primary = next(row for row in native_report["trials"]["full1024"]["transfer"]
                          if row["scale"] == 1.0)
    baseline_native = {"cohort": native_report["source_baseline"]["cohort"],
                       "transfer": native_report["source_baseline"]["transfer"]}
    route_checks = {}
    for population in ("cohort", "transfer"):
        base = float(sum(row["native_loss"] for row in baseline_native[population])
                     / len(baseline_native[population]))
        route_checks[population] = {"source_mean": base,
            "stopped_primary_mean": next(row for row in native_report["trials"]["full1024"][population]
                                          if row["scale"] == 1.0)["trial_native_mean_loss"],
            "decreased": next(row for row in native_report["trials"]["full1024"][population]
                              if row["scale"] == 1.0)["actual_native_change"] < 0}

    old_source_rows = reference["source_baseline"]["transfer"]
    old_full = next(row for row in reference["trials"]["full1024"]["transfer"]
                    if row["scale"] == 1.0)
    candidate_rows = native_primary["images"]
    old_full_rows = old_full["images"]
    endpoint_checks = {}
    for endpoint in ("source_F", "native_QCC"):
        source_mean = sum(row["metrics"][endpoint]["fg_ari"] for row in old_source_rows) / len(old_source_rows)
        old_primary_mean = sum(row["metrics"][endpoint]["fg_ari"] for row in old_full_rows) / len(old_full_rows)
        stopped_mean = sum(row["metrics"][endpoint]["fg_ari"] for row in candidate_rows) / len(candidate_rows)
        endpoint_checks[endpoint] = {
            "source_mean_fg_ari": source_mean,
            "frozen_SW0173_full1024_primary_mean_fg_ari": old_primary_mean,
            "stopped_primary_mean_fg_ari": stopped_mean,
            "at_least_source": stopped_mean >= source_mean,
            "at_least_frozen_native_full_primary": stopped_mean >= old_primary_mean,
        }
    passed = (all(row["decreased"] for row in route_checks.values())
              and all(item["at_least_source"] and item["at_least_frozen_native_full_primary"]
                      for item in endpoint_checks.values()))
    return {"primary_scale": 1.0, "diagnostic_scale": 1.0 / 16.0,
            "route_loss_checks": route_checks, "transfer_foreground_checks": endpoint_checks,
            "registered_primary_pass": bool(passed)}


def run_audit(*, prepared=None, device="cuda:0", output=OUTPUT):
    prepared = prepare_cpu() if prepared is None else prepared
    reference, reference_sha = _load_reference(prepared)
    if prepared.get("sw0173_reference_report_sha256") != reference_sha:
        raise ValueError("SW0173 comparator changed after CPU admission")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0174 worker must preserve frozen CPU-before-CUDA admission")

    rollout = probe._import("collaborative_test.SW_0134_native_spike_binding.rollout")
    base_forward = native_audit._forward
    direct_rows = []
    paired_positions = set()
    cache = prepared["admitted"]["prepared"]
    arrays = cache["arrays"]
    gain = float(cache["producer_bindings"]["source_gain"])
    solver = probe._import("collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed")
    contract = probe._import("collaborative_test.SW_0135_native32_spike_binding.evaluation_contract32")
    sw136 = probe._import("collaborative_test.SW_0136_native32_transfer.evaluate")

    def paired_forward(train, found, row_arrays, position, device, horizon, scope):
        if horizon != "full1024" or position != 0 or position in paired_positions:
            return base_forward(train, found, row_arrays, position, device, horizon)
        isolated, native_snapshot, isolated_snapshot, checks = _paired_forward(
            base_forward, scope, (train, found, row_arrays, position, device, horizon),
            native_audit.TRACE_FIELDS)
        if row_arrays is not arrays:
            raise ValueError("paired forward received a different admitted input tensor set")
        native_sig = _readout_signature_from_snapshot(native_snapshot, row_arrays, position,
                                                       gain, solver, contract, sw136)
        isolated_sig = _readout_signature_from_snapshot(isolated_snapshot, row_arrays, position,
                                                          gain, solver, contract, sw136)
        for key in ("qcc_labels_sha256", "readout_labels_sha256", "metrics", "merge_counts"):
            if native_sig[key] != isolated_sig[key]:
                raise ValueError(f"event stop changed same-RNG source readout {key}")
        frozen_check = _assert_native_snapshot_matches_reference(
            reference, position, native_snapshot, native_sig, row_arrays)
        direct_rows.append({"position": int(position), "image_id": int(row_arrays["image_ids"][position]),
                            "native_vs_stopped": checks, "readouts_exact": True,
                            "frozen_SW0173_comparator": frozen_check})
        paired_positions.add(position)
        return isolated

    # Capture the comparator's source checks while the event alias is scoped;
    # the helper restores both aliases even if measurement fails.
    temp_root = ROOT / "collaborative_testing" / "temporary"
    temp_root.mkdir(parents=True, exist_ok=True)

    def execute(scope):
        def scoped_forward(train, found, row_arrays, position, device, horizon):
            # The callback needs the active scope supplied by the wrapper.
            return paired_forward(train, found, row_arrays, position, device, horizon, scope)

        native_audit._forward = scoped_forward
        with tempfile.TemporaryDirectory(prefix="sw0174_raw_", dir=temp_root) as scratch:
            raw_output = Path(scratch) / "sw0173_intervened_report.json"
            return native_audit.run_audit(prepared=prepared, device=device, output=raw_output)

    raw_report, event_calls = _run_scoped_intervention(rollout, native_audit, execute)

    if raw_report is None or raw_report.get("status") != "complete_gradient_audit":
        raise ValueError("frozen SW0173 comparator did not complete under event intervention")
    if 0 not in paired_positions:
        raise ValueError("same-RNG native/stopped full forward was not captured")
    if raw_report.get("source_proof") != prepared["source_proof"]:
        raise ValueError("intervened comparator source proof changed")
    # The old native report is the immutable comparator for all eight source
    # rows. Its all-six trace hashes, route values, and readouts must still bind.
    old_rows = {int(row["position"]): row for row in reference["same_rng_parity_rows"]}
    stopped_rows = {int(row["position"]): row for row in raw_report["same_rng_parity_rows"]}
    old_base = {int(row["position"]): row for name in ("cohort", "transfer")
                for row in reference["source_baseline"][name]}
    new_base = {int(row["position"]): row for name in ("cohort", "transfer")
                for row in raw_report["source_baseline"][name]}
    row_matches = []
    for position in (*native_audit.COHORT, *native_audit.TRANSFER):
        if (old_rows[position]["full_trace_sha256"] != stopped_rows[position]["full_trace_sha256"]
                or old_base[position]["image_id"] != new_base[position]["image_id"]
                or old_base[position]["qcc_labels_sha256"] != new_base[position]["qcc_labels_sha256"]
                or old_base[position]["readout_labels_sha256"] != new_base[position]["readout_labels_sha256"]
                or old_base[position]["metrics"] != new_base[position]["metrics"]
                or old_base[position]["merge_counts"] != new_base[position]["merge_counts"]
                or old_base[position]["native_loss"] != new_base[position]["native_loss"]
                or old_base[position]["fixed_route_loss"] != new_base[position]["fixed_route_loss"]):
            raise ValueError(f"intervened source baseline changed frozen SW0173 row {position}")
        row_matches.append(int(position))

    decision = _primary_decision(raw_report, reference)
    implementation = fingerprint()
    report = {
        "experiment": implementation["experiment"],
        "status": ("complete_event_gradient_audit" if decision["registered_primary_pass"]
                   else "event_gradient_audit_rejected"),
        "implementation_fingerprint": implementation,
        "sw0173_native_comparator_report_sha256": reference_sha,
        "sw0173_native_comparator_fingerprint": reference["implementation_fingerprint"],
        "source_proof": prepared["source_proof"],
        "intervention": {"module": "collaborative_test.SW_0134_native_spike_binding.rollout",
            "alias": "act_fun_adp", "operation": "original(input).detach()",
            "detaches_each_raw_event_output": True,
            "membrane_update_and_reset_equations_unchanged": True,
            "final_S_trace_not_replaced": True,
            "scoped_alias_restored": True,
            "detached_event_calls": int(event_calls)},
        "native_vs_stopped_same_rng_forward": direct_rows,
        "frozen_SW0173_source_rows_exact": row_matches,
        "frozen_SW0173_comparator_report_status": reference["status"],
        "frozen_SW0173_comparator_source_model_digest_sha256": reference["source_model_digest_before_sha256"],
        "stopped_event_gradient_audit": raw_report,
        "registered_primary_decision": decision,
        "optimizer_updates": 0, "validation_data_read": False,
        "reserve_data_read": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    train = probe._import(probe.TRAIN)
    train.preflight._write_report(Path(output), report)
    return report


def _readout_signature_from_snapshot(snapshot, arrays, position, gain, solver, contract, sw136):
    trace = snapshot["trace"]
    labels, metrics, merges, qcc = native_audit._readouts(
        snapshot["q"][0], snapshot["adjacency"][0], trace, arrays, position,
        gain, solver, contract, sw136)
    return {"qcc_labels_sha256": probe._tensor_sha(torch.from_numpy(qcc)),
            "readout_labels_sha256": {name: probe._tensor_sha(torch.from_numpy(value))
                                      for name, value in labels.items()},
            "metrics": metrics, "merge_counts": merges}


def _assert_native_snapshot_matches_reference(reference, position, snapshot, signature, arrays):
    parity = {int(row["position"]): row for row in reference["same_rng_parity_rows"]}[position]
    hashes = {name: probe._tensor_sha(value) for name, value in snapshot["trace"].items()}
    if hashes != parity["full_trace_sha256"]:
        raise ValueError("fresh native forward trace differs from frozen SW0173 comparator")
    base = {int(row["position"]): row for population in ("cohort", "transfer")
            for row in reference["source_baseline"][population]}[position]
    if base["image_id"] != int(arrays["image_ids"][position]):
        raise ValueError("frozen SW0173 comparator image ID differs from fresh source row")
    for key in ("qcc_labels_sha256", "readout_labels_sha256", "metrics", "merge_counts"):
        if signature[key] != base[key]:
            raise ValueError(f"fresh native readout differs from SW0173 baseline field {key}")
    return {"trace_exact": True, "source_readouts_exact": True}

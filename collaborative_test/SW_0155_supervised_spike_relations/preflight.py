"""Actual-source, TRAIN-mask-only two-update admission preflight for SW0155."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0150_spike_spatial_mixture.integration import forward_assignment
from collaborative_test.SW_0150_spike_spatial_mixture import preflight as sw150
from collaborative_test.SW_0155_supervised_spike_relations import objective, train

EXPERIMENT = train.EXPERIMENT
OUTPUT = train.HERE / "results_archive" / "preflight_seed1.json"
PREFLIGHT_UPDATES = 2


def validate_report(report, *, expected_fingerprint=None):
    if expected_fingerprint is None:
        expected_fingerprint = train.implementation_fingerprint()
    try:
        if (report.get("status") != "source_preflight_complete"
                or report.get("experiment") != EXPERIMENT
                or report.get("seed") != train.SEED
                or report.get("training_admission") is not True
                or report.get("ground_truth_used") is not True
                or report.get("ground_truth_scope") != "TRAIN masks only"
                or report.get("implementation_fingerprint") != expected_fingerprint
                or report.get("registered", {}).get("updates") != PREFLIGHT_UPDATES
                or report.get("registered", {}).get("logical_batch") != train.BATCH
                or report.get("registered", {}).get("microbatch") != train.MICRO
                or report.get("registered", {}).get("steps") != train.STEPS
                or report.get("registered", {}).get("settle") != train.SETTLE
                or report.get("registered", {}).get("live_tail") != train.TAIL
                or set(report.get("arms", {})) != set(train.ARMS)):
            return False
        for arm in train.ARMS:
            record = report["arms"][arm]
            names = {"spatial_residual", "graph"} | ({"native_core", "integration_tensors"} if arm == "joint" else set())
            if (record.get("status") != "two_actual_updates_complete"
                    or record.get("source_zero_adapter_parity_exact") is not True
                    or record.get("selected_families") != sorted(names)
                    or set(record.get("family_sha256_before", {})) != names
                    or set(record.get("family_sha256_after", {})) != names
                    or any(record["family_sha256_before"][key] == record["family_sha256_after"][key]
                           for key in names)
                    or len(record.get("updates", [])) != PREFLIGHT_UPDATES
                    or record.get("frozen_base_encoder_sha256_before_after", {}).get("before")
                       != record.get("frozen_base_encoder_sha256_before_after", {}).get("after")):
                return False
            for update in record["updates"]:
                if (not isinstance(update.get("mean_loss"), (int, float))
                        or not np.isfinite(update["mean_loss"])
                        or set(update.get("family_gradient_norms", {})) != names
                        or any(not np.isfinite(value) or value <= 0
                               for value in update["family_gradient_norms"].values())):
                    return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def validate_admission(report, *, source, assets, hdf5_path, hdf5_size, hdf5_mtime_ns):
    """Bind a passing preflight to current source, RGB assets, and canonical HDF5."""
    if not validate_report(report, expected_fingerprint=train.implementation_fingerprint()):
        return False
    saved = report.get("source", {})
    data = report.get("canonical_hdf5", {})
    expected_ids = source["image_ids"]
    return (
        saved.get("checkpoint_sha256") == source["source_sha256"]
        and saved.get("manifest_sha256") == train.sha256_file(source["manifest_path"])
        and saved.get("training_ids") == expected_ids
        and saved.get("training_ids_sha256") == train.ids_sha(expected_ids)
        and saved.get("selected_train_ids") == expected_ids[:train.BATCH * PREFLIGHT_UPDATES]
        and saved.get("selected_train_ids_sha256")
            == train.ids_sha(expected_ids[:train.BATCH * PREFLIGHT_UPDATES])
        and saved.get("rgb_asset_validation") == assets
        and data.get("path") == str(Path(hdf5_path).resolve())
        and data.get("size") == hdf5_size
        and data.get("mtime_ns") == hdf5_mtime_ns
        and data.get("mask_shape") in ([100000, 128, 128], [100000, 128, 128, 1])
    )


def _atomic_create_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _move_trace_to_cpu(result):
    values = {key: result[key].detach().cpu() for key in ("q", "qcc", "gamma")}
    for key in ("theta", "membrane", "spikes", "component_membrane",
                "component_spikes", "component_gates"):
        values[key] = result["trace"][key].detach().cpu()
    return values


def canonical_hdf5_record(path, handle):
    """Capture schema/stat metadata while HDF5 datasets are still open."""
    stat = Path(path).stat()
    image_shape = list(handle["image"].shape)
    mask_shape = list(handle["mask"].shape)
    if image_shape != [100000, 128, 128, 3] or mask_shape not in (
            [100000, 128, 128], [100000, 128, 128, 1]):
        raise ValueError("canonical supervised HDF5 schema changed")
    return {"path": str(Path(path).resolve()), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "image_shape": image_shape, "mask_shape": mask_shape}


def run_preflight(*, device="cuda:0", output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserving existing SW0155 preflight output")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual-source preflight requires assigned CUDA")
    torch.cuda.set_device(dev)
    source = train.source_contract()
    assets = sw130.validate_rgb_assets()
    saved_parity, parity_sha = sw150.load_and_validate_parity(
        sw150.PARITY_PATH, source_sha=source["source_sha256"],
        source_manifest_sha=train.sha256_file(source["manifest_path"]),
        training_ids=source["image_ids"], asset_validation=assets)
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered TRAIN RGB cache schema changed")
    train_ids = source["image_ids"][:train.BATCH * PREFLIGHT_UPDATES]
    rows = source["rows"][:train.BATCH * PREFLIGHT_UPDATES]
    objective.validate_training_prefix(source["image_ids"][:2048], source["image_ids"])
    arms = {}
    with __import__("h5py").File(train.HDF5_PATH, "r") as hdf5:
        hdf5_metadata = canonical_hdf5_record(train.HDF5_PATH, hdf5)
        for arm in train.ARMS:
            foundation = train._load_foundation(dev)
            if (foundation.provenance.get("source_core_sha256") != source["source_sha256"]
                    or foundation.provenance.get("source_manifest_sha256")
                       != train.sha256_file(source["manifest_path"])
                    or foundation.provenance.get("rgb_asset_validation") != assets
                    or list(map(int, foundation.image_ids)) != source["image_ids"]
                    or not np.array_equal(foundation.pool_indices, source["rows"])):
                raise ValueError("fresh source foundation differs from registered source contract")
            rgb0, mask0 = train.load_aligned_train_rows(
                train.HDF5_PATH, train_ids, np.asarray(cache[rows]).copy(), 0,
                train.MICRO, hdf5_handle=hdf5)
            rgb0 = rgb0.to(dev)
            source_result = forward_assignment(foundation, rgb0, live_tail_steps=train.TAIL)
            source_reference = _move_trace_to_cpu(source_result)
            del source_result
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(train.RESIDUAL_INIT_SEED)
                foundation.encoder = train.SourcePreservingDeepEncoder(
                    foundation.encoder, freeze_base=True).to(dev)
            student_result = forward_assignment(foundation, rgb0, live_tail_steps=train.TAIL)
            exact = all(torch.equal(source_reference[key], student_result[key].detach().cpu())
                        for key in ("q", "qcc", "gamma"))
            exact = exact and all(torch.equal(source_reference[key],
                student_result["trace"][key].detach().cpu()) for key in (
                    "theta", "membrane", "spikes", "component_membrane",
                    "component_spikes", "component_gates"))
            del student_result, source_reference, rgb0, mask0
            if not exact:
                raise AssertionError("SW0155 zero-residual source parity failed")
            families = train.configure_trainability(foundation, arm)
            optimizer = train.make_optimizer(families)
            before = {name: train.source_training.parameter_hash(params)
                      for name, params in families.items()}
            base_before = train.source_training.state_sha(foundation.encoder.base_encoder)
            update_rows = []
            for update in range(PREFLIGHT_UPDATES):
                def compute_micro(micro):
                    start = update * train.BATCH + micro * train.MICRO
                    rgb_cpu, masks_cpu = train.load_aligned_train_rows(
                        train.HDF5_PATH, train_ids, np.asarray(cache[rows]).copy(),
                        start, train.MICRO, hdf5_handle=hdf5)
                    labels = objective.modal_train_targets32(masks_cpu).to(dev)
                    result = foundation.rollout(foundation.encode(rgb_cpu.to(dev)),
                        total_steps=train.STEPS, live_tail_steps=train.TAIL)
                    loss, _ = objective.supervised_relation_loss(
                        result, labels, settle=train.SETTLE)
                    return loss
                update_rows.append(train.logical_update(optimizer=optimizer,
                    family_map=families, compute_micro=compute_micro,
                    wrapped=foundation.wrapped))
            after = {name: train.source_training.parameter_hash(params)
                     for name, params in families.items()}
            base_after = train.source_training.state_sha(foundation.encoder.base_encoder)
            arms[arm] = {"status": "two_actual_updates_complete",
                "source_zero_adapter_parity_exact": True,
                "selected_families": sorted(families),
                "family_sha256_before": before, "family_sha256_after": after,
                "updates": update_rows,
                "frozen_base_encoder_sha256_before_after": {"before": base_before, "after": base_after}}
            if (base_before != base_after or any(before[key] == after[key] for key in families)
                    or any(value["family_gradient_norms"].get(key, 0) <= 0
                           for value in update_rows for key in families)):
                raise AssertionError(f"SW0155 {arm} preflight failed selected-family change/gradient checks")
    manifest_path = source["manifest_path"]
    report = {"experiment": EXPERIMENT, "stage": "actual_source_masked_two_update_preflight",
        "status": "source_preflight_complete", "seed": train.SEED, "training_admission": True,
        "ground_truth_used": True, "ground_truth_scope": "TRAIN masks only",
        "source": {"checkpoint_sha256": source["source_sha256"],
            "manifest_sha256": train.sha256_file(manifest_path),
            "training_ids": source["image_ids"],
            "training_ids_sha256": train.ids_sha(source["image_ids"]),
            "selected_train_ids": train_ids, "selected_train_ids_sha256": train.ids_sha(train_ids),
            "rgb_asset_validation": assets, "saved_sw0135_parity_sha256": parity_sha,
            "saved_sw0135_parity_status": saved_parity.get("status")},
        "canonical_hdf5": hdf5_metadata,
        "registered": {"updates": PREFLIGHT_UPDATES, "logical_batch": train.BATCH,
            "microbatch": train.MICRO, "steps": train.STEPS, "settle": train.SETTLE,
            "live_tail": train.TAIL},
        "arms": arms, "implementation_fingerprint": train.implementation_fingerprint()}
    if not validate_report(report):
        raise AssertionError("constructed SW0155 preflight report failed its own validator")
    _atomic_create_json(output, report)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    report = run_preflight(device=args.device, output=args.output)
    print(json.dumps({"status": report["status"], "training_admission": True,
        "arms": list(report["arms"]), "ground_truth_scope": report["ground_truth_scope"]},
        allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

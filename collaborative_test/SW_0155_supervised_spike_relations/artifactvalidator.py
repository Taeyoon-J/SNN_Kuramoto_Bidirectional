"""Strict reader for completed SW0155 training artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

from collaborative_test.SW_0155_supervised_spike_relations import preflight_migration, train
from collaborative_test.SW_0130_phase_state_integration import run as sw130


def validate_training_artifact(task, *, source=None, assets=None):
    """Validate the actual ``training_complete`` manifest and every payload hash."""
    try:
        arm = task["arm"]
        if arm not in train.ARMS:
            return False
        output = Path(task["output"])
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return False
        source = source or train.source_contract()
        assets = assets or sw130.validate_rgb_assets()
        ids = [int(value) for value in source["image_ids"]]
        rows = ids[:2048]
        expected_impl = train.implementation_fingerprint()
        preflight_path = train.HERE / "results_archive" / preflight_migration.MIGRATED_REPORT_NAME
        if not preflight_path.is_file():
            return False
        hdf_path = Path(train.HDF5_PATH)
        if not hdf_path.is_file():
            return False
        hdf_stat = hdf_path.stat()
        bindings = manifest.get("bindings")
        if not isinstance(bindings, dict):
            return False
        if (manifest.get("status") != "training_complete"
                or manifest.get("experiment") != train.EXPERIMENT
                or manifest.get("arm") != arm or manifest.get("seed") != train.SEED
                or manifest.get("completed_updates") != train.UPDATES
                or manifest.get("logical_batch") != train.BATCH
                or manifest.get("microbatch") != train.MICRO
                or manifest.get("exposures") != train.BATCH * train.UPDATES
                or manifest.get("train_ids") != rows
                or bindings.get("experiment") != train.EXPERIMENT
                or bindings.get("arm") != arm or bindings.get("seed") != train.SEED
                or bindings.get("source_checkpoint_sha256") != source["source_sha256"]
                or bindings.get("source_manifest_sha256") != train.sha256_file(source["manifest_path"])
                or bindings.get("source_training_ids_sha256") != train.ids_sha(ids)
                or bindings.get("ordered_train_ids_sha256") != train.ids_sha(rows)
                or bindings.get("rgb_asset_validation") != assets
                or bindings.get("preflight_sha256") != train.sha256_file(preflight_path)
                or bindings.get("hdf5_path") != str(hdf_path.resolve())
                or bindings.get("hdf5_size") != hdf_stat.st_size
                or bindings.get("hdf5_mtime_ns") != hdf_stat.st_mtime_ns
                or bindings.get("steps") != train.STEPS
                or bindings.get("settle") != train.SETTLE
                or bindings.get("live_tail") != train.TAIL
                or bindings.get("ground_truth_used") is not True
                or bindings.get("ground_truth_scope") != "TRAIN masks only"
                or manifest.get("ground_truth_used") is not True
                or manifest.get("ground_truth_scope") != "TRAIN masks only"
                or manifest.get("implementation_fingerprint") != expected_impl
                or bindings.get("implementation_fingerprint") != expected_impl):
            return False
        expected_groups = ({"spatial_residual", "graph"} if arm == "encoder_graph"
                           else {"spatial_residual", "graph", "native_core", "integration_tensors"})
        initial = manifest.get("initial_parameter_sha256")
        final = manifest.get("final_parameter_sha256")
        if (not isinstance(initial, dict) or not isinstance(final, dict)
                or set(initial) != expected_groups or set(final) != expected_groups):
            return False
        for family in expected_groups:
            if (not isinstance(initial[family], str) or len(initial[family]) != 64
                    or not isinstance(final[family], str) or len(final[family]) != 64):
                return False
        changed_required = {"spatial_residual", "graph"}
        if arm == "joint":
            changed_required.add("native_core")
        if any(initial[name] == final[name] for name in changed_required):
            return False
        steps = manifest.get("optimizer_step_summary")
        if (not isinstance(steps, dict) or set(steps) != expected_groups
                or any(not isinstance(value, int) or isinstance(value, bool)
                       or not 1 <= value <= train.UPDATES for value in steps.values())):
            return False
        frozen = manifest.get("frozen_base_encoder_sha256_before_after")
        if not isinstance(frozen, str) or len(frozen) != 64:
            return False
        hashes = {}
        for name in ("checkpoint", "optimizer", "history"):
            path = output / f"{name}.{'json' if name == 'history' else 'pt'}"
            digest = manifest.get(f"{name}_sha256")
            if (not path.is_file() or not isinstance(digest, str) or len(digest) != 64
                    or train.sha256_file(path) != digest):
                return False
            hashes[name] = path
        if manifest.get("checkpoint_sha256") != train.sha256_file(hashes["checkpoint"]):
            return False
        if manifest.get("optimizer_sha256") != train.sha256_file(hashes["optimizer"]):
            return False
        history = json.loads(hashes["history"].read_text(encoding="utf-8"))
        if not isinstance(history, list) or len(history) != train.UPDATES:
            return False
        for index, row in enumerate(history, start=1):
            if (not isinstance(row, dict) or row.get("update") != index
                    or row.get("image_ids") != rows[(index - 1) * train.BATCH:index * train.BATCH]
                    or not isinstance(row.get("mean_loss"), (int, float))
                    or not math.isfinite(row["mean_loss"])):
                return False
            norms = row.get("family_gradient_norms")
            if (not isinstance(norms, dict) or set(norms) != expected_groups
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                           for v in norms.values())):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AssertionError, RuntimeError):
        return False

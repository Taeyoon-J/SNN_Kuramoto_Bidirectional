#!/usr/bin/env python3
"""Validate one-update real-data arm preflight and bind it to current inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from arms import arm_config

DEPENDENCIES = (
    "snn_kuramoto_bidirectional/s2net_cls.py",
    "snn_kuramoto_bidirectional/graph_generator.py",
    "snn_kuramoto_bidirectional/input_layer_generator.py",
    "snn_kuramoto_bidirectional/gamma_initializer.py",
    "snn_kuramoto_bidirectional/loss_function.py",
    "snn_kuramoto_bidirectional/training/train_s2net_core.py",
)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(arm, manifest_path, model_dir, root, gamma, encoder, stats, trainer, initial_core=None):
    config = arm_config(arm)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("experiment_arm") != arm:
        raise ValueError("preflight arm identity mismatch")
    if manifest.get("arm_spec") != config or manifest.get("arm_spec_sha256") != sha(Path(trainer).with_name("arms.py")):
        raise ValueError("stale or mismatched arm specification")
    if manifest.get("steps_completed") != 1 or manifest.get("epochs_completed") != 1:
        raise ValueError("preflight must contain exactly one optimizer update")
    history = manifest.get("history", [])
    if len(history) != 1 or history[0].get("samples_seen") != 16 or any(not math.isfinite(float(history[0][key]))
                                for key in ("loss", "primary", "spike", "slot_reconstruction")):
        raise ValueError("preflight loss components must be present and finite")
    if manifest.get("warmup_epochs") != 0 or manifest.get("samples") != 2500:
        raise ValueError("wrong SW0084 exposure/warmup contract")
    if (manifest.get("freeze_core") is not False or manifest.get("freeze_graph") is not False
            or manifest.get("graph_checkpoint") is not None):
        raise ValueError("SW0084 must train the full core and graph from the declared start state")
    if (manifest.get("batch_size") != 16 or manifest.get("core_lr") != 3e-4
            or manifest.get("encoder_lr") != 3e-5
            or manifest.get("slot_num_slots") != 7
            or manifest.get("slot_temperature") != 0.3):
        raise ValueError("SW0084 recipe mismatch")
    if manifest.get("training_ids", {}).get("full_segments") != [[0, 999], [1640, 3139]]:
        raise ValueError("wrong SW0084 training scene IDs")
    if manifest.get("ground_truth_used_for_training") is not False or manifest.get("hdf5_fields_opened") != ["image"]:
        raise ValueError("training provenance includes forbidden non-image fields")
    if float(manifest.get("slot_reconstruction_weight", -1)) != config["reconstruction_weight"]:
        raise ValueError("preflight arm reconstruction weight mismatch")
    if float(manifest.get("core_max_parameter_change", 0)) <= 0 or float(manifest.get("graph_max_parameter_change", 0)) <= 0:
        raise ValueError("joint preflight did not update core and graph")
    if float(manifest.get("encoder_max_parameter_change", 0)) <= 0:
        raise ValueError("joint preflight did not update encoder")
    if float(manifest.get("initial_gamma_max_abs_diff", math.inf)) > 2e-6:
        raise ValueError("pretrained encoder fails registered cached-gamma reproduction")
    if config["reconstruction_weight"] > 0:
        gradient = manifest.get("phase_slot_reconstruction_encoder_gradient_norm_first_batch")
        if gradient is None or not math.isfinite(float(gradient)) or float(gradient) <= 0:
            raise ValueError("reconstruction encoder gradient must be finite and nonzero")
    if sha(gamma) != manifest.get("anchor_gamma_sha256"):
        raise ValueError("stale preflight gamma hash")
    if sha(encoder) != manifest.get("source_encoder_sha256") or sha(stats) != manifest.get("stats_sha256"):
        raise ValueError("stale preflight encoder/statistics hashes")
    if sha(trainer) != manifest.get("trainer_sha256"):
        raise ValueError("stale preflight trainer hash")
    if initial_core:
        if sha(initial_core) != manifest.get("source_core_sha256"):
            raise ValueError("stale initialized core hash")
    elif manifest.get("source_core_sha256") is not None:
        raise ValueError("fresh-core arm unexpectedly used an initial checkpoint")
    actual_dependencies = {name: sha(Path(root) / name) for name in DEPENDENCIES}
    if actual_dependencies != manifest.get("code_dependencies_sha256"):
        raise ValueError("stale model/objective code dependency hash")
    model_dir = Path(model_dir)
    if not (model_dir / "core.pt").is_file() or not (model_dir / "encoder.pt").is_file():
        raise ValueError("preflight checkpoints are incomplete")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    for name in ("arm", "manifest", "model-dir", "root", "gamma", "encoder", "stats", "trainer"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--initial-core")
    args = parser.parse_args()
    result = validate(args.arm, args.manifest, args.model_dir, args.root,
                      args.gamma, args.encoder, args.stats, args.trainer, args.initial_core)
    print(json.dumps({"valid": True, "arm": args.arm,
                      "reconstruction_gradient_norm": result.get(
                          "phase_slot_reconstruction_encoder_gradient_norm_first_batch")}, indent=2))


if __name__ == "__main__":
    main()

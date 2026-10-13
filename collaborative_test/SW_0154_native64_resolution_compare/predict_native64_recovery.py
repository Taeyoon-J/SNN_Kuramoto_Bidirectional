"""Seed-specific parity admission wrapper for SW0154 source seeds 0 and 2.

The frozen SW0150 parity validator has a legacy default seed of 1. This
process-local adapter scopes that variable to the requested seed, then calls
the unchanged SW0154 predictor and records the adapter provenance in its
published protocol. It does not alter the predictor's fingerprint or read GT.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from collaborative_test.SW_0150_spike_spatial_mixture import preflight as legacy_preflight
from collaborative_test.SW_0154_native64_resolution_compare import predict_native64 as predictor

SEEDS = (0, 2)
EXPECTED_PREDICTOR_SHA256 = "4b9f0e45852bc36394e1adadcb5e39417e651c491dd419c1842bdfb569e746a8"
EXPECTED_SOURCE_SHA256 = {
    0: "36f2481dd1fa51fa29bd4dc34275a876b71d8b76fbee13ac47a0a1bfe6766fbf",
    2: "798ad3e9d4bf837b1bbeb1bd7c13900df511b5c76f5736d2b9f8b973d7fa5661",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256():
    return sha256_file(Path(__file__))


def validate_seed_parity(seed, *, source_sha, source_manifest_sha, training_ids,
                         assets, parity_path=None):
    """Validate the real seed-specific SW0135 receipt with scoped legacy seed."""
    if seed not in SEEDS or source_sha != EXPECTED_SOURCE_SHA256[seed]:
        raise ValueError("recovery accepts only source seed0/2 with their frozen source hashes")
    if parity_path is None:
        parity_path = (predictor.ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive"
                       / f"native32_zero_adapter_parity_seed{seed}_b4.json")
    old_seed = legacy_preflight.SEED
    try:
        legacy_preflight.SEED = seed
        payload, parity_sha = legacy_preflight.load_and_validate_parity(
            parity_path, source_sha=source_sha, source_manifest_sha=source_manifest_sha,
            training_ids=training_ids, asset_validation=assets)
    finally:
        legacy_preflight.SEED = old_seed
    if payload.get("source_seed") != seed:
        raise ValueError("seed-specific native32 parity receipt reports the wrong source seed")
    return payload, parity_sha


def predict_seed(seed, *, device="cuda:0", dataset=predictor.DATASET,
                 prediction_root=predictor.PREDICTION_ROOT, work_root=predictor.WORK_ROOT):
    if seed not in SEEDS:
        raise ValueError("recovery wrapper is registered only for missing native64 seeds0 and2")
    if sha256_file(Path(predictor.__file__)) != EXPECTED_PREDICTOR_SHA256:
        raise ValueError("frozen native64 predictor bytes changed")
    _checkpoint, manifest_path, _manifest, pool, train_ids, source_sha = predictor.sw130.source_contract(seed)
    if source_sha != EXPECTED_SOURCE_SHA256[seed] or len(train_ids) != 4096 or len(pool) != 4096:
        raise ValueError("actual SW0097 source contract is not the registered seed-specific source")
    assets = predictor.sw130.validate_rgb_assets()
    parity_path = (predictor.ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive"
                   / f"native32_zero_adapter_parity_seed{seed}_b4.json")
    _payload, parity_sha = validate_seed_parity(seed, source_sha=source_sha,
        source_manifest_sha=sha256_file(manifest_path), training_ids=train_ids,
        assets=assets, parity_path=parity_path)
    repair = {"status": "seed_specific_legacy_validator_override",
        "recovery_entrypoint_sha256": implementation_sha256(),
        "frozen_predictor_sha256": EXPECTED_PREDICTOR_SHA256,
        "legacy_validator_module": "SW0150_spike_spatial_mixture.preflight",
        "legacy_default_seed": 1, "scoped_seed": seed,
        "parity_path": str(Path(parity_path).resolve()), "parity_sha256": parity_sha,
        "source_checkpoint_sha256": source_sha,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_training_ids_sha256": hashlib.sha256(
            predictor.np.asarray(train_ids, dtype="<i8").tobytes()).hexdigest()}
    original_publish = predictor._publish_prediction

    def publish_with_repair(output_dir, image_ids, labels, protocol_builder):
        def build_with_repair(prediction_sha):
            protocol = protocol_builder(prediction_sha)
            protocol["seed_specific_parity_recovery"] = repair
            return protocol
        return original_publish(output_dir, image_ids, labels, build_with_repair)

    original_seed = legacy_preflight.SEED
    predictor._publish_prediction = publish_with_repair
    legacy_preflight.SEED = seed
    try:
        protocol = predictor.predict_seed(seed, device=device, dataset=dataset,
            prediction_root=prediction_root, work_root=work_root)
    finally:
        legacy_preflight.SEED = original_seed
        predictor._publish_prediction = original_publish
    if protocol.get("seed_specific_parity_recovery") != repair:
        raise ValueError("recovery provenance was not sealed into the prediction protocol")
    return protocol


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    report = predict_seed(args.seed, device=args.device)
    print(json.dumps({"status": report["status"], "seed": args.seed,
        "count": predictor.COUNT, "ground_truth_used_for_prediction": False,
        "repair_sha256": report["seed_specific_parity_recovery"]["recovery_entrypoint_sha256"]},
        allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

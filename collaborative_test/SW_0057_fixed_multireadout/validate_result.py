"""Validate SW0057 report schema, provenance, and finite metrics."""
import argparse
import json
import math
from pathlib import Path


def validate(path, checkpoint, seed, expected_count=320):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["experiment"] == "SW0057 fixed multi-readout"
    assert report["seed"] == int(seed)
    assert report["ids"] == [1320, 1320 + expected_count - 1]
    assert report["images"] == expected_count
    assert report["checkpoint"]["sha256"]
    assert report["readout_contract"]["ground_truth_used_for_prediction"] is False
    assert report["inference"]["steps"] == 1024 and report["inference"]["settle"] == 512
    rows = report["rows"]
    expected = {"spike_cc_threshold_0p50", "membrane_spatial_sigma1p5_k10"}
    assert {row["readout"] for row in rows} == expected
    for row in rows:
        assert len(row["per_image_object_counts"]) == expected_count
        assert all(math.isfinite(float(v)) for v in row["metrics"].values())
        assert math.isfinite(row["predicted_foreground_fraction"])
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("report")
    p.add_argument("checkpoint")
    p.add_argument("seed", type=int, choices=(0, 1, 2))
    p.add_argument("--expected-count", type=int, default=320)
    args = p.parse_args()
    import hashlib
    report = validate(args.report, args.checkpoint, args.seed, args.expected_count)
    digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    assert report["checkpoint"]["sha256"] == digest, "checkpoint changed after evaluation"
    evaluator_digest = hashlib.sha256((Path(__file__).parent / "evaluate.py").read_bytes()).hexdigest()
    assert report["evaluator"]["sha256"] == evaluator_digest, "evaluator changed after evaluation"
    print(f"validated SW0057 seed{args.seed}: {args.expected_count} images")


if __name__ == "__main__":
    main()

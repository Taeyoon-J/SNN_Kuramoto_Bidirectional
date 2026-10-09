"""CLI adapter for completed SW0125 full-horizon endpoint evaluations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0125_late_rollout_credit import run


def evaluation_fingerprint():
    return {"evaluation_adapter_sha256": run.sha(Path(__file__).resolve()),
            "training_runner_sha256": run.sha(run.RUNNER),
            "shared_evaluator_sha256": run.sha(
                ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=run.SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run.evaluate(args.seed, args.checkpoint, args.output, args.device)
    print(json.dumps({"status": report["status"], "experiment": "SW0125",
                      "seed": args.seed, "output": str(args.output)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

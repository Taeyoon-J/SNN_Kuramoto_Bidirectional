"""Ordered state contract for the SW0056 low-load sequential runner."""
import argparse
from pathlib import Path

ORDER = ("WAITING_FOR_LOW_LOAD", "SMOKE_VALIDATED", "SEED0_COMPLETED", "SEED1_COMPLETED",
         "SEED2_COMPLETED", "SUMMARY_COMPLETED", "COMPLETED")


def validate_state(state_dir, complete=False):
    present = [name for name in ORDER if (Path(state_dir) / name).is_file()]
    if present != list(ORDER[:len(present)]):
        raise ValueError(f"invalid scheduler phase order: {present}")
    if complete and present != list(ORDER):
        raise ValueError(f"scheduler incomplete: {present}")
    return present


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-state", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(" -> ".join(ORDER))
    elif args.validate_state:
        print("validated " + " -> ".join(validate_state(args.validate_state, args.require_complete)))
    else:
        parser.error("choose --dry-run or --validate-state")


if __name__ == "__main__":
    main()

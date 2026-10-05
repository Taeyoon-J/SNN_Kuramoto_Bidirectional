"""SW0057 one-shot scheduler state contract and dry-run plan."""
import argparse
from pathlib import Path

ORDER = ("WAITING_FOR_SW0055", "SW0055_READY", "SEEDS01_STARTED", "SEEDS01_COMPLETED",
         "SEED2_STARTED", "SEED2_COMPLETED", "SUMMARY_COMPLETED", "LAUNCH_COMPLETED")


def validate_markers(state_dir, require_complete=False):
    present = [name for name in ORDER if (Path(state_dir) / name).is_file()]
    if present != list(ORDER[:len(present)]):
        raise ValueError(f"scheduler markers are not a valid ordered prefix: {present}")
    if require_complete and present != list(ORDER):
        raise ValueError(f"scheduler is incomplete; markers present: {present}")
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
        print("validated " + " -> ".join(validate_markers(args.validate_state, require_complete=args.require_complete)))
    else:
        parser.error("choose --dry-run or --validate-state")


if __name__ == "__main__":
    main()

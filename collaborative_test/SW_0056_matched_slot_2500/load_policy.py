"""Pure policy for the SW0056 system-load gate."""
import argparse
import math


def load_is_acceptable(load1, load5, load15, mem_available_kb):
    values = [float(load1), float(load5), float(load15)]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        return False
    return values[0] <= 32 and values[1] <= 36 and values[2] <= 40 and int(mem_available_kb) >= 8 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("load1", type=float); parser.add_argument("load5", type=float)
    parser.add_argument("load15", type=float); parser.add_argument("mem_available_kb", type=int)
    args = parser.parse_args()
    raise SystemExit(0 if load_is_acceptable(args.load1, args.load5, args.load15, args.mem_available_kb) else 1)


if __name__ == "__main__":
    main()

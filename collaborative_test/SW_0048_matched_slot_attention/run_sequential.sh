#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-full}"
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
for seed in 0 1 2; do
  bash "$RUN_DIR/run_seed.sh" "$seed" "$MODE"
done

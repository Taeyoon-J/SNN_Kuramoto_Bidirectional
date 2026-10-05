#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-full}"
for seed in 0 1 2; do
  bash /Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0056_matched_slot_2500/run_seed.sh "$seed" "$MODE"
done
echo "SW0056 sequential seeds completed in mode=$MODE"

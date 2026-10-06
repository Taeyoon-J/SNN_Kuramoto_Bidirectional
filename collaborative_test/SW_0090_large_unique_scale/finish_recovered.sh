#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0090_large_unique_scale"
OUT="$ROOT/trained_models/SW0090_full320_long"
mkdir -p "$OUT"
for seed in 0 1; do
 for epoch in 1 3 10; do
  if [[ ! -e "$OUT/seed${seed}_epoch${epoch}.json" ]]; then
   cp "$ROOT/trained_models/SW0090_early_seed01/seed${seed}_epoch${epoch}.json" "$OUT/seed${seed}_epoch${epoch}.json"
  fi
 done
done
for epoch in 1 3 10; do bash "$DIR/evaluate.sh" 0 2 "$epoch"; done
/Data0/kevinswk/envs/snn/bin/python "$DIR/summarize.py" --results "$OUT" --output "$OUT/summary.json"
printf '{"status":"complete"}\n' > "$ROOT/trained_models/SW0090_QUEUE.json"

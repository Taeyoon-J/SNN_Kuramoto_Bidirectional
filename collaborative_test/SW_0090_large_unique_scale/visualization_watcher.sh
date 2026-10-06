#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
SUMMARY="$ROOT/trained_models/SW0090_full320_long/summary.json"
while [[ ! -s "$SUMMARY" ]]; do sleep 60; done
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0090_large_unique_scale/finish_visualization.py" \
  --root "$ROOT" > "$ROOT/trained_models/SW0090_visualization.log" 2>&1
echo complete > "$ROOT/trained_models/SW0090_VISUALIZATION_COMPLETE"


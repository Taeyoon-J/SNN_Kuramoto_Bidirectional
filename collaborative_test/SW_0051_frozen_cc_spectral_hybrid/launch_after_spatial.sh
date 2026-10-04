#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
SEED="${2:?seed 0, 1, or 2 required}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s${SEED}"
while true; do
  test -s "$OUT/core.pt" && test -s "$OUT/validation_short_T256_settle64.json" && \
    test -s "$OUT/validation_long_T1024_settle512.json" && test -s "$OUT/SPATIAL_EVALUATED" && break
  sleep 20
done
while true; do
  if ! nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    # Recheck all prerequisites immediately before starting to avoid a stale-ready race.
    test -s "$OUT/core.pt" && test -s "$OUT/SPATIAL_EVALUATED"
    "$(dirname "$0")/preflight.sh" "$GPU_ID" "$SEED"
    [[ ! -e "$OUT/frozen_cc_hybrid_short_T256_settle64.json" && ! -e "$OUT/frozen_cc_hybrid_long_T1024_settle512.json" ]] || {
      echo "Refusing to overwrite existing SW0051 results" >&2; exit 3;
    }
    "$(dirname "$0")/evaluate.sh" "$GPU_ID" "$SEED" short
    "$(dirname "$0")/evaluate.sh" "$GPU_ID" "$SEED" long
    printf 'completed\n' > "$OUT/FROZEN_CC_HYBRID_EVALUATED"
    exit 0
  fi
  sleep 20
done

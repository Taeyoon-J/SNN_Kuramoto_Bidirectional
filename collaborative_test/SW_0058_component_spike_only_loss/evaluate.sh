#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0/1 required}"; SEED="${2:?seed 0/1/2 required}"; WINDOW="${3:?short or long required}"; MODE="${4:?smoke or full required}"
case "$GPU" in 0|1) ;; *) echo "GPU0/1 only" >&2; exit 2 ;; esac
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0/1/2" >&2; exit 2 ;; esac
case "$WINDOW" in short) STEPS=256; SETTLE=64 ;; long) STEPS=1024; SETTLE=512 ;; *) echo "window must be short or long" >&2; exit 2 ;; esac
case "$MODE" in smoke) COUNT=4 ;; full) COUNT=320 ;; *) echo "mode must be smoke or full" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0058_component_spike_only_loss"
OUT="$ROOT/trained_models/SW_0058_component_spike_only_s${SEED}_lr0p0003_epoch25"
PY=/Data0/kevinswk/envs/snn/bin/python
CHECKPOINT="$OUT/core.pt"
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
RESULT="$OUT/fixed_spike_${WINDOW}_${MODE}_n${COUNT}_T${STEPS}_settle${SETTLE}.json"
LOG="$OUT/fixed_spike_${WINDOW}_${MODE}_n${COUNT}_T${STEPS}_settle${SETTLE}.log"
test -s "$OUT/TRAINING_COMPLETED" -a -s "$CHECKPOINT" -a -s "$OUT/manifest.json"
if [[ "$MODE" == full ]]; then test -s "$OUT/REAL_ASSET_SMOKE_COMPLETED"; fi
test -s "$GAMMA" -a -s "$MANIFEST" -a -s "$HDF5"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "Refusing overwrite/partial evaluation: $RESULT" >&2; exit 3; }
if ! PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; then echo "$PIDS" >&2; exit 2; fi
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU occupied; refusing evaluation" >&2; exit 2; }
mkdir -p "$OUT/cache"
export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
"$PY" "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --gamma-global-start 1320 \
  --gamma-manifest "$MANIFEST" --dataset-path "$HDF5" --output-path "$RESULT" \
  --start 1320 --count "$COUNT" --steps "$STEPS" --settle "$SETTLE" \
  --membrane-vth 0.06 --min-group-size 2 --background largest_component \
  --dendritic-projection shared --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --thresholds 0.50 --phase-endpoint --device cuda > "$LOG" 2>&1
"$PY" "$DIR/validate_result.py" spike "$RESULT" "$CHECKPOINT" "$SEED" "$WINDOW" "$COUNT"
echo "SW0058 fixed spike evaluation completed: seed=$SEED window=$WINDOW"

#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0/1 required}"; SEED="${2:?seed 0/1/2 required}"; MODE="${3:?smoke or full required}"
case "$GPU" in 0|1) ;; *) echo "GPU0/1 only" >&2; exit 2 ;; esac
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0/1/2" >&2; exit 2 ;; esac
case "$MODE" in smoke) COUNT=4; SUFFIX="smoke_n4" ;; full) COUNT=320; SUFFIX="full_n320" ;; *) echo "mode must be smoke or full" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0058_component_spike_only_loss"
OUT="$ROOT/trained_models/SW_0058_component_spike_only_s${SEED}_lr0p0003_epoch25"
PY=/Data0/kevinswk/envs/snn/bin/python
CHECKPOINT="$OUT/core.pt"
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
RESULT="$OUT/sw0057_fixed_multireadout_long_${SUFFIX}.json"
LOG="$OUT/sw0057_fixed_multireadout_long_${SUFFIX}.log"
test -s "$OUT/TRAINING_COMPLETED" -a -s "$CHECKPOINT"
if [[ "$MODE" == full ]]; then test -s "$OUT/REAL_ASSET_SMOKE_COMPLETED"; fi
test -s "$GAMMA" -a -s "$MANIFEST" -a -s "$HDF5"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "Refusing overwrite/partial multireadout: $RESULT" >&2; exit 3; }
if ! PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; then echo "$PIDS" >&2; exit 2; fi
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU occupied; refusing evaluation" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
"$PY" "$ROOT/collaborative_test/SW_0057_fixed_multireadout/evaluate.py" \
  --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --gamma-manifest "$MANIFEST" \
  --dataset-path "$HDF5" --output-path "$RESULT" --seed "$SEED" \
  --global-start 1320 --count "$COUNT" --steps 1024 --settle 512 --device cuda \
  > "$LOG" 2>&1
"$PY" "$ROOT/collaborative_test/SW_0057_fixed_multireadout/validate_result.py" \
  "$RESULT" "$CHECKPOINT" "$SEED" --expected-count "$COUNT"
echo "SW0058 fixed long multireadout completed: seed=$SEED"

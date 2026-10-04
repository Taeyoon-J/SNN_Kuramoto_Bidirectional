#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Pass assigned GPU ID}"
SEED="${2:?Pass seed 0, 1, or 2}"
TRAIN_PID="${3:?Pass training PID}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac
if [[ ! "$TRAIN_PID" =~ ^[0-9]+$ ]]; then
  echo "TRAIN_PID must be numeric" >&2
  exit 2
fi

ROOT=/Data0/kevinswk/patch_v2_sw
EXP="$ROOT/collaborative_test/SW_0042_hdf5_aligned_bim6"
OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s${SEED}"
TRAIN_LOG="$OUT/training.log"
mkdir -p "$OUT/cache"

echo "Waiting for training PID $TRAIN_PID (seed $SEED)" | tee "$OUT/watch.log"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 20; done

if [[ ! -s "$OUT/core.pt" ]] || ! grep -Fq "trained S2NetCore: $OUT/core.pt" "$TRAIN_LOG"; then
  echo "Training completion marker/checkpoint missing; see $TRAIN_LOG" | tee -a "$OUT/watch.log" >&2
  exit 1
fi
echo "Training complete; checkpoint verified" | tee -a "$OUT/watch.log"

bash "$EXP/evaluate.sh" "$GPU_ID" "$SEED" short
echo "Completed short T256/settle64" | tee -a "$OUT/watch.log"
bash "$EXP/evaluate.sh" "$GPU_ID" "$SEED" long
echo "Completed long T1024/settle512" | tee -a "$OUT/watch.log"

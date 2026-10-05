#!/usr/bin/env bash
set -euo pipefail
GPU1="${1:?first idle GPU required}"; GPU2="${2:?second idle GPU required}"; GPU3="${3:?third idle GPU required}"
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0052_checkpoint_trajectory"
OUT="$ROOT/trained_models/SW_0052_checkpoint_trajectory"
mkdir -p "$OUT"
for gpu in "$GPU1" "$GPU2" "$GPU3"; do
  pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)"
  [[ ! "$pids" =~ [0-9] ]] || { echo "GPU $gpu is occupied" >&2; exit 2; }
done
for expected in \
  low_lr_epoch20_short_T256_settle64.json low_lr_epoch25_short_T256_settle64.json \
  low_lr_epoch30_short_T256_settle64.json low_lr_epoch35_short_T256_settle64.json \
  diversity_epoch20_short_T256_settle64.json diversity_epoch25_short_T256_settle64.json \
  diversity_epoch30_short_T256_settle64.json diversity_epoch35_short_T256_settle64.json; do
  [[ ! -e "$OUT/$expected" ]] || { echo "refusing existing output $expected" >&2; exit 1; }
done
nohup bash "$DIR/worker_short.sh" "$GPU1" low_lr 20 low_lr 35 diversity 30 > "$OUT/worker_gpu${GPU1}.log" 2>&1 & echo $! > "$OUT/worker_gpu${GPU1}.pid"
nohup bash "$DIR/worker_short.sh" "$GPU2" low_lr 25 diversity 20 diversity 35 > "$OUT/worker_gpu${GPU2}.log" 2>&1 & echo $! > "$OUT/worker_gpu${GPU2}.pid"
nohup bash "$DIR/worker_short.sh" "$GPU3" low_lr 30 diversity 25 > "$OUT/worker_gpu${GPU3}.log" 2>&1 & echo $! > "$OUT/worker_gpu${GPU3}.pid"
echo "launched SW0052 short screen on GPUs $GPU1 $GPU2 $GPU3"

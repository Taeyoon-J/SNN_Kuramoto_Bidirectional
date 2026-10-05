#!/usr/bin/env bash
set -euo pipefail
GPU_BASELINE="${1:?idle GPU for seed2 baseline required}"
GPU_SEED0="${2:?idle GPU for seed0 low-LR required}"
GPU_SEED1="${3:?idle GPU for seed1 low-LR required}"
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0053_lr_early_stop_matrix"
for gpu in "$GPU_BASELINE" "$GPU_SEED0" "$GPU_SEED1"; do
  pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)"
  [[ ! "$pids" =~ [0-9] ]] || { echo "GPU $gpu occupied" >&2; exit 2; }
done
for out in \
  "$ROOT/trained_models/SW_0050_sample_diversity_s2_w0_lr0p001_epoch25" \
  "$ROOT/trained_models/SW_0050_sample_diversity_s0_w0_lr0p0003_epoch25" \
  "$ROOT/trained_models/SW_0050_sample_diversity_s1_w0_lr0p0003_epoch25"; do
  [[ ! -e "$out" ]] || { echo "refusing existing output $out" >&2; exit 1; }
done
bash "$ROOT/collaborative_test/SW_0050_sample_diversity/preflight.sh" "$GPU_BASELINE"
nohup bash "$DIR/train_evaluate.sh" "$GPU_BASELINE" 2 0.001 > "$ROOT/trained_models/SW0053_seed2_baseline_epoch25.log" 2>&1 & echo $! > "$ROOT/trained_models/SW0053_seed2_baseline_epoch25.pid"
nohup bash "$DIR/train_evaluate.sh" "$GPU_SEED0" 0 0.0003 > "$ROOT/trained_models/SW0053_seed0_low_lr_epoch25.log" 2>&1 & echo $! > "$ROOT/trained_models/SW0053_seed0_low_lr_epoch25.pid"
nohup bash "$DIR/train_evaluate.sh" "$GPU_SEED1" 1 0.0003 > "$ROOT/trained_models/SW0053_seed1_low_lr_epoch25.log" 2>&1 & echo $! > "$ROOT/trained_models/SW0053_seed1_low_lr_epoch25.pid"
echo "launched SW0053 on GPUs $GPU_BASELINE $GPU_SEED0 $GPU_SEED1"

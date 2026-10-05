#!/usr/bin/env bash
set -euo pipefail
GPU0="${1:?GPU for seed0}"; GPU1="${2:?GPU for seed1}"; GPU2="${3:?GPU for seed2}"
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0055_unique_data_scale"
[[ "$GPU0" != "$GPU1" && "$GPU0" != "$GPU2" && "$GPU1" != "$GPU2" ]] || { echo "GPU IDs must be three distinct devices" >&2; exit 2; }
check_gpus_idle() {
  for gpu in "$GPU0" "$GPU1" "$GPU2"; do
    p="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)"
    [[ ! "$p" =~ [0-9] ]] || { echo "GPU $gpu occupied" >&2; return 2; }
  done
}
check_gpus_idle
for seed in 0 1 2; do [[ ! -e "$ROOT/trained_models/SW0055_unique2500_s${seed}_e10_lr0p0003" ]] || { echo "seed $seed output exists" >&2; exit 1; }; done
bash "$DIR/preflight_training.sh" "$GPU0"
check_gpus_idle || { echo "GPU occupancy changed during preflight; refusing partial launch" >&2; exit 2; }
for pair in "0:$GPU0" "1:$GPU1" "2:$GPU2"; do seed="${pair%%:*}"; gpu="${pair##*:}"; nohup bash "$DIR/train_evaluate.sh" "$gpu" "$seed" > "$ROOT/trained_models/SW0055_unique2500_s${seed}_launcher.log" 2>&1 & echo $! > "$ROOT/trained_models/SW0055_unique2500_s${seed}_launcher.pid"; done
echo "launched SW0055 matched-exposure seeds 0/1/2"

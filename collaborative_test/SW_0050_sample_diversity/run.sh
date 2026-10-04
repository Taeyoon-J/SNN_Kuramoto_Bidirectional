#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Pass an assigned GPU ID}"
SEED="${2:?Use rescue seed 2 or harm-control seed 0}"
WEIGHT="${3:?Pass a measured positive diversity weight}"
LR="${4:?Pass optimizer learning rate}"
MODE="${5:?Use pilot or full}"
case "$SEED" in 0|2) ;; *) echo "SEED must be 0 or 2" >&2; exit 2 ;; esac
case "$MODE" in
  pilot) EPOCHS=2 ;;
  full) EPOCHS=40 ;;
  *) echo "MODE must be pilot or full" >&2; exit 2 ;;
esac
PYTHON=/Data0/kevinswk/envs/snn/bin/python
if [[ ! "$WEIGHT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] \
  || ! "$PYTHON" -c 'import math,sys; v=float(sys.argv[1]); sys.exit(0 if math.isfinite(v) and v>=0 else 1)' "$WEIGHT"; then
  echo "WEIGHT must be a finite non-negative number" >&2; exit 2
fi
if [[ ! "$LR" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] \
  || ! "$PYTHON" -c 'import math,sys; v=float(sys.argv[1]); sys.exit(0 if math.isfinite(v) and v>0 else 1)' "$LR"; then
  echo "LR must be a finite positive number" >&2; exit 2
fi
TAG="${WEIGHT//./p}"
LR_TAG="${LR//./p}"
ROOT=/Data0/kevinswk/patch_v2_sw
SOURCE=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002
GAMMA="$SOURCE/gamma_train.pt"
PREFLIGHT="$ROOT/trained_models/SW0050_PREFLIGHT_V1.txt"
CHECKSUM="$(sha256sum "$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s2/core.pt" | awk '{print $1}')"
GAMMA_SUM="$(sha256sum "$GAMMA" | awk '{print $1}')"
CODE_SUM="$(sha256sum "$ROOT/snn_kuramoto_bidirectional/training/train_s2net_core.py" \
  "$ROOT/snn_kuramoto_bidirectional/loss_function.py" "$ROOT/collaborative_test/SW_0050_sample_diversity/run.sh" \
  "$ROOT/collaborative_test/SW_0050_sample_diversity/probe_gradient_scale.py" \
  "$ROOT/collaborative_test/SW_0050_sample_diversity/preflight.sh" | sha256sum | awk '{print $1}')"
grep -Fxq "SW0050_PREFLIGHT_V1 $CHECKSUM $GAMMA_SUM $CODE_SUM" "$PREFLIGHT" || {
  echo "Asset-bound SW0050 preflight v1 is required before training" >&2; exit 3;
}
if ! GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_ID" 2>&1)"; then
  echo "Unable to query GPU $GPU_ID: $GPU_PIDS" >&2; exit 2
fi
if [[ "$GPU_PIDS" =~ [0-9] ]]; then echo "GPU $GPU_ID is occupied; refusing overlapping SW0050 training" >&2; exit 2; fi
OUT="$ROOT/trained_models/SW_0050_sample_diversity_s${SEED}_w${TAG}_lr${LR_TAG}_${MODE}"
if [[ "$MODE" == full ]]; then
  CHECKPOINT_ARGS=(--checkpoint-dir "$OUT/checkpoints" --checkpoint-epochs 20 25 30 35 40)
else
  CHECKPOINT_ARGS=()
fi
if [[ -e "$OUT" ]]; then echo "Refusing to overwrite existing SW0050 output: $OUT" >&2; exit 1; fi
test -s "$GAMMA"
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"
source /Data0/kevinswk/miniforge3/etc/profile.d/conda.sh
conda activate /Data0/kevinswk/envs/snn
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" python -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
  --gamma-seq-path "$GAMMA" --save-path "$OUT/core.pt" \
  --num-regions 256 --num-feature-maps 8 --device cuda \
  --epochs "$EPOCHS" --batch-size 16 --lr "$LR" --seed "$SEED" --osc-dim 4 \
  --gamma-drive-mode static --num-time-steps 64 --plv-settle 32 \
  --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
  --graph-mode learned --graph-top-k 32 --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --spike-spatial-grid-size 16 --k 256 --membrane-vth 0.06 \
  --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 \
  --branch 4 --gate-mode raw --plv-source phase --plv-combine mean \
  --spike-plv-weight 5.0 --loss-signal sigmoid_membrane \
  --sample-activity-diversity-weight "$WEIGHT" \
  "${CHECKPOINT_ARGS[@]}" \
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 \
  --structural-weight 0 --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0 \
  --plv-balance-weight 10.0 --plv-target-density 0.867 --plv-coherence-weight 0.5 \
  --spike-per-component --dendritic-projection shared --verbose \
  > "$OUT/training.log" 2>&1
grep -Fq "trained S2NetCore: $OUT/core.pt" "$OUT/training.log"
test -s "$OUT/core.pt"
printf 'completed\n' > "$OUT/TRAINING_COMPLETED"
echo "SW0050 training completed: $OUT"

#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU required}"; SEED="${2:?seed required}"
case "$GPU" in 0|1) ;; *) echo "GPU must be 0 or 1 while tkim1 is active" >&2; exit 2;; esac
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
GAMMA="$ROOT/data/SW_0090_large_unique_scale/gamma_train_70000.pt"
SOURCE="$ROOT/trained_models/SW0055_unique2500_s0_e10_lr0p0003/core.pt"
OUT="$ROOT/trained_models/SW0090_unique70000_s${SEED}_e10"
test -s "$GAMMA" -a -s "$SOURCE"
[[ ! -e "$OUT" ]] || { echo "refusing existing $OUT" >&2; exit 3; }
p="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; [[ ! "$p" =~ [0-9] ]]
mkdir -p "$OUT/cache" "$OUT/checkpoints"
export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
{
  printf 'experiment=SW0090_large_unique_scale\nseed=%s\ngpu=%s\n' "$SEED" "$GPU"
  printf 'gamma_sha256='; sha256sum "$GAMMA" | cut -d' ' -f1
  printf 'source_checkpoint_sha256='; sha256sum "$SOURCE" | cut -d' ' -f1
  printf 'trainer_sha256='; sha256sum "$ROOT/snn_kuramoto_bidirectional/training/train_s2net_core.py" | cut -d' ' -f1
} > "$OUT/manifest.txt"
/Data0/kevinswk/envs/snn/bin/python -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
 --gamma-seq-path "$GAMMA" --save-path "$OUT/core.pt" --num-regions 256 --num-feature-maps 8 \
 --device cuda --epochs 10 --batch-size 16 --lr 0.0003 --seed "$SEED" --graph-checkpoint "$SOURCE" --freeze-graph \
 --osc-dim 4 --gamma-drive-mode static --num-time-steps 64 --plv-settle 32 --theta-init gamma \
 --gamma-phase-mode standardize_tanh --freq-gain 2 --graph-mode learned --graph-top-k 32 \
 --graph-spatial-decay 0.35 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 \
 --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
 --spike-spatial-grid-size 16 --k 256 --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 \
 --low-n -4 --high-n 0 --branch 4 --gate-mode raw --plv-source phase --plv-combine mean \
 --spike-plv-weight 5 --loss-signal sigmoid_membrane --sample-activity-diversity-weight 0 \
 --checkpoint-dir "$OUT/checkpoints" --checkpoint-epochs 1 3 10 \
 --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 --structural-weight 0 \
 --plv-collapse-weight 1 --plv-bimodality-weight 6 --plv-balance-weight 10 \
 --plv-target-density 0.867 --plv-coherence-weight 0.5 --spike-per-component \
 --dendritic-projection shared --verbose > "$OUT/training.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python - "$SOURCE" "$OUT/core.pt" <<'PY'
import sys, torch
source=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
trained=torch.load(sys.argv[2],map_location='cpu',weights_only=True)
keys=[k for k in source if k.startswith('graph_generator.')]
assert keys and all(torch.equal(source[k],trained[k]) for k in keys)
assert any(not torch.equal(source[k],trained[k]) for k in source if not k.startswith('graph_generator.'))
print('FROZEN_GRAPH_EXACT',len(keys))
PY
test -s "$OUT/core.pt" -a -s "$OUT/checkpoints/epoch_01.pt" -a -s "$OUT/checkpoints/epoch_03.pt" -a -s "$OUT/checkpoints/epoch_10.pt"
echo completed > "$OUT/TRAINING_COMPLETED"


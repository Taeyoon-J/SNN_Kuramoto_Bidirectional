#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU required}"; SEED="${2:?seed 1 or 2 required}"; case "$GPU" in 0|1) ;; *) exit 2 ;; esac; case "$SEED" in 1|2) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; OUT="$ROOT/trained_models/SW0066_graphinit0_s${SEED}_e10"; GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
[[ ! -e "$OUT" ]] || { echo "existing $OUT" >&2; exit 3; }; PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; [[ ! "$PIDS" =~ [0-9] ]]
mkdir -p "$OUT/cache" "$OUT/checkpoints"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"; cd "$ROOT"
{
  printf 'experiment=SW0066_fixed_graph_initialization\nseed=%s\ngraph_init_seed=0\ngpu=%s\n' "$SEED" "$GPU"
  printf 'gamma_sha256='; sha256sum "$GAMMA" | cut -d' ' -f1
  printf 'trainer_sha256='; sha256sum "$ROOT/snn_kuramoto_bidirectional/training/train_s2net_core.py" | cut -d' ' -f1
  printf 'command_script_sha256='; sha256sum "$ROOT/collaborative_test/SW_0066_fixed_graph_initialization/run.sh" | cut -d' ' -f1
} > "$OUT/manifest.txt"
/Data0/kevinswk/envs/snn/bin/python -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
 --gamma-seq-path "$GAMMA" --save-path "$OUT/core.pt" --num-regions 256 --num-feature-maps 8 \
 --device cuda --epochs 10 --batch-size 16 --lr 0.0003 --seed "$SEED" --graph-init-seed 0 \
 --osc-dim 4 --gamma-drive-mode static --num-time-steps 64 --plv-settle 32 --theta-init gamma \
 --gamma-phase-mode standardize_tanh --freq-gain 2 --graph-mode learned --graph-top-k 32 \
 --graph-spatial-decay 0.35 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 \
 --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
 --spike-spatial-grid-size 16 --k 256 --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 \
 --low-n -4 --high-n 0 --branch 4 --gate-mode raw --plv-source phase --plv-combine mean \
 --spike-plv-weight 5 --loss-signal sigmoid_membrane --sample-activity-diversity-weight 0 \
 --checkpoint-dir "$OUT/checkpoints" --checkpoint-epochs 10 \
 --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 --structural-weight 0 \
 --plv-collapse-weight 1 --plv-bimodality-weight 6 --plv-balance-weight 10 \
 --plv-target-density 0.867 --plv-coherence-weight 0.5 --spike-per-component \
 --dendritic-projection shared --verbose > "$OUT/training.log" 2>&1
test -s "$OUT/core.pt" -a -s "$OUT/checkpoints/epoch_10.pt"; echo completed > "$OUT/TRAINING_COMPLETED"

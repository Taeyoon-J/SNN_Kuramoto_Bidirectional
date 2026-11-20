#!/bin/bash
# PV2_0120. Route C -- grid 64, one seed first.
#
# The last untested route in ROUTE_MAP_0.9_v3. Nine others are closed by measurement, and
# PV2_0119 showed why: the readout dial trades fg_ari against foreground IoU along the same
# axis PV2_0090 caught Slot Attention on, peaking at 0.7862 with 84% of the image called
# foreground. Raising the grid is the one untested way to move that axis rather than slide
# along it.
#
# Two things support it. Grid 16 -> 32 bought +0.1017 of actual pixel-protocol score and
# lifted the ceiling from 0.6973 to 0.8355 (recorded in BENCHMARK_PLAN.md). And patch
# purity, which PV2_0114 tied to the boundary misses, rises with the grid:
#   grid 16  0.9572      grid 32  0.9779      grid 64  0.9890
#
# ONE SEED FIRST. PV2_0087 prices a 4x finer grid at 18.6x the compute, so grid 32's 103
# minutes becomes roughly 32 hours per seed. Three seeds is a day and a half; a single seed
# says whether the direction is worth that.
#
# Hyperparameters carried from PV2_0094's finding, not guessed: --membrane-threshold-mode
# population, because --membrane-vth 0.06 is an ABSOLUTE threshold tuned for 256
# oscillators and PV2_0094 measured that it kills the spike path at 1024. At 4096 that
# argument is stronger, not weaker. --geodesic-chunk 8 from the memory probe, which
# cleared at peak 2.08 GB of 24.
#
# GATE, registered before the run: the three metrics at seed 0 against PV2_0104's grid-32
# seed 0 (0.7699 / 0.7531 / 0.7017 at pred_fg 0.1512), with predicted foreground at or
# below 0.20. A single seed cannot promote anything; it decides whether seeds 1 and 2 are
# worth 64 more GPU-hours.
#
# FALSIFIER: if seed 0 does not clear grid 32's seed 0 on at least matched-object IoU,
# route C closes and every route in ROUTE_MAP_0.9_v3 is closed by measurement.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G64=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid64.pt
T64=/work/USERS/tkim1/clevr/with_masks/targets_v3_grid64.pt
OUT=$R/pv2_0120_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0120: route C, grid 64, seed 0 ==="
say "grid 32 seed 0 (PV2_0104): .7699 / .7531 / .7017 at pred_fg .1512"
say "patch purity rises with the grid: 16 .9572  32 .9779  64 .9890"
say "GATE: beat grid 32 seed 0, obj_iou at minimum, pred_fg <= 0.20"
say "FALSIFIER: if it does not, route C closes and every route is closed"

t0=$(date +%s); mkdir -p $R/G64_s0
(cd /export_home/tkim1/snn_kuramoto_bidirectional && $PY training/train_s2net_core.py \
  --num-feature-maps 8 --num-regions 4096 --device cuda --batch-size 8 --lr 1e-3 \
  --osc-dim 4 --epochs 40 --seed 0 --gamma-drive-mode static --num-time-steps 64 \
  --plv-settle 32 --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
  --graph-mode learned --graph-top-k 32 --spike-spatial-grid-size 64 --k 4096 \
  --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 \
  --membrane-threshold-mode population --membrane-threshold-k 0.5 \
  --gate-mode raw --plv-source phase --plv-combine mean --loss-signal sigmoid_membrane \
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 \
  --structural-weight 0 --verbose --spike-per-component --geodesic-steps 3 \
  --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --spike-plv-weight 5.0 \
  --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0 --plv-balance-weight 10.0 \
  --plv-coherence-weight 0.5 --plv-target-density 0.867 --geodesic-chunk 8 \
  --train-limit 1000 --gamma-seq-path $G64 \
  --save-path $R/G64_s0/core.pt) > $R/G64_s0.log 2>&1
say "G64_s0 $(( ($(date +%s)-t0)/60 )) min  collapse $(grep -oE 'spike_plv_collapse=[0-9.]+' $R/G64_s0.log | tail -1 | cut -d= -f2)"
[ -s $R/G64_s0/core.pt ] || { say "TRAINING FAILED"; tail -8 $R/G64_s0.log >> $OUT; exit 1; }

out=$(cd $CT && $PY evaluate_model.py --checkpoint $R/G64_s0/core.pt \
  --gamma-seq-path $G64 --targets $T64 --manifest $MAN \
  --grid 64 --num-regions 4096 --split validation --limit 150 \
  --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
  --graph-spatial-decay 0.35 --geodesic-chunk 8 \
  --membrane-threshold-mode population --membrane-threshold-k 0.5 \
  --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
  --spatial-sigma 8.0 --min-group-size 4 2>&1)
r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
say ""
say "label                   fg_ari   fg_iou  obj_iou   groups  pred_fg"
if [ -z "$r" ]; then say "EVALUATION FAILED"; echo "$out" | tail -8 >> $OUT
else printf "%-22s %s\n" "grid 64, seed 0" "$r" | tee -a $OUT; fi
say "=== PV2_0120 DONE ==="

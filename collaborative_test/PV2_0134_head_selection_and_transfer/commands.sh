#!/bin/bash
# PV2_0134 -- reproduce every number in README.md. Run from /work/USERS/tkim1/sw0255b.
# Environment: /work/USERS/tkim1/envs/miniforge3/envs/snn (torch), .../envs/tfgpu (tf 2.15.1).
# GPUs 2 and 3 only; 0 and 1 belong to the collaborator.
B=/work/USERS/tkim1/sw0255b
PY=$B/../envs/miniforge3/envs/snn/bin/python
ARGS="--code $B/benchmark_codes --config $B/benchmark_codes/configs/frontier.json --run $B/run"

# ---- change C: cross-seed agreement, with and without collapse guards (negative result)
# needs the second-seed cache written by cache_seed2.py (features_b / gamma_b / regions_b)
CUDA_VISIBLE_DEVICES=2 $PY $B/benchmark_codes/fit4.py $ARGS --dataset clevr \
  --train-cache $B/ev/clevr_train --val-cache $B/ev/clevr_validation \
  --eval-cache $B/ev/clevr_fixed320 --eval-split fixed320 \
  --epochs 20 --val-every 1 --out $B/fit4/clevr.json
# tetrominoes / multi_dsprites: same, with --eval-cache $B/ev/<d>_test --eval-split test

# ---- B': validation FG-ARI with a Foreground IoU floor, epoch 0 a candidate
for d in clevr tetrominoes multi_dsprites; do
  case $d in clevr) E=clevr_fixed320; S=fixed320;; *) E=${d}_test; S=test;; esac
  CUDA_VISIBLE_DEVICES=2 $PY $B/benchmark_codes/fit5.py $ARGS --dataset $d --regions 253 \
    --train-cache $B/ev/${d}_train --val-cache $B/ev/${d}_validation \
    --eval-cache $B/ev/$E --eval-split $S \
    --epochs 20 --val-every 1 --iou-floor 0.01 --out $B/fit5/$d.json
done

# ---- transfer: the CLEVR head, unchanged, on the other datasets
for d in tetrominoes multi_dsprites; do
  CUDA_VISIBLE_DEVICES=2 $PY $B/benchmark_codes/xfer.py $ARGS \
    --checkpoint $B/fit5/clevr_253_local_rgb_tv.pt \
    --dataset $d --eval-cache $B/ev/${d}_test --eval-split test --out $B/fit5/xfer_$d.json
done

# ---- recipe: the CLEVR arm and epoch reused, no labels from the new dataset
for d in clevr tetrominoes multi_dsprites; do
  case $d in clevr) E=clevr_fixed320; S=fixed320;; *) E=${d}_test; S=test;; esac
  CUDA_VISIBLE_DEVICES=2 $PY $B/benchmark_codes/recipe.py $ARGS --dataset $d \
    --train-cache $B/ev/${d}_train --eval-cache $B/ev/$E --eval-split $S \
    --arm local_rgb_tv --epochs 11 --out $B/fit5/recipe_$d.json
done

# ---- Slot Attention: dump frames (torch env), infer (tf env, CPU), choose the rule
for s in validation fixed320; do
  $PY $B/benchmark_codes/sa_dump.py $ARGS --split $s --out $B/sa/frames_$s.npz
done
SA=/work/USERS/tkim1/baselines/slot_attention
for s in validation fixed320; do
  CUDA_VISIBLE_DEVICES="" $B/../envs/miniforge3/envs/tfgpu/bin/python \
    $B/benchmark_codes/sa_infer.py --frames $B/sa/frames_$s.npz \
    --checkpoint-dir $SA --model-dir $SA --grid 64 --num-slots 7 --seed 0 \
    --out $B/sa/slots64_$s.npz
done
$PY $B/benchmark_codes/sa_bg.py --val $B/sa/slots64_validation.npz \
  --eval $B/sa/slots64_fixed320.npz $ARGS --out $B/sa/bg_rules.json

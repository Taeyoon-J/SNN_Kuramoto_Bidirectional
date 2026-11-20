#!/bin/bash
# The project's first genuine 3-seed table, from cores already on disk.
#
# /work/USERS/tkim1/sw0255/run/jobs/model253/<ds>/seed{0,1,2} holds three real training seeds
# per dataset at 3,500 unique x 10 epochs (2,190 updates). That is NOT the headline claim --
# the headline is one CLEVR-trained core reused elsewhere, which exists at seed 0 only -- but
# it is a real 3-seed set for a clearly different claim: a core trained on its own dataset.
#
# Scoring it produces per-seed values, means and standard deviations for all three metrics,
# which is the reporting format the contract demands and which this project has never yet
# produced. Run it AFTER the 261 cores finish; it would otherwise contend for the same GPUs.
set -u
B=/work/USERS/tkim1/sw0255b
OLD=/work/USERS/tkim1/sw0255/run
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
GPU=$1
export CUDA_VISIBLE_DEVICES=$GPU
CFG="--code $B/benchmark_codes --config $B/benchmark_codes/configs/frontier.json"

if ! $PY $B/benchmark_codes/cache_any.py --help 2>&1 | grep -q -- "--job"; then
  echo "ABORT: cache_any.py has no --job flag; it would cache the CLEVR core."
  exit 1
fi

for DS in tetrominoes multi_dsprites; do
  for S in 0 1 2; do
    J=$OLD/jobs/model253/$DS/seed$S
    [ -f $J/resume.pt ] || { echo "ABORT: no checkpoint at $J"; exit 1; }
    for sp in train validation test; do
      case $sp in train) N=256;; validation) N=160;; test) N=320;; esac
      O=$B/ev3500/${DS}_s${S}_$sp
      [ -d $O ] && continue
      $PY $B/benchmark_codes/cache_any.py $CFG --run $B/run --job $J \
          --dataset $DS --split $sp --count $N --min-group 8 --out $O || exit 1
    done
    mkdir -p $B/fit3500
    $PY $B/benchmark_codes/recipe.py $CFG --run $B/run --dataset $DS \
        --train-cache $B/ev3500/${DS}_s${S}_train \
        --eval-cache $B/ev3500/${DS}_s${S}_test --eval-split test \
        --arm local_rgb_trust --epochs 2 \
        --out $B/fit3500/${DS}_s${S}.json || exit 1
  done
  $PY $B/benchmark_codes/agg3.py \
      --results $B/fit3500/${DS}_s0.json $B/fit3500/${DS}_s1.json $B/fit3500/${DS}_s2.json \
      --label "253 core trained on $DS at 3500x10, plus the 261 readout (label-free recipe)" \
      --key head_recipe --expect-seeds 3 --out $B/fit3500/${DS}_3seed.json || exit 1
done
echo SEEDS3_DONE

#!/bin/bash
# Everything after the 261 core finishes, chained so the pipeline completes unattended.
# Launch detached per dataset:  nohup bash chain261.sh tetrominoes 2 > chain261_tetrominoes.log 2>&1 &
#
# Waits for the core, caches activity, fits the 261 readout, scores on the SAME 320 evaluation
# images the existing transfer rows use so the two columns differ only in what the core trained on.
set -u
B=/work/USERS/tkim1/sw0255b
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
DS=$1; GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU
RUN=$B/run_261_$DS
CFG="--code $B/benchmark_codes --config $B/benchmark_codes/configs/frontier.json"

# 1. wait for the core. CORE261_<ds>_DONE is written by go261.sh; also stop if it died.
until grep -q "CORE261_${DS}_DONE" $B/core261_$DS.log 2>/dev/null; do
  if ! pgrep -f "run_261_$DS train" > /dev/null; then
    if ! grep -q "CORE261_${DS}_DONE" $B/core261_$DS.log 2>/dev/null; then
      echo "CHAIN_ABORT: the $DS core is no longer running and never reported done"; exit 1
    fi
  fi
  sleep 120
done
if ! grep -q "CORE261_${DS}_DONE rc=0" $B/core261_$DS.log; then
  echo "CHAIN_ABORT: the $DS core exited nonzero"; tail -5 $B/core261_$DS.log; exit 1
fi
echo "core done for $DS, caching"

# 2. cache activity from THIS dataset's own core.
#
# PRECONDITION, and the one that can silently ruin this experiment. cache_seed2.py hardcodes
#   job = Path(a.run)/'jobs/model253/clevr/seed0'
# If cache_any.py does the same, it would load the CLEVR core no matter what --run says, cache
# CLEVR-core activity under a tetrominoes name, and the comparison would be measuring nothing.
# So refuse to run unless cache_any.py exposes a way to point at this dataset's own checkpoint.
if ! $PY $B/benchmark_codes/cache_any.py --help 2>&1 | grep -q -- "--job"; then
  echo "CHAIN_ABORT: cache_any.py has no --job flag, so it probably hardcodes"
  echo "  jobs/model253/clevr/seed0 and would cache the CLEVR core instead of the $DS core."
  echo "  Parameterize the job path in cache_any.py first, then rerun this chain."
  $PY $B/benchmark_codes/cache_any.py --help 2>&1 | head -20
  exit 1
fi
for sp in train validation test; do
  case $sp in train) N=256;; *) N=320;; esac
  $PY $B/benchmark_codes/cache_any.py $CFG --run $RUN --job $RUN/jobs/model253/$DS/seed0 \
      --dataset $DS --split $sp --count $N --min-group 8 \
      --out $B/ev261/${DS}_$sp || { echo "CHAIN_ABORT: caching $sp failed"; exit 1; }
done

# 3. fit the 261 readout under rule B', and also the fixed label-free recipe
mkdir -p $B/fit261
$PY $B/benchmark_codes/fit5.py $CFG --run $RUN --dataset $DS --regions 253 \
    --train-cache $B/ev261/${DS}_train --val-cache $B/ev261/${DS}_validation \
    --eval-cache $B/ev261/${DS}_test --eval-split test \
    --epochs 20 --val-every 1 --iou-floor 0.01 --out $B/fit261/$DS.json || exit 1
$PY $B/benchmark_codes/recipe.py $CFG --run $RUN --dataset $DS \
    --train-cache $B/ev261/${DS}_train --eval-cache $B/ev261/${DS}_test --eval-split test \
    --arm local_rgb_trust --epochs 2 --out $B/fit261/recipe_$DS.json || exit 1

echo "CHAIN261_${DS}_DONE"

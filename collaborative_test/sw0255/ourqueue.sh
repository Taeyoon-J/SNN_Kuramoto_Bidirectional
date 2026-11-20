#!/bin/bash
# Our queue over the SW0255 package: one worker per GPU, our order, our GPUs only.
#
# We do not use the package's own `queue` because it requires preflight on all five
# datasets and takes its order from the peer's priority file (252 -> 253 -> 254 -> 240).
# We agreed to run the reverse so the two of us meet in the middle rather than duplicating.
#
# Order is 253 first, on both datasets, all three seeds. Reasons:
#   - 253 is the only unsupervised model of the four. Their own scheduler asserts it:
#     report['mask_opened_during_training'] == (model != 253). It is the only candidate
#     for a table beside CAE, CtCAE, Rotating Features, SynCx and Slot Attention.
#   - the peer is running 253 on CLEVR only; Tetrominoes and Multi-dSprites are empty,
#     and those two are ours under the agreed split.
#   - 253 is grid 32 against the others' grid 64, so it costs about half: measured 36 s per
#     update on their side against 75, and 53 on ours.
#   - 240 and 254 are both mask-supervised, so they fill the ablation table, which the
#     peer's 252 is already filling six cells of.
#
# GPUs 2 and 3 only. 0 and 1 belong to the collaborator.
# Each job claims a lock file; a worker skips claimed or completed jobs. The package itself
# resumes from its own saved checkpoint if a job is restarted.
set -u
BC=/work/USERS/tkim1/sw0255/benchmark_codes
OUT=/work/USERS/tkim1/sw0255/run
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
Q=/work/USERS/tkim1/sw0255/queue
mkdir -p $Q
GPU=$1

JOBS="
253 tetrominoes 0
253 multi_dsprites 0
253 tetrominoes 1
253 multi_dsprites 1
253 tetrominoes 2
253 multi_dsprites 2
240 tetrominoes 0
254 tetrominoes 0
240 multi_dsprites 0
254 multi_dsprites 0
240 tetrominoes 1
254 tetrominoes 1
"

echo "[gpu$GPU] worker up $(date '+%m-%d %H:%M')"
while :; do
  picked=""
  while read -r m d s; do
    [ -z "$m" ] && continue
    tag="${m}_${d}_${s}"
    [ -f "$OUT/jobs/model$m/$d/seed$s/complete.json" ] && continue
    # atomic claim
    if ( set -o noclobber; echo "gpu$GPU $$ $(date -Is)" > "$Q/$tag.claim" ) 2>/dev/null; then
      picked="$m $d $s"; break
    fi
  done <<< "$JOBS"

  if [ -z "$picked" ]; then echo "[gpu$GPU] nothing left to claim, exiting $(date '+%m-%d %H:%M')"; break; fi

  set -- $picked; m=$1; d=$2; s=$3
  echo "[gpu$GPU] START model$m $d seed$s  $(date '+%m-%d %H:%M')"
  t0=$(date +%s)
  ( cd $BC && CUDA_VISIBLE_DEVICES=$GPU $PY -u -B run.py --config configs/frontier.json \
      --output $OUT train --model $m --dataset $d --seed $s ) > $Q/${m}_${d}_${s}.log 2>&1
  rc=$?
  echo "[gpu$GPU] END   model$m $d seed$s rc=$rc  $(( ($(date +%s)-t0)/60 )) min  $(date '+%m-%d %H:%M')"
  if [ $rc -ne 0 ]; then
    echo "[gpu$GPU] failed, leaving the claim in place so it is not retried blindly; see $Q/${m}_${d}_${s}.log"
    tail -5 $Q/${m}_${d}_${s}.log
  fi
done

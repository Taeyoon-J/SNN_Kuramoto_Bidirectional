#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?Pass seed 0, 1, or 2}"
MODE="${2:-full}"
[[ "$SEED" =~ ^[0-2]$ ]] || { echo "seed must be 0, 1, or 2" >&2; exit 2; }
[[ "$MODE" == full || "$MODE" == smoke || "$MODE" == dry-run ]] || { echo "mode must be full, smoke, or dry-run" >&2; exit 2; }

ROOT=/Data0/kevinswk/patch_v2_sw
RUN_DIR="$ROOT/collaborative_test/SW_0056_matched_slot_2500"
MODEL_PY=/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002/model.py
DATASET=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
TF_PY=/Data0/kevinswk/envs/slot_attention_eval_tf215/bin/python
SNN_PY=/Data0/kevinswk/envs/snn/bin/python
OUT="$ROOT/trained_models/SW0056_matched_slot_2500_seed${SEED}"
if [[ "$MODE" == smoke ]]; then OUT="$OUT-smoke"; fi
if [[ "$MODE" == dry-run ]]; then
  printf 'DRY RUN: seed=%s train_ids=0-999,1640-3139 val_ids=1320-1639 exposures=25000 batch=16 updates=1563 final_batch=8 CPU-only\n' "$SEED"
  exit 0
fi
test -s "$MODEL_PY" && test -s "$DATASET" && test -x "$TF_PY" && test -x "$SNN_PY"

export CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SW0056_GATE_PYTHON="$TF_PY"
mkdir -p "$(dirname "$OUT")" "$ROOT/trained_models/SW0056_launcher_logs"
LOG_DIR="$ROOT/trained_models/SW0056_launcher_logs"
VAL_OUT="$OUT/validation1320_1639"

mark_failed() {
  local phase rc log
  phase="$1"; rc="$2"; log="$3"
  [[ ! -e "$OUT/FAILED" ]] || return 0
  (
    set -o noclobber
    printf 'phase=%s\n' "$phase"
    printf 'rc=%s\n' "$rc"
    printf 'timestamp_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'log=%s\n' "$log"
  ) > "$OUT/FAILED" 2>/dev/null || true
}

wait_for_load_if_requested() {
  if [[ "${SW0056_REQUIRE_LOW_LOAD:-0}" == 1 ]]; then
    # shellcheck source=load_gate.sh
    source "$RUN_DIR/load_gate.sh"
    wait_for_low_load 10 60
  fi
}

run_training() {
  local smoke_flag log_path rc validate_mode
  smoke_flag=()
  validate_mode=training
  if [[ "$MODE" == smoke ]]; then validate_mode=smoke; fi
  [[ "$MODE" == smoke ]] && smoke_flag=(--smoke)
  log_path="$LOG_DIR/SW0056_seed${SEED}_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
  [[ ! -e "$log_path" ]] || { echo "Launcher log collision: $log_path" >&2; return 9; }
  wait_for_load_if_requested
  if nice -n 10 "$TF_PY" -u "$RUN_DIR/train_seed.py" --dataset "$DATASET" \
      --model-py "$MODEL_PY" --output-dir "$OUT" --seed "$SEED" "${smoke_flag[@]}" \
      > "$log_path" 2>&1; then
    :
  else
    rc=$?
    mark_failed "$MODE-training" "$rc" "$log_path"
    echo "Training failed; preserving partial output and external log $log_path" >&2
    return "$rc"
  fi
  if "$TF_PY" "$RUN_DIR/validate_phase.py" "$validate_mode" "$OUT" "$SEED"; then
    if [[ -e "$OUT/training.log" ]]; then
      echo "training.log already exists; refusing to replace" >&2
      mark_failed "$MODE-training-log-finalize" 10 "$log_path"
      return 10
    fi
    if mv -n -- "$log_path" "$OUT/training.log" && [[ ! -e "$log_path" && -s "$OUT/training.log" ]]; then
      return 0
    else
      rc=$?
      mark_failed "$MODE-training-log-finalize" "$rc" "$log_path"
      return "$rc"
    fi
  else
    rc=$?
    mark_failed "$MODE-training-validation" "$rc" "$log_path"
    echo "Training artifacts invalid; preserving $log_path" >&2
    return "$rc"
  fi
}

restore_completed_training_log() {
  local candidates=()
  shopt -s nullglob
  candidates=("$LOG_DIR/SW0056_seed${SEED}_full_"*.log)
  shopt -u nullglob
  if [[ ! -e "$OUT/training.log" ]]; then
    if [[ ${#candidates[@]} -eq 1 ]]; then
      mv -n -- "${candidates[0]}" "$OUT/training.log"
      [[ ! -e "${candidates[0]}" && -s "$OUT/training.log" ]]
    elif [[ ${#candidates[@]} -gt 1 ]]; then
      echo "Multiple external full-training logs; manual recovery required" >&2
      return 1
    fi
  fi
}

if [[ "$MODE" == smoke ]]; then
  if [[ -e "$OUT/FAILED" ]]; then echo "Existing failure marker; manual review required" >&2; exit 1; fi
  if [[ -e "$OUT/TRAINING_COMPLETED" ]]; then
    "$TF_PY" "$RUN_DIR/validate_phase.py" smoke "$OUT" "$SEED"
    exit 0
  fi
  if [[ -d "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Partial smoke output exists; refusing to rerun: $OUT" >&2
    [[ -e "$OUT/FAILED" ]] || mark_failed smoke-recovery 11 unknown
    exit 1
  fi
  mkdir -p "$OUT"
  run_training
  echo "SW0056 CPU smoke validated: $OUT"
  exit 0
fi

if [[ -e "$OUT/FAILED" ]]; then
  echo "Existing FAILED marker; preserving output for manual review: $OUT/FAILED" >&2
  exit 1
fi
if [[ -e "$OUT/COMPLETED" ]]; then
  if "$SNN_PY" "$RUN_DIR/validate_phase.py" complete "$VAL_OUT" "$SEED" --training-protocol "$OUT/training_protocol.json"; then
    echo "SW0056 seed $SEED already valid and complete: $OUT"
    exit 0
  fi
  echo "COMPLETED exists but validation failed; refusing modification" >&2
  exit 1
fi

if [[ -e "$OUT/TRAINING_COMPLETED" ]]; then
  if "$TF_PY" "$RUN_DIR/validate_phase.py" training "$OUT" "$SEED"; then
    restore_completed_training_log || { mark_failed training-log-recovery 12 unknown; exit 1; }
    echo "Valid training artifacts found; skipping training for seed$SEED"
  else
    mark_failed training-recovery-validation 13 unknown
    echo "Invalid completed training artifacts; preserving output and stopping" >&2
    exit 1
  fi
else
  if [[ -d "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    mark_failed training-partial-recovery 14 unknown
    echo "Partial/invalid training output; automatic retraining refused" >&2
    exit 1
  fi
  mkdir -p "$OUT"
  run_training
fi

if [[ -e "$VAL_OUT/INFERENCE_COMPLETED" ]]; then
  if "$TF_PY" "$RUN_DIR/validate_phase.py" inference "$VAL_OUT" "$SEED" --training-protocol "$OUT/training_protocol.json"; then
    echo "Valid inference artifacts found; skipping inference for seed$SEED"
  else
    mark_failed inference-recovery-validation 15 "$VAL_OUT"
    echo "Invalid completed inference artifacts; preserving and stopping" >&2
    exit 1
  fi
else
  for artifact in predictions.npz protocol.json INFERENCE_COMPLETED FAILED inference.log evaluation_summary.json per_image.csv patch_masks.pt scoring.log SCORING_COMPLETED; do
    if [[ -e "$VAL_OUT/$artifact" ]]; then
      mark_failed inference-partial-recovery 16 "$VAL_OUT/$artifact"
      echo "Partial validation output; refusing to overwrite: $VAL_OUT/$artifact" >&2
      exit 1
    fi
  done
  mkdir -p "$VAL_OUT"
  wait_for_load_if_requested
  if nice -n 10 "$TF_PY" -u "$ROOT/collaborative_test/SW_0046_aligned_slot_audit/slot_attention_checkpoint_predict.py" \
      --checkpoint-dir "$OUT/checkpoint" --checkpoint-prefix ckpt-1563 \
      --checkpoint-source "SW0056 scratch matched-data 10-pass budget" \
      --training-seed "$SEED" --inference-seed 0 --training-protocol "$OUT/training_protocol.json" \
      --tf-intra-threads 2 --tf-inter-threads 1 --model-py "$MODEL_PY" --dataset "$DATASET" \
      --start 1320 --count 320 --output-dir "$VAL_OUT" > "$VAL_OUT/inference.log" 2>&1; then
    :
  else
    rc=$?; mark_failed inference "$rc" "$VAL_OUT/inference.log"; echo "Inference failed; partial output preserved" >&2; exit "$rc"
  fi
  if "$TF_PY" "$RUN_DIR/validate_phase.py" inference "$VAL_OUT" "$SEED" --training-protocol "$OUT/training_protocol.json"; then :
  else rc=$?; mark_failed inference-validation "$rc" "$VAL_OUT"; exit "$rc"; fi
fi

if [[ -e "$VAL_OUT/SCORING_COMPLETED" ]]; then
  if "$SNN_PY" "$RUN_DIR/validate_phase.py" scoring "$VAL_OUT" "$SEED"; then
    echo "Valid scoring artifacts found; skipping scoring for seed$SEED"
  else
    mark_failed scoring-recovery-validation 17 "$VAL_OUT"
    echo "Invalid completed scoring artifacts; preserving and stopping" >&2
    exit 1
  fi
else
  for artifact in evaluation_summary.json per_image.csv patch_masks.pt SCORING_COMPLETED scoring.log; do
    if [[ -e "$VAL_OUT/$artifact" ]]; then
      mark_failed scoring-partial-recovery 18 "$VAL_OUT/$artifact"
      echo "Partial scoring output; refusing to overwrite: $VAL_OUT/$artifact" >&2
      exit 1
    fi
  done
  wait_for_load_if_requested
  if nice -n 10 "$SNN_PY" -u "$ROOT/collaborative_test/SW_0046_aligned_slot_audit/score_predictions.py" \
      --predictions "$VAL_OUT/predictions.npz" --protocol "$VAL_OUT/protocol.json" \
      --dataset "$DATASET" --start 1320 --count 320 --output-dir "$VAL_OUT" > "$VAL_OUT/scoring.log" 2>&1; then
    :
  else
    rc=$?; mark_failed scoring "$rc" "$VAL_OUT/scoring.log"; echo "Scoring failed; partial output preserved" >&2; exit "$rc"
  fi
  if "$SNN_PY" "$RUN_DIR/validate_phase.py" scoring "$VAL_OUT" "$SEED"; then :
  else rc=$?; mark_failed scoring-validation "$rc" "$VAL_OUT"; exit "$rc"; fi
fi

if "$SNN_PY" "$RUN_DIR/validate_phase.py" complete "$VAL_OUT" "$SEED" --training-protocol "$OUT/training_protocol.json"; then
  :
else
  rc=$?; mark_failed final-validation "$rc" "$VAL_OUT"; exit "$rc"
fi
[[ ! -e "$OUT/COMPLETED" ]] || { echo "Existing COMPLETED marker after invalid state" >&2; exit 1; }
printf 'seed=%s completed_utc=%s\n' "$SEED" "$(date -u +%FT%TZ)" > "$OUT/COMPLETED"
echo "SW0056 seed $SEED complete: $OUT"

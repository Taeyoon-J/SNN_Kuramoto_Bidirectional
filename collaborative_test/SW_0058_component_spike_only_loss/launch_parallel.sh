#!/usr/bin/env bash
set -euo pipefail
DRY=0
if [[ "${1:-}" == --dry-run ]]; then DRY=1; shift; fi
GPU0="${1:?GPU0 required}"; GPU1="${2:?GPU1 required}"
[[ "$GPU0" =~ ^[01]$ && "$GPU1" =~ ^[01]$ && "$GPU0" != "$GPU1" ]] || { echo "Use distinct GPU0 and GPU1" >&2; exit 2; }
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0058_component_spike_only_loss"
PY=/Data0/kevinswk/envs/snn/bin/python
MODELROOT="$ROOT/trained_models"; STATE="$MODELROOT/SW0058_three_seed_launcher"
LOCK="$MODELROOT/SW0058_three_seed_launcher.lock"
SUMMARY="$DIR/results/sw0058_three_seed_comparison.json"
PILOT="$ROOT/collaborative_test/SW_0054_causal_mechanism_ablation/results"
GAMMA=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/gamma_train.pt
PREFLIGHT="$MODELROOT/SW0058_preflight_v5/PREFLIGHT.json"
mkdir -p "$DIR/results"
"$PY" "$DIR/pilot_gate.py" --folder "$PILOT"
test -s "$GAMMA"
CHECKSUM="$(sha256sum "$GAMMA" | awk '{print $1}')"
CODE_SUM="$(sha256sum "$DIR/preflight.sh" "$DIR/preflight_one_update.py" "$DIR/pilot_gate.py" "$DIR/run.sh" \
  "$DIR/evaluate.sh" "$DIR/evaluate_multireadout.sh" "$DIR/validate_result.py" "$DIR/summarize.py" "$DIR/launch_parallel.sh" \
  "$ROOT/snn_kuramoto_bidirectional/training/train_s2net_core.py" \
  "$ROOT/snn_kuramoto_bidirectional/loss_function.py" "$ROOT/snn_kuramoto_bidirectional/s2net_cls.py" \
  "$ROOT/snn_kuramoto_bidirectional/hyperparameter.py" "$ROOT/snn_kuramoto_bidirectional/sc_generator.py" \
  "$ROOT/snn_kuramoto_bidirectional/graph_generator.py" "$ROOT/snn_kuramoto_bidirectional/kuramoto_layer.py" \
  "$ROOT/snn_kuramoto_bidirectional/dendric_layer.py" "$ROOT/snn_kuramoto_bidirectional/membrane_layer.py" \
  "$ROOT/snn_kuramoto_bidirectional/sinusoidal_gating.py" "$ROOT/snn_kuramoto_bidirectional/spike_classifier.py" \
  "$ROOT/snn_kuramoto_bidirectional/image_conditioned_sc.py" | sha256sum | awk '{print $1}')"
current_preflight() {
  "$PY" - "$PREFLIGHT" "$CHECKSUM" "$CODE_SUM" <<'PY'
import json,sys
try: m=json.load(open(sys.argv[1],encoding='utf-8'))
except Exception: raise SystemExit(1)
raise SystemExit(0 if m.get('version')==5 and m.get('passed') is True and m.get('gamma_sha256')==sys.argv[2] and m.get('code_sha256')==sys.argv[3] else 1)
PY
}
if [[ -e "$PREFLIGHT" ]] && ! current_preflight; then echo "Stale preflight; manual refresh required" >&2; exit 3; fi
for seed in 0 1 2; do
  out="$MODELROOT/SW_0058_component_spike_only_s${seed}_lr0p0003_epoch25"
  [[ ! -e "$out" ]] || { echo "Existing/partial output: $out" >&2; exit 3; }
done
[[ ! -e "$STATE" && ! -e "$LOCK" && ! -e "$SUMMARY" && ! -e "${SUMMARY%.json}.md" ]] || { echo "Existing launcher state/summary" >&2; exit 3; }
if (( DRY )); then
  echo "DRY-RUN: SW0054 pilot validated; preflight then seeds0/1 on GPU$GPU0/GPU$GPU1, seed2 on GPU$GPU0, fixed-threshold summary."
  exit 0
fi
gpu_idle() {
  local pids
  pids="$(nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader)" || return 1
  [[ ! "$pids" =~ [0-9] ]]
}
gpu_idle "$GPU0" && gpu_idle "$GPU1" || { echo "Both GPUs must be idle" >&2; exit 2; }
mkdir "$LOCK" 2>/dev/null || { echo "Launcher lock exists; inspect before retry" >&2; exit 3; }
mkdir "$STATE"
printf 'pid=%s\nstarted_utc=%s\n' "$$" "$(date -u +%FT%TZ)" > "$LOCK/owner"
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
fail() { local phase="$1" rc="$2"; printf 'phase=%s\nrc=%s\nfailed_utc=%s\n' "$phase" "$rc" "$(date -u +%FT%TZ)" > "$STATE/FAILED"; exit "$rc"; }
if ! current_preflight; then
  gpu_idle "$GPU0" || fail preflight_gpu_busy 4
  if bash "$DIR/preflight.sh" "$GPU0"; then :; else rc=$?; fail preflight "$rc"; fi
  current_preflight || fail preflight_marker 5
fi
gpu_idle "$GPU0" && gpu_idle "$GPU1" || fail prelaunch_gpu_race 6
bash "$DIR/run.sh" "$GPU0" 0 > "$STATE/seed0.log" 2>&1 & p0=$!
bash "$DIR/run.sh" "$GPU1" 1 > "$STATE/seed1.log" 2>&1 & p1=$!
rc0=0; wait "$p0" || rc0=$?
rc1=0; wait "$p1" || rc1=$?
(( rc0 == 0 )) || fail seed0 "$rc0"
(( rc1 == 0 )) || fail seed1 "$rc1"
for seed in 0 1; do test -s "$MODELROOT/SW_0058_component_spike_only_s${seed}_lr0p0003_epoch25/EVALUATION_COMPLETED" || fail "seed${seed}_completion" 7; done
gpu_idle "$GPU0" || fail seed2_gpu_busy 8
if bash "$DIR/run.sh" "$GPU0" 2 > "$STATE/seed2.log" 2>&1; then :; else rc=$?; fail seed2 "$rc"; fi
test -s "$MODELROOT/SW_0058_component_spike_only_s2_lr0p0003_epoch25/EVALUATION_COMPLETED" || fail seed2_completion 9
if "$PY" "$DIR/summarize.py" --model-root "$MODELROOT" --output "$SUMMARY"; then :; else rc=$?; fail summary "$rc"; fi
"$PY" - "$SUMMARY" <<'PY' || fail summary_validation 10
import json,math,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
assert r['seeds']==[0,1,2] and r['validation']['threshold_postselection'] is False
for k in ('fg_ari','foreground_iou','matched_object_iou'): assert math.isfinite(r['primary_spike_cc'][k]['candidate_mean'])
PY
test -s "${SUMMARY%.json}.md" || fail summary_markdown 11
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/LAUNCH_COMPLETED"
echo "SW0058 three-seed training/evaluation/summary completed."

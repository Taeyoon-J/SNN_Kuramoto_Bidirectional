#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU ID required (0 or 1)}"
SEED="${2:?seed 0, 1, or 2 required}"
case "$GPU" in 0|1) ;; *) echo "SW0058 uses GPU0/1 only" >&2; exit 2 ;; esac
case "$SEED" in 0|1|2) ;; *) echo "SEED must be 0, 1, or 2" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0058_component_spike_only_loss"
PY=/Data0/kevinswk/envs/snn/bin/python
GAMMA=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/gamma_train.pt
PREFLIGHT="$ROOT/trained_models/SW0058_preflight_v5/PREFLIGHT.json"
PILOT="$ROOT/collaborative_test/SW_0054_causal_mechanism_ablation/results"
OUT="$ROOT/trained_models/SW_0058_component_spike_only_s${SEED}_lr0p0003_epoch25"
test -s "$GAMMA"
"$PY" "$DIR/pilot_gate.py" --folder "$PILOT"
[[ ! -e "$OUT" ]] || { echo "Refusing existing/partial output: $OUT" >&2; exit 3; }
if ! PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; then
  echo "Cannot inspect GPU $GPU: $PIDS" >&2; exit 2
fi
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU occupied; refusing training" >&2; exit 2; }
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
PILOT_RAW_SHA="$(sha256sum "$PILOT/seed0_epoch25_short_n32_pilot_T256_settle64.json" | awk '{print $1}')"
PILOT_SUMMARY_SHA="$(sha256sum "$PILOT/seed0_epoch25_short_n32_pilot_summary.json" | awk '{print $1}')"
PILOT_VALIDATOR_SHA="$(sha256sum "$ROOT/collaborative_test/SW_0054_causal_mechanism_ablation/summarize.py" | awk '{print $1}')"
"$PY" - "$PREFLIGHT" "$CHECKSUM" "$CODE_SUM" "$PILOT_RAW_SHA" "$PILOT_SUMMARY_SHA" "$PILOT_VALIDATOR_SHA" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding='utf-8'))
assert m.get('version')==5 and m.get('passed') is True
assert m.get('gamma_sha256')==sys.argv[2] and m.get('code_sha256')==sys.argv[3]
assert m.get('sw0054_raw_sha256')==sys.argv[4] and m.get('sw0054_summary_sha256')==sys.argv[5]
assert m.get('sw0054_validator_sha256')==sys.argv[6]
PY
mkdir -p "$OUT"
ARGS=(
  --gamma-seq-path "$GAMMA" --save-path "$OUT/core.pt"
  --num-regions 256 --num-feature-maps 8 --device cuda
  --epochs 25 --batch-size 16 --lr 0.0003 --seed "$SEED" --osc-dim 4
  --gamma-drive-mode static --num-time-steps 64 --plv-settle 32
  --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0
  --graph-mode learned --graph-top-k 32 --graph-spatial-decay 0.35
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized
  --spike-spatial-grid-size 16 --k 256 --membrane-vth 0.06
  --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0
  --branch 4 --gate-mode raw --plv-source phase --plv-combine mean
  --primary-loss-weight 0 --spike-plv-weight 5.0 --loss-signal sigmoid_membrane
  --sample-activity-diversity-weight 0
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0
  --structural-weight 0 --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0
  --plv-balance-weight 10.0 --plv-target-density 0.867 --plv-coherence-weight 0.5
  --spike-per-component --dendritic-projection shared --verbose
)
"$PY" - "$OUT/manifest.json" "$GAMMA" "$CHECKSUM" "$CODE_SUM" "$SEED" "${ARGS[@]}" <<'PY'
import json,sys
manifest,gamma,gamma_sha,code_sha,seed,*argv=sys.argv[1:]
record={"schema_version":1,"experiment":"SW0058 component-spike-only loss",
 "seed":int(seed),"gamma":{"path":gamma,"sha256":gamma_sha,"train_ids":[0,999]},
 "pilot":{"raw_sha256":__import__('hashlib').sha256(open('/Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0054_causal_mechanism_ablation/results/seed0_epoch25_short_n32_pilot_T256_settle64.json','rb').read()).hexdigest(),
 "summary_sha256":__import__('hashlib').sha256(open('/Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0054_causal_mechanism_ablation/results/seed0_epoch25_short_n32_pilot_summary.json','rb').read()).hexdigest()},
 "validation_ids":[1320,1639],"baseline":{"experiment":"SW0053 mixed loss",
 "primary_loss_weight":1.0,"spike_plv_weight":5.0,"lr":0.0003,"epochs":25,
 "source_reports":["SW_0053_lr_early_stop_matrix/results/seed0_low_lr",
 "SW_0053_lr_early_stop_matrix/results/seed1_low_lr",
 "SW_0052_checkpoint_trajectory/results/low_lr_epoch25_long_T1024_settle512.json"]},
 "intervention":{"only_loss_change":"primary_loss_weight 1.0 -> 0.0",
 "primary_loss_weight":0.0,"spike_plv_weight":5.0},
 "code_sha256":code_sha,"exact_training_argv":argv}
with open(manifest,'x',encoding='utf-8') as f:
 json.dump(record,f,indent=2); f.write('\n')
PY
if CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m snn_kuramoto_bidirectional.training.train_s2net_core "${ARGS[@]}" > "$OUT/training.log" 2>&1; then :
else rc=$?; printf 'phase=training\nrc=%s\n' "$rc" > "$OUT/FAILED"; exit "$rc"; fi
test -s "$OUT/core.pt"
grep -Fq "trained S2NetCore: $OUT/core.pt" "$OUT/training.log"
"$PY" - "$OUT/manifest.json" "$OUT/core.pt" <<'PY'
import hashlib,json,sys
p,ckpt=sys.argv[1:]
r=json.load(open(p,encoding='utf-8'))
r['checkpoint_sha256']=hashlib.sha256(open(ckpt,'rb').read()).hexdigest()
with open(p+'.tmp','x',encoding='utf-8') as f: json.dump(r,f,indent=2); f.write('\n')
import os; os.replace(p+'.tmp',p)
PY
printf 'completed\n' > "$OUT/TRAINING_COMPLETED"
if bash "$DIR/evaluate.sh" "$GPU" "$SEED" short smoke; then :
else rc=$?; printf 'phase=spike_smoke\nrc=%s\n' "$rc" > "$OUT/FAILED"; exit "$rc"; fi
if bash "$DIR/evaluate_multireadout.sh" "$GPU" "$SEED" smoke; then :
else rc=$?; printf 'phase=multireadout_smoke\nrc=%s\n' "$rc" > "$OUT/FAILED"; exit "$rc"; fi
test -s "$OUT/fixed_spike_short_smoke_n4_T256_settle64.provenance.json"
test -s "$OUT/sw0057_fixed_multireadout_long_smoke_n4.provenance.json" || {
  # The upstream SW0057 validator leaves its report unchanged; this marker records the passed validator.
  test -s "$OUT/sw0057_fixed_multireadout_long_smoke_n4.json"
}
printf 'completed\n' > "$OUT/REAL_ASSET_SMOKE_COMPLETED"
for window in short long; do
  if bash "$DIR/evaluate.sh" "$GPU" "$SEED" "$window" full; then :
  else rc=$?; printf 'phase=evaluate_%s\nrc=%s\n' "$window" "$rc" > "$OUT/FAILED"; exit "$rc"; fi
done
if bash "$DIR/evaluate_multireadout.sh" "$GPU" "$SEED" full; then :
else rc=$?; printf 'phase=evaluate_multireadout\nrc=%s\n' "$rc" > "$OUT/FAILED"; exit "$rc"; fi
test -s "$OUT/fixed_spike_short_full_n320_T256_settle64.provenance.json"
test -s "$OUT/fixed_spike_long_full_n320_T1024_settle512.provenance.json"
test -s "$OUT/sw0057_fixed_multireadout_long_full_n320.json"
printf 'completed\n' > "$OUT/EVALUATION_COMPLETED"
echo "SW0058 training and all fixed evaluations completed for seed=$SEED"

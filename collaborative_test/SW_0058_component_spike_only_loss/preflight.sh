#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU id 0 or 1 required}"
case "$GPU" in 0|1) ;; *) echo "SW0058 permits GPU0/1 only" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0058_component_spike_only_loss"
PY=/Data0/kevinswk/envs/snn/bin/python
GAMMA=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/gamma_train.pt
PILOT="$ROOT/collaborative_test/SW_0054_causal_mechanism_ablation/results"
SC=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/sc.pt
OUT="$ROOT/trained_models/SW0058_preflight_v5"
MARKER="$OUT/PREFLIGHT.json"
mkdir -p "$OUT"
test -s "$GAMMA" -a -s "$SC"
"$PY" "$DIR/pilot_gate.py" --folder "$PILOT"
CHECKSUM="$(sha256sum "$GAMMA" | awk '{print $1}')"
PILOT_RAW="$PILOT/seed0_epoch25_short_n32_pilot_T256_settle64.json"
PILOT_SUMMARY="$PILOT/seed0_epoch25_short_n32_pilot_summary.json"
PILOT_RAW_SHA="$(sha256sum "$PILOT_RAW" | awk '{print $1}')"
PILOT_SUMMARY_SHA="$(sha256sum "$PILOT_SUMMARY" | awk '{print $1}')"
PILOT_VALIDATOR_SHA="$(sha256sum "$ROOT/collaborative_test/SW_0054_causal_mechanism_ablation/summarize.py" | awk '{print $1}')"
CODE_SUM="$(sha256sum "$DIR/preflight.sh" "$DIR/preflight_one_update.py" "$DIR/pilot_gate.py" "$DIR/run.sh" \
  "$DIR/evaluate.sh" "$DIR/evaluate_multireadout.sh" "$DIR/validate_result.py" "$DIR/summarize.py" "$DIR/launch_parallel.sh" \
  "$ROOT/snn_kuramoto_bidirectional/training/train_s2net_core.py" \
  "$ROOT/snn_kuramoto_bidirectional/loss_function.py" "$ROOT/snn_kuramoto_bidirectional/s2net_cls.py" \
  "$ROOT/snn_kuramoto_bidirectional/hyperparameter.py" "$ROOT/snn_kuramoto_bidirectional/sc_generator.py" \
  "$ROOT/snn_kuramoto_bidirectional/graph_generator.py" "$ROOT/snn_kuramoto_bidirectional/kuramoto_layer.py" \
  "$ROOT/snn_kuramoto_bidirectional/dendric_layer.py" "$ROOT/snn_kuramoto_bidirectional/membrane_layer.py" \
  "$ROOT/snn_kuramoto_bidirectional/sinusoidal_gating.py" "$ROOT/snn_kuramoto_bidirectional/spike_classifier.py" \
  "$ROOT/snn_kuramoto_bidirectional/image_conditioned_sc.py" | sha256sum | awk '{print $1}')"
if [[ -e "$MARKER" ]]; then
  "$PY" - "$MARKER" "$CHECKSUM" "$CODE_SUM" "$PILOT_RAW_SHA" "$PILOT_SUMMARY_SHA" "$PILOT_VALIDATOR_SHA" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding='utf-8'))
assert m.get('version')==5 and m.get('passed') is True
assert m.get('gamma_sha256')==sys.argv[2] and m.get('code_sha256')==sys.argv[3]
assert m.get('sw0054_raw_sha256')==sys.argv[4] and m.get('sw0054_summary_sha256')==sys.argv[5]
assert m.get('sw0054_validator_sha256')==sys.argv[6]
print('Existing SW0058 one-update preflight is current for gamma and code.')
PY
  exit 0
fi
if ! PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; then
  echo "Cannot inspect GPU $GPU: $PIDS" >&2; exit 2
fi
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU occupied; refusing preflight" >&2; exit 2; }
if [[ -e "$OUT/one_update" || -e "$OUT/one_update.log" || -e "$MARKER" ]]; then
  echo "Partial preflight artifacts exist; manual review required" >&2; exit 3
fi
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$DIR/preflight_one_update.py" \
  --gamma "$GAMMA" --sc "$SC" --output-dir "$OUT/one_update" --device cuda \
  > "$OUT/one_update.log" 2>&1
"$PY" - "$OUT/one_update/report.json" "$CHECKSUM" "$CODE_SUM" <<'PY'
import json,math,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
assert r['gamma_sha256']==sys.argv[2] and r['recipe']=={'primary_loss_weight':0.0,'spike_plv_weight':5.0,'steps_per_batch':64,'plv_settle':32,'epochs':1,'batches':1}
assert r['canonical_training_module'] is True and r['sc_matches_full_gamma'] is True
assert r['finite_checkpoint'] is True and math.isfinite(r['loss']) and math.isfinite(r['spike_aux_total_weighted_logged'])
assert r['primary_total_weighted_logged']==0.0 and abs(r['loss']-r['spike_aux_total_weighted_logged'])<1e-5
PY
# Create the immutable marker only after every assertion passes.
"$PY" - "$MARKER" "$CHECKSUM" "$CODE_SUM" "$PILOT_RAW_SHA" "$PILOT_SUMMARY_SHA" "$PILOT_VALIDATOR_SHA" <<'PY'
import json,sys
with open(sys.argv[1],'x',encoding='utf-8') as f:
 json.dump({'version':5,'passed':True,'gamma_sha256':sys.argv[2],'code_sha256':sys.argv[3],
  'sw0054_raw_sha256':sys.argv[4],'sw0054_summary_sha256':sys.argv[5],
  'sw0054_validator_sha256':sys.argv[6],
  'one_update_report':'one_update/report.json','preflight_ids':[0,15],
  'checks':['canonical trainer module','one real batch','fixed full-gamma SC','finite loss','finite checkpoint']},f,indent=2)
 f.write('\n')
PY
echo "SW0058 preflight passed: $MARKER"

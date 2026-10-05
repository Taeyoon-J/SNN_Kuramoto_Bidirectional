#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0054_causal_mechanism_ablation"
CHECKPOINT="$ROOT/trained_models/SW_0050_sample_diversity_s0_w0_lr0p0003_epoch25/checkpoints/epoch_25.pt"
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
OUT="$ROOT/trained_models/SW_0054_causal_mechanism_ablation"
MARKER="$OUT/PREFLIGHT_V1.json"
mkdir -p "$OUT/cache"
test -s "$CHECKPOINT" && test -s "$GAMMA" && test -s "$MANIFEST" && test -s "$HDF5"
CHECKPOINT_SUM="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
GAMMA_SUM="$(sha256sum "$GAMMA" | awk '{print $1}')"
MANIFEST_SUM="$(sha256sum "$MANIFEST" | awk '{print $1}')"
HDF5_META="$(stat -c '%s:%Y' "$HDF5")"
CODE_SUM="$(sha256sum "$DIR/evaluate.py" "$DIR/interventions.py" "$DIR/preflight.sh" \
  "$ROOT/collaborative_test/evaluate_fixed_split.py" "$ROOT/snn_kuramoto_bidirectional/s2net_cls.py" \
  "$ROOT/snn_kuramoto_bidirectional/kuramoto_layer.py" "$ROOT/snn_kuramoto_bidirectional/sinusoidal_gating.py" | sha256sum | awk '{print $1}')"
if [[ -s "$MARKER" ]] && /Data0/kevinswk/envs/snn/bin/python - "$MARKER" "$CHECKPOINT_SUM" "$GAMMA_SUM" "$MANIFEST_SUM" "$HDF5_META" "$CODE_SUM" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
keys=("checkpoint_sha256","gamma_sha256","manifest_sha256","hdf5_size_mtime","code_sha256")
raise SystemExit(0 if m.get("version")==1 and m.get("passed") is True and
                 all(m.get(k)==v for k,v in zip(keys,sys.argv[2:])) else 1)
PY
then echo "SW0054 preflight v1 current for these assets/code"; exit 0; fi
if ! PIDS="$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader 2>&1)"; then
  echo "Unable to query GPU $GPU_ID: $PIDS" >&2; exit 2
fi
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU_ID occupied; refusing preflight overlap" >&2; exit 2; }
TMP="$(mktemp -d "$OUT/cache/.preflight.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
export CUDA_VISIBLE_DEVICES="$GPU_ID" TMPDIR="$TMP" TRITON_CACHE_DIR="$TMP"
/Data0/kevinswk/envs/snn/bin/python "$DIR/evaluate.py" --checkpoint "$CHECKPOINT" \
  --gamma-path "$GAMMA" --gamma-manifest "$MANIFEST" --dataset-path "$HDF5" \
  --output-path "$TMP/smoke.json" --start 1320 --count 4 --steps 256 --settle 64 --batch-size 2 --device cuda \
  > "$TMP/smoke.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python - "$TMP/smoke.json" <<'PY'
import json,math,sys
r=json.load(open(sys.argv[1])); c=r["conditions"]
assert r["target"]["ground_truth_used_for_prediction"] is False
assert set(c)=={"normal","gate_perm_s0","gate_perm_s1","gate_perm_s2",
 "carrier_perm_s0","carrier_perm_s1","carrier_perm_s2","gate_mean","carrier_mean","K0"}
assert c["K0"]["kuramoto_K_during_rollout"]==0.0
for name in ("gate_perm_s0","gate_perm_s1","gate_perm_s2","carrier_perm_s0",
             "carrier_perm_s1","carrier_perm_s2","gate_mean","carrier_mean"):
 rows=c[name]["gate_invariants"]
 assert rows and all(x["graph_bitwise_equal"] and x["theta_allclose"] and math.isfinite(x["theta_max_abs_delta"]) for x in rows)
for condition in c.values():
 assert set(condition["fixed_readouts"])=={"spike_cc","membrane_spatial"}
 for readout in condition["fixed_readouts"].values():
  assert all(math.isfinite(float(x)) for x in readout["metrics"].values())
 assert set(condition["distance_controlled_macro_auc"])=={"phase","gate","carrier","h_wave","membrane","spike"}
 for x in condition["activity_scale"].values():
  if isinstance(x,float): assert math.isfinite(x)
PY
/Data0/kevinswk/envs/snn/bin/python - "$MARKER" "$CHECKPOINT_SUM" "$GAMMA_SUM" "$MANIFEST_SUM" "$HDF5_META" "$CODE_SUM" <<'PY'
import json,sys
json.dump({"version":1,"passed":True,"checkpoint_sha256":sys.argv[2],"gamma_sha256":sys.argv[3],
 "manifest_sha256":sys.argv[4],"hdf5_size_mtime":sys.argv[5],"code_sha256":sys.argv[6],
 "preflight_ids":[1320,1323],"steps":256,"settle":64,
 "validation":"finite readouts; gate theta/graph invariants; K-zero intervention"},open(sys.argv[1],"w"),indent=2)
PY
echo "SW0054 preflight passed: $MARKER"

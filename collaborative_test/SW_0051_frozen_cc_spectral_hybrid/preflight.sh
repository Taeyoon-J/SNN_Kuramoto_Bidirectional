#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
SEED="${2:?seed 0, 1, or 2 required}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0051_frozen_cc_spectral_hybrid"
OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s${SEED}"
CHECKPOINT="$OUT/core.pt"
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
MARKER="$OUT/SW0051_PREFLIGHT_V1.json"
mkdir -p "$OUT/cache"
test -s "$CHECKPOINT" && test -s "$GAMMA" && test -s "$MANIFEST" && test -s "$HDF5"
CHECKSUM="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
GAMMA_SUM="$(sha256sum "$GAMMA" | awk '{print $1}')"
MANIFEST_SUM="$(sha256sum "$MANIFEST" | awk '{print $1}')"
CODE_SUM="$(sha256sum "$DIR/evaluate.py" "$DIR/hybrid.py" "$DIR/preflight.sh" | sha256sum | awk '{print $1}')"
if [[ -s "$MARKER" ]] && /Data0/kevinswk/envs/snn/bin/python - "$MARKER" "$CHECKSUM" "$GAMMA_SUM" "$MANIFEST_SUM" "$CODE_SUM" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
expected={"checkpoint_sha256":sys.argv[2],"gamma_sha256":sys.argv[3],
          "manifest_sha256":sys.argv[4],"code_sha256":sys.argv[5]}
sys.exit(0 if m.get("version")==1 and m.get("passed") is True and
         all(m.get(k)==v for k,v in expected.items()) else 1)
PY
then
  echo "SW0051 preflight v1 already passed for these assets and evaluator code"
  exit 0
fi
if ! nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader > "$OUT/cache/preflight_gpu_pids.$$" 2> "$OUT/cache/preflight_gpu_error.$$"; then
  cat "$OUT/cache/preflight_gpu_error.$$" >&2; rm -f "$OUT/cache/preflight_gpu_pids.$$" "$OUT/cache/preflight_gpu_error.$$"; exit 2
fi
if grep -q '[0-9]' "$OUT/cache/preflight_gpu_pids.$$"; then
  rm -f "$OUT/cache/preflight_gpu_pids.$$" "$OUT/cache/preflight_gpu_error.$$"
  echo "GPU $GPU_ID is occupied; refusing SW0051 preflight overlap" >&2; exit 2
fi
rm -f "$OUT/cache/preflight_gpu_pids.$$" "$OUT/cache/preflight_gpu_error.$$"
TMP="$(mktemp -d "$OUT/cache/.sw0051_preflight_${SEED}.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
export CUDA_VISIBLE_DEVICES="$GPU_ID" TMPDIR="$TMP" TRITON_CACHE_DIR="$TMP"
/Data0/kevinswk/envs/snn/bin/python "$DIR/evaluate.py" \
  --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --gamma-manifest "$MANIFEST" \
  --dataset-path "$HDF5" --output-path "$TMP/preflight.json" --global-start 1320 --count 4 \
  --steps 256 --settle 64 --batch-size 4 --device cuda > "$TMP/preflight.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python - "$TMP/preflight.json" <<'PY'
import json, math, sys
r=json.load(open(sys.argv[1]))
assert r.get("ids")==[1320,1323], r.get("ids")
assert r.get("readout",{}).get("prediction_uses_ground_truth") is False
rows=r.get("rows",[])
assert len(rows)==7, len(rows)
for row in rows:
    assert row.get("foreground_mask_exactly_matches_spike_cc") is True, row.get("mode")
    for key in ("fg_ari", "foreground_iou", "matched_object_iou", "predicted_foreground_fraction", "predicted_object_count_mean"):
        assert math.isfinite(float(row[key])), (row.get("mode"), key, row.get(key))
PY
/Data0/kevinswk/envs/snn/bin/python - "$MARKER" "$CHECKSUM" "$GAMMA_SUM" "$MANIFEST_SUM" "$CODE_SUM" <<'PY'
import json,sys
json.dump({"version":1,"passed":True,"checkpoint_sha256":sys.argv[2],
           "gamma_sha256":sys.argv[3],"manifest_sha256":sys.argv[4],
           "code_sha256":sys.argv[5],
           "preflight_ids":[1320,1323],"window":{"steps":256,"settle":64},
           "validation":"all seven readouts finite; hybrid masks equal frozen spike-CC mask"},
          open(sys.argv[1],"w"),indent=2)
PY
echo "SW0051 preflight passed: $MARKER"

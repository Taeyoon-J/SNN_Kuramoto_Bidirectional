#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"; SEED="${2:?seed required}"
case "$SEED" in 0|1|2) ;; *) exit 2 ;; esac
case "$GPU_ID" in 0|1) ;; *) echo "only GPUs 0 and 1 are permitted" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0057_fixed_multireadout"
OUT="$ROOT/trained_models/SW0055_unique2500_s${SEED}_e10_lr0p0003"
CHECKPOINT="$OUT/core.pt"; GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
MARKER="$OUT/SW0057_PREFLIGHT_V1.json"
mkdir -p "$OUT/cache"
for f in "$CHECKPOINT" "$GAMMA" "$MANIFEST" "$HDF5"; do test -s "$f" || { echo "missing asset $f" >&2; exit 1; }; done
if nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then echo "GPU $GPU_ID occupied" >&2; exit 2; fi
CS=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
GS=$(sha256sum "$GAMMA" | awk '{print $1}')
MS=$(sha256sum "$MANIFEST" | awk '{print $1}')
CODE=$(/Data0/kevinswk/envs/snn/bin/python - "$DIR" "$ROOT" <<'PY'
import hashlib,sys
from pathlib import Path
d=Path(sys.argv[1]); r=Path(sys.argv[2]); h=hashlib.sha256()
files=[d/'evaluate.py',d/'validate_result.py',d/'preflight.sh',d/'gamma_contract.py',d/'target_contract.py',
 r/'collaborative_test/evaluate_fixed_split.py',
 r/'collaborative_test/SW_0027_component_membrane_spectral/evaluate.py',
 r/'collaborative_test/SW_0028_spatial_membrane_spectral/spatial_evaluate.py',
 r/'snn_kuramoto_bidirectional/spike_classifier.py',
 r/'snn_kuramoto_bidirectional/evaluation.py',
 r/'snn_kuramoto_bidirectional/training/evaluate_binding.py']
for f in files: h.update(f.read_bytes())
print(h.hexdigest())
PY
)
if [[ -e "$MARKER" ]]; then
  /Data0/kevinswk/envs/snn/bin/python - "$MARKER" "$CS" "$GS" "$MS" "$CODE" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); expected=(sys.argv[2:])
actual=[m.get(k) for k in ('checkpoint_sha256','gamma_sha256','manifest_sha256','code_sha256')]
if m.get('version') != 1 or m.get('passed') is not True or actual != expected:
    raise SystemExit('stale or invalid preflight marker; refusing overwrite')
PY
  echo "matching SW0057 preflight already passed"; exit 0
fi
TMP=$(mktemp -d "$OUT/cache/.sw0057_preflight_${SEED}.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
export CUDA_VISIBLE_DEVICES="$GPU_ID" TMPDIR="$TMP" TRITON_CACHE_DIR="$TMP"
/Data0/kevinswk/envs/snn/bin/python "$DIR/evaluate.py" --checkpoint "$CHECKPOINT" --seed "$SEED" \
 --gamma-path "$GAMMA" --gamma-manifest "$MANIFEST" --dataset-path "$HDF5" --output-path "$TMP/preflight.json" \
 --global-start 1320 --count 4 --steps 1024 --settle 512 --batch-size 4 --device cuda > "$TMP/preflight.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python "$DIR/validate_result.py" "$TMP/preflight.json" "$CHECKPOINT" "$SEED" --expected-count 4
/Data0/kevinswk/envs/snn/bin/python - "$MARKER" "$CS" "$GS" "$MS" "$CODE" <<'PY'
import json,sys
json.dump(dict(version=1,passed=True,checkpoint_sha256=sys.argv[2],gamma_sha256=sys.argv[3],manifest_sha256=sys.argv[4],code_sha256=sys.argv[5],preflight_ids=[1320,1323],window={'steps':1024,'settle':512}),open(sys.argv[1],'x'),indent=2)
PY
echo "SW0057 preflight passed"

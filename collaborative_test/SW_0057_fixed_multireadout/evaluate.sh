#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"; SEED="${2:?seed required}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac
case "$GPU_ID" in 0|1) ;; *) echo "only GPUs 0 and 1 are permitted" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0057_fixed_multireadout"
OUT="$ROOT/trained_models/SW0055_unique2500_s${SEED}_e10_lr0p0003"
RESULT="$OUT/sw0057_fixed_multireadout_long.json"; LOG="${RESULT%.json}.log"
MARKER="$OUT/SW0057_PREFLIGHT_V1.json"
[[ -s "$OUT/core.pt" && -s "$MARKER" ]] || { echo "missing checkpoint or passing preflight marker" >&2; exit 1; }
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "refusing overwrite $RESULT" >&2; exit 1; }
/Data0/kevinswk/envs/snn/bin/python - "$MARKER" "$OUT/core.pt" "$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt" "$ROOT/data/SW_0042_hdf5_aligned/manifest.json" "$DIR" "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads(Path(sys.argv[1]).read_text()); d=Path(sys.argv[5]); r=Path(sys.argv[6])
files=[d/'evaluate.py',d/'validate_result.py',d/'preflight.sh',d/'gamma_contract.py',d/'target_contract.py',
 r/'collaborative_test/evaluate_fixed_split.py',
 r/'collaborative_test/SW_0027_component_membrane_spectral/evaluate.py',
 r/'collaborative_test/SW_0028_spatial_membrane_spectral/spatial_evaluate.py',
 r/'snn_kuramoto_bidirectional/spike_classifier.py',
 r/'snn_kuramoto_bidirectional/evaluation.py',
 r/'snn_kuramoto_bidirectional/training/evaluate_binding.py']
code=hashlib.sha256(b''.join(f.read_bytes() for f in files)).hexdigest()
expected=(sha(sys.argv[2]),sha(sys.argv[3]),sha(sys.argv[4]),code)
actual=tuple(m.get(k) for k in ('checkpoint_sha256','gamma_sha256','manifest_sha256','code_sha256'))
if m.get('version') != 1 or m.get('passed') is not True or actual != expected:
    raise SystemExit('preflight marker does not match checkpoint/assets/code')
PY
if nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU $GPU_ID occupied; refusing overlap" >&2; exit 2
fi
mkdir -p "$OUT/cache"
export CUDA_VISIBLE_DEVICES="$GPU_ID" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
/Data0/kevinswk/envs/snn/bin/python "$DIR/evaluate.py" \
  --checkpoint "$OUT/core.pt" --seed "$SEED" \
  --gamma-path "$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt" \
  --gamma-manifest "$ROOT/data/SW_0042_hdf5_aligned/manifest.json" \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$RESULT" --global-start 1320 --count 320 --steps 1024 --settle 512 \
  --batch-size 4 --device cuda > "$LOG" 2>&1
/Data0/kevinswk/envs/snn/bin/python "$DIR/validate_result.py" "$RESULT" "$OUT/core.pt" "$SEED"
echo "SW0057 complete: $RESULT"

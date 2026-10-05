#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
ARM="${2:?arm required}"
EPOCH="${3:?epoch required}"
case "$ARM" in
  low_lr) MODEL_DIR=SW_0050_sample_diversity_s2_w0_lr0p0003_full ;;
  diversity) MODEL_DIR=SW_0050_sample_diversity_s2_w2p45126043147_lr0p001_full ;;
  *) echo "arm must be low_lr or diversity" >&2; exit 2 ;;
esac
case "$EPOCH" in 20|25|30|35) ;; *) echo "invalid epoch" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0052_checkpoint_trajectory"
CHECKPOINT="$ROOT/trained_models/$MODEL_DIR/checkpoints/epoch_${EPOCH}.pt"
OUT="$ROOT/trained_models/SW_0052_checkpoint_trajectory"
MARKER="$OUT/${ARM}_epoch${EPOCH}_PREFLIGHT.json"
mkdir -p "$OUT/cache_gpu${GPU_ID}"
test -s "$CHECKPOINT"
CHECKSUM="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
CODE_SUM="$(sha256sum "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" "$DIR/evaluate_checkpoint.sh" "$DIR/preflight_checkpoint.sh" "$DIR/validate_result.py" | sha256sum | awk '{print $1}')"
if [[ -s "$MARKER" ]] && /Data0/kevinswk/envs/snn/bin/python - "$MARKER" "$CHECKSUM" "$CODE_SUM" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
raise SystemExit(0 if m.get("passed") is True and m.get("checkpoint_sha256")==sys.argv[2] and m.get("code_sha256")==sys.argv[3] else 1)
PY
then echo "current preflight exists: $ARM epoch $EPOCH"; exit 0; fi
TMP="$(mktemp -d "$OUT/cache_gpu${GPU_ID}/.preflight_${ARM}_${EPOCH}.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
export CUDA_VISIBLE_DEVICES="$GPU_ID" TMPDIR="$TMP" TRITON_CACHE_DIR="$TMP"
/Data0/kevinswk/envs/snn/bin/python - "$CHECKPOINT" <<'PY'
import sys,torch
s=torch.load(sys.argv[1],map_location="cpu",weights_only=True)
assert isinstance(s,dict) and s
assert all(torch.is_tensor(v) and torch.isfinite(v).all() for v in s.values())
PY
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$CHECKPOINT" --gamma-path "$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt" \
  --gamma-global-start 1320 --gamma-manifest "$ROOT/data/SW_0042_hdf5_aligned/manifest.json" \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$TMP/smoke.json" --start 1320 --count 4 --steps 256 --settle 64 \
  --membrane-vth 0.06 --min-group-size 2 --background largest_component \
  --dendritic-projection shared --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --thresholds 0.05 0.10 0.20 0.35 0.50 --phase-endpoint --device cuda > "$TMP/smoke.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python "$DIR/validate_result.py" "$TMP/smoke.json" "$CHECKPOINT" 4 256 64
/Data0/kevinswk/envs/snn/bin/python - "$MARKER" "$CHECKSUM" "$CODE_SUM" "$ARM" "$EPOCH" <<'PY'
import json,sys
json.dump({"passed":True,"checkpoint_sha256":sys.argv[2],"code_sha256":sys.argv[3],"arm":sys.argv[4],"epoch":int(sys.argv[5]),"smoke_images":4},open(sys.argv[1],"w"),indent=2)
PY
echo "preflight passed: $ARM epoch $EPOCH"

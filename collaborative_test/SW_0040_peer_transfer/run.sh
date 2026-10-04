#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Select an idle GPU after inspecting nvidia-smi}"
CHECKPOINT="${2:?Pass the checkpoint to evaluate}"
OUT="${3:?Pass an output JSON path}"
PEER_TARGETS="${4:-}"
PEER_MANIFEST="${5:-}"
PEER_ARGS=()
if [[ -n "$PEER_TARGETS" || -n "$PEER_MANIFEST" ]]; then
  if [[ -z "$PEER_TARGETS" || -z "$PEER_MANIFEST" ]]; then
    echo "Pass both PEER_TARGETS and PEER_MANIFEST, or neither" >&2
    exit 2
  fi
  PEER_ARGS=(--peer-targets "$PEER_TARGETS" --peer-manifest "$PEER_MANIFEST" --peer-index-map identity)
fi
ROOT=/Data0/kevinswk/patch_v2_sw
mkdir -p "$(dirname "$OUT")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$CHECKPOINT" \
  --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$OUT" --start 1320 --count 320 --steps 256 --settle 64 \
  --membrane-vth 2.0 --min-group-size 2 \
  --device cuda "${PEER_ARGS[@]}"

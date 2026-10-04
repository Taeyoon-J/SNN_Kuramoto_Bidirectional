#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Pass an assigned GPU ID}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s2"
CHECKPOINT="$OUT/core.pt"
GAMMA=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/gamma_train.pt
RESULT="$OUT/sample_diversity_gradient_probe.json"
if [[ -e "$RESULT" ]]; then echo "Refusing to overwrite probe result: $RESULT" >&2; exit 1; fi
test -s "$CHECKPOINT"; test -s "$GAMMA"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0050_sample_diversity/probe_gradient_scale.py" \
  --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --output "$RESULT" \
  --device cuda --batch-size 16
echo "Measured and saved gradient scale: $RESULT"

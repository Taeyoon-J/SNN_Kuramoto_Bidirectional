#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
shift
DIR=/Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0052_checkpoint_trajectory
while (( $# )); do
  ARM="$1"; EPOCH="$2"; shift 2
  bash "$DIR/evaluate_checkpoint.sh" "$GPU_ID" "$ARM" "$EPOCH" short
done
echo completed > "/Data0/kevinswk/patch_v2_sw/trained_models/SW_0052_checkpoint_trajectory/GPU${GPU_ID}_SHORT_COMPLETE"

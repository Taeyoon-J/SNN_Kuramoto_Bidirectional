#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1 required}"
case "$GPU" in 0|1) ;; *) exit 2 ;; esac
DIR=/Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0083_threshold_recalibration
for VTH in v006 v05 v10 v20; do
  bash "$DIR/evaluate.sh" "$GPU" "$VTH"
done

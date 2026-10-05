#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0084_joint_feature_reconstruction"
for ARM in A B C D; do
 test -s "$ROOT/trained_models/SW0084_preflight_${ARM}/PREFLIGHT_VALIDATED.json" || {
  echo "missing validated preflight for arm $ARM" >&2; exit 3;
 }
 [[ ! -e "$ROOT/trained_models/SW0084_${ARM}_seed1" ]] || {
  echo "refusing existing output for arm $ARM" >&2; exit 3;
 }
done
for GPU in 0 1 2 3; do
 PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
 [[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU busy" >&2; exit 4; }
done
bash "$DIR/run.sh" 0 A & PIDA=$!
bash "$DIR/run.sh" 1 B & PIDB=$!
bash "$DIR/run.sh" 2 C & PIDC=$!
bash "$DIR/run.sh" 3 D & PIDD=$!
RC=0
wait "$PIDA" || RC=$?; wait "$PIDB" || RC=$?; wait "$PIDC" || RC=$?; wait "$PIDD" || RC=$?
exit "$RC"

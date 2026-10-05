#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw; EXP="$ROOT/collaborative_test/SW_0068_joint_feature_core"; OUT="$ROOT/trained_models/SW0068b_stage1"
LOCK="$ROOT/trained_models/SW0068b_stage1_controller.lock"; mkdir "$LOCK"; mkdir -p "$OUT"
trap 'printf "failed\n" > "$OUT/STAGE1_FAILED"' ERR
deadline=$((SECONDS+5400))
while [[ ! -f "$ROOT/trained_models/SW0068b_joint_s1_lr3e6/TRAINING_COMPLETED" || ! -f "$ROOT/trained_models/SW0068b_joint_s1_lr3e5/TRAINING_COMPLETED" ]]; do
 ((SECONDS<deadline)) || exit 4; sleep 20
done
for gpu in 0 1; do while nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>&1 | grep -Eq '[0-9]'; do ((SECONDS<deadline)) || exit 5; sleep 5; done; done
bash "$EXP/evaluate.sh" 0 lr3e6 & p0=$!; bash "$EXP/evaluate.sh" 1 lr3e5 & p1=$!; wait "$p0"; wait "$p1"
/Data0/kevinswk/envs/snn/bin/python "$EXP/summarize.py" --result-dir "$OUT" --output "$OUT/summary.json" > "$OUT/summary.log"
printf 'completed\n' > "$OUT/STAGE1_COMPLETED"

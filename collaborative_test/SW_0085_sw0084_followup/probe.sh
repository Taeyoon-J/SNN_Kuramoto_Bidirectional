#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0-3 required}"; MODE="${2:?baseline or candidate required}"
case "$GPU" in 0|1|2|3) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; OUT="$ROOT/trained_models/SW0085_followup"
if [[ "$MODE" == baseline ]]; then
 CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
 GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"; TAG=baseline
else
 [[ "$MODE" == candidate ]] || exit 2
 read -r ARM EPOCH < <(/Data0/kevinswk/envs/snn/bin/python - "$OUT/selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))["chosen"]
print(x["arm"], x["epoch"])
PY
 )
 MODEL="$ROOT/trained_models/SW0084_${ARM}_seed1/epochs/epoch_${EPOCH}"
 CORE="$MODEL/core.pt"; GAMMA="$MODEL/gamma_validation.pt"; TAG="${ARM}_epoch${EPOCH}"
fi
RESULT="$OUT/flow_${TAG}.json"; LOG="${RESULT%.json}.log"
test -s "$CORE" -a -s "$GAMMA"; [[ ! -e "$RESULT" && ! -e "$LOG" ]] || exit 3
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || exit 4
mkdir -p "$OUT/cache${GPU}"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0064_activation_flow_probe/probe.py" \
 --checkpoint "$CORE" --gamma "$GAMMA" --gamma-global-start 1320 \
 --hdf5 /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --seed 1 \
 --start 1320 --count 32 --steps 256 --device cuda --output "$RESULT" > "$LOG" 2>&1
test -s "$RESULT"


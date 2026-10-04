#!/usr/bin/env bash
# Asset-bound v1 gate: real checkpoint gradient probe plus one real-row update.
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0050_sample_diversity"
BASE=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002
BASE_OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s2"
CHECKPOINT="$BASE_OUT/core.pt"
GAMMA="$BASE/gamma_train.pt"
PROBE="$BASE_OUT/sample_diversity_gradient_probe.json"
LOCK="$BASE_OUT/.sample_diversity_probe_lock"
PYTHON=/Data0/kevinswk/envs/snn/bin/python
MARKER="$ROOT/trained_models/SW0050_PREFLIGHT_V1.txt"
test -x "$PYTHON" && test -s "$CHECKPOINT" && test -s "$GAMMA"
cd "$ROOT"
if ! "$PYTHON" -m snn_kuramoto_bidirectional.training.train_s2net_core --help > /dev/null; then
  echo "Absolute SNN Python cannot load the training CLI" >&2; exit 2
fi
"$PYTHON" -m snn_kuramoto_bidirectional.training.train_s2net_core --help | \
  grep -q -- '--sample-activity-diversity-weight'
CHECKSUM="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
GAMMA_SUM="$(sha256sum "$GAMMA" | awk '{print $1}')"
CODE_SUM="$(sha256sum "$ROOT/snn_kuramoto_bidirectional/training/train_s2net_core.py" \
  "$ROOT/snn_kuramoto_bidirectional/loss_function.py" "$DIR/run.sh" \
  "$DIR/probe_gradient_scale.py" "$DIR/preflight.sh" | sha256sum | awk '{print $1}')"
EXPECTED="SW0050_PREFLIGHT_V1 $CHECKSUM $GAMMA_SUM $CODE_SUM"
if [[ -s "$MARKER" ]] && grep -Fxq "$EXPECTED" "$MARKER"; then
  echo "SW0050 preflight v1 already passed for these checkpoint/gamma assets"
  exit 0
fi
if ! nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader >/tmp/sw0050_gpu_pids.$$ 2>/tmp/sw0050_gpu_error.$$; then
  cat /tmp/sw0050_gpu_error.$$ >&2; rm -f /tmp/sw0050_gpu_pids.$$ /tmp/sw0050_gpu_error.$$; exit 2
fi
if grep -q '[0-9]' /tmp/sw0050_gpu_pids.$$; then
  rm -f /tmp/sw0050_gpu_pids.$$ /tmp/sw0050_gpu_error.$$
  echo "GPU $GPU_ID is occupied; refusing preflight rather than overlapping work" >&2; exit 2
fi
rm -f /tmp/sw0050_gpu_pids.$$ /tmp/sw0050_gpu_error.$$
mkdir -p "$ROOT/trained_models"
TMP="$(mktemp -d "$ROOT/trained_models/.SW0050_preflight.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
probe_is_current() {
  [[ -s "$PROBE" ]] || return 1
  "$PYTHON" - "$PROBE" "$CHECKSUM" "$GAMMA_SUM" >/dev/null 2>&1 <<'PY'
import json, sys
try:
    r=json.load(open(sys.argv[1]))
except (OSError, ValueError):
    raise SystemExit(1)
ok=(r.get("checkpoint_sha256")==sys.argv[2] and
    r.get("gamma_sha256")==sys.argv[3] and
    r.get("checkpoint")=="/Data0/kevinswk/patch_v2_sw/trained_models/SW_0042_HDF5_aligned_BIM6_s2/core.pt" and
    r.get("gamma_path")=="/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/gamma_train.pt")
raise SystemExit(0 if ok else 1)
PY
}
while [[ -d "$LOCK" ]]; do echo "Waiting for the existing SW0050 gradient probe lock."; sleep 20; done
if ! probe_is_current; then
  if ! mkdir "$LOCK" 2>/dev/null; then
    echo "Gradient probe lock was acquired concurrently; rerun preflight after it finishes." >&2; exit 2
  fi
  trap 'rmdir "$LOCK" 2>/dev/null || true; rm -rf "$TMP"' EXIT
  if ! probe_is_current; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$DIR/probe_gradient_scale.py" \
      --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --output "$TMP/probe.json" \
      --device cuda --batch-size 16 > "$TMP/probe.log" 2>&1
    "$PYTHON" - "$TMP/probe.json" <<'PY'
import json, math, sys
r=json.load(open(sys.argv[1]))
assert r["probe_batch_count"] >= 2
assert r["gamma_train_rows"][0] == 0 and r["gamma_train_rows"][1] >= 1
assert r["ground_truth_used"] is False
assert r["model_protocol"]["steps"] == 64
assert r["model_protocol"]["plv_settle"] == 32
assert r["model_protocol"]["activity_source"] == "sigmoid(membrane)"
assert math.isfinite(r["suggested_weights"]["target_0.1x_baseline_gradient"])
assert r["suggested_weights"]["target_0.1x_baseline_gradient"] > 0
PY
    mv "$TMP/probe.json" "$PROBE"
  fi
  rmdir "$LOCK"
  trap 'rm -rf "$TMP"' EXIT
fi
"$PYTHON" - "$PROBE" "$CHECKSUM" "$GAMMA_SUM" <<'PY'
import json, math, sys
r=json.load(open(sys.argv[1]))
assert r["probe_batch_count"] >= 2 and r["ground_truth_used"] is False
assert r["checkpoint"] == "/Data0/kevinswk/patch_v2_sw/trained_models/SW_0042_HDF5_aligned_BIM6_s2/core.pt"
assert r["gamma_path"] == "/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/gamma_train.pt"
assert r["checkpoint_sha256"] == sys.argv[2]
assert r["gamma_sha256"] == sys.argv[3]
assert r["model_protocol"]["steps"] == 64 and r["model_protocol"]["plv_settle"] == 32
assert math.isfinite(r["suggested_weights"]["target_0.1x_baseline_gradient"])
PY

# Make a one-row tensor from actual aligned training gamma, yielding exactly
# one optimizer update while still checking the real loader/model/loss path.
"$PYTHON" - "$GAMMA" "$TMP/gamma_one.pt" <<'PY'
import sys, torch
gamma=torch.load(sys.argv[1],map_location="cpu",weights_only=True)
assert gamma.ndim >= 2 and gamma.shape[0] >= 1
torch.save(gamma[:1].clone(),sys.argv[2])
PY
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
  --gamma-seq-path "$TMP/gamma_one.pt" --save-path "$TMP/core.pt" \
  --num-regions 256 --num-feature-maps 8 --device cuda --epochs 1 --batch-size 1 \
  --lr 0.001 --seed 2 --osc-dim 4 --gamma-drive-mode static --num-time-steps 64 \
  --plv-settle 32 --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
  --graph-mode learned --graph-top-k 32 --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --spike-spatial-grid-size 16 --k 256 --membrane-vth 0.06 \
  --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 \
  --branch 4 --gate-mode raw --plv-source phase --plv-combine mean \
  --spike-plv-weight 5.0 --loss-signal sigmoid_membrane --sample-activity-diversity-weight 0 \
  --checkpoint-dir "$TMP/checkpoints" --checkpoint-epochs 1 \
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 \
  --structural-weight 0 --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0 \
  --plv-balance-weight 10.0 --plv-target-density 0.867 --plv-coherence-weight 0.5 \
  --spike-per-component --dendritic-projection shared --verbose > "$TMP/train.log" 2>&1
test -s "$TMP/core.pt" && test -s "$TMP/checkpoints/epoch_01.pt"
grep -q 'Epoch 0001/0001' "$TMP/train.log"
printf '%s\n' "$EXPECTED" > "$MARKER.tmp.$$"
mv "$MARKER.tmp.$$" "$MARKER"
echo "SW0050 asset-bound preflight passed; one-step model/checkpoint smoke complete: $MARKER"

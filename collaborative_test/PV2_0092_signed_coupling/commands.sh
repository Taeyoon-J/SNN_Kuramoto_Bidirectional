#!/bin/bash
# PV2_0092. Signed coupling: does letting dissimilar patches repel break the merges?
#
# The argument is in PROPOSAL_0.9_v2.md. In short: the adjacency is strictly
# non-negative, so every coupling is attractive and nothing pushes two patches apart;
# groups exist only because the graph is sparse. Measured on a trained checkpoint with
# the oscillators held in two phase groups, the mean coupling drive on each group is
# +-0.00904 as it stands and +-1.27025 with rows balanced to sum zero -- 140x larger,
# and in the separating direction. Our recorded error mode is coherent object merges,
# which is what a pairwise attractive rule cannot correct.
#
# Gate, fixed in advance (PROPOSAL_0.9_v2.md): the gain must appear with predicted
# foreground at or below ~0.20, must not be bought with foreground IoU, and
# matched_object_iou must move too, since over-prediction does not help it.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G=/work/USERS/tkim1/gamma_sequences/wm_patch_gamma_seq_k8_grid16.pt
OUT=$R/pv2_0092_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2

pkill -u tkim1 -f extent.sh 2>/dev/null
sleep 2

ev () {  # label checkpoint balance extra...
  local lab=$1 ck=$2 bal=$3; shift 3
  [ -s "$ck" ] || { say "$lab MISSING"; return; }
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
    --targets /work/USERS/tkim1/clevr/with_masks/targets_v1.pt --manifest $MAN \
    --split validation --limit 300 --spike-per-component --geodesic-steps 3 \
    --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --geodesic-chunk 64 \
    --coupling-balance $bal --num-time-steps 1024 --settle 512 --spatial-sigma 1.0 "$@" 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|predicted_foreground" \
      | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -4 >> $OUT
  else printf "%-34s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say "=== PV2_0092: signed coupling, seed 0, contract v1 ==="
say "regression row: the patch must leave the published number alone at balance 0.0"
ev "REGRESSION BIM6_s1 (expect .7239 .7251 .4951)" $R/BIM6_s1/core.pt 0.0 --synchrony-threshold 0.05

BASE="--num-feature-maps 8 --num-regions 256 --device cuda --batch-size 16 --lr 1e-3
  --osc-dim 4 --epochs 40 --seed 0 --gamma-drive-mode static --num-time-steps 64
  --plv-settle 32 --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0
  --graph-mode learned --graph-top-k 32 --spike-spatial-grid-size 16 --k 256
  --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0
  --gate-mode raw --plv-source phase --plv-combine mean --loss-signal sigmoid_membrane
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0
  --structural-weight 0 --verbose --spike-per-component --geodesic-steps 3
  --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --spike-plv-weight 5.0
  --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0 --plv-balance-weight 10.0
  --plv-coherence-weight 0.5 --plv-target-density 0.867 --geodesic-chunk 64
  --gamma-seq-path $G"

cd /export_home/tkim1/snn_kuramoto_bidirectional
for b in 0.0 0.25 0.5 1.0; do
  name=SGN_b${b/./}
  mkdir -p $R/$name
  t0=$(date +%s)
  $PY training/train_s2net_core.py $BASE --coupling-balance $b \
     --save-path $R/$name/core.pt > $R/$name.log 2>&1
  say "--- $name  $(( ($(date +%s)-t0)/60 )) min  $(grep -oE 'loss: [0-9.]+ -> [0-9.]+' $R/$name.log | tail -1) ---"
  grep -oE "spike_plv_collapse=[0-9.]+" $R/$name.log | tail -1 | sed 's/^/    /' | tee -a $OUT
done

say ""
say "label                                fg_ari   fg_iou  obj_iou  pred_fg"
for b in 0.0 0.25 0.5 1.0; do
  name=SGN_b${b/./}
  ev "$name components" $R/$name/core.pt $b --synchrony-threshold 0.05
  ev "$name spectral k=14" $R/$name/core.pt $b --spectral-k 14
done

say ""
$PY - <<'EOF' 2>&1 | tee -a $OUT
import re
rows = {}
for line in open("/work/USERS/tkim1/runs/pv2_0092_results.txt"):
    m = re.match(r"SGN_b(\S+) (\S+)\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", line)
    if m:
        rows[(m.group(1), m.group(2))] = tuple(float(m.group(i)) for i in (3,4,5,6))

print("THE GATE (PROPOSAL_0.9_v2.md), applied to the components readout:")
print("  1. gain must appear with predicted foreground <= 0.20")
print("  2. must not be bought with foreground IoU")
print("  3. matched_object_iou must move too")
print()
base = rows.get(("00", "components"))
if not base:
    print("  the balance 0.0 control is missing; nothing can be judged.")
else:
    print("  control (balance 0.0): fg_ari %.4f  fg_iou %.4f  obj_iou %.4f  pred_fg %.4f" % base)
    passed = []
    for b in ("025", "05", "10"):
        v = rows.get((b, "components"))
        if not v: continue
        d = [v[i] - base[i] for i in range(4)]
        gate1 = v[3] <= 0.20
        gate2 = d[1] > -0.02
        gate3 = d[2] > 0.005
        print("  balance 0.%s: %+.4f fg_ari %+.4f fg_iou %+.4f obj_iou | pred_fg %.4f | "
              "extent %s  iou-kept %s  obj %s"
              % (b, d[0], d[1], d[2], v[3],
                 "ok" if gate1 else "NO", "ok" if gate2 else "NO", "ok" if gate3 else "NO"))
        if gate1 and gate2 and gate3 and d[0] > 0.005:
            passed.append((b, d[0]))
    print()
    if passed:
        b, d = max(passed, key=lambda x: x[1])
        print("SIGNED COUPLING PASSES THE GATE at balance 0.%s, %+.4f fg_ari." % (b, d))
        print("Promote to seeds 0, 1, 2, then select on validation, then read test once.")
    else:
        print("Signed coupling does not pass the gate at seed 0. The mechanism is real --")
        print("the group-separating drive is 140x larger -- but it does not convert into")
        print("the score, which is the fifth time a mechanism-level improvement has not.")
EOF
say "=== PV2_0092 DONE ==="

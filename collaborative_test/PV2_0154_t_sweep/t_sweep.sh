#!/bin/bash
# PV2_0154: how short can the rollout be? T = 1024 (today) down to 128, CLEVR first.
#
# The existing caches cannot answer this. cache_any.py stores activity_features(trace), and
# head.py:12-23 collapses the time axis -- the settled window is averaged into 32 bins -- so a
# shorter T cannot be recovered from them. The rollout has to be re-run per T, which is cheap
# because its cost is linear in T: measured ~11 s/frame at 1024, so ~1.4 s/frame at 128.
#
# `settle` was hardcoded at 512 of 1024, the second half. A fixed 512 is impossible for T<512,
# so the sweep keeps the rule: settle = T//2 at every T (patch_steps.py).
#
# The core is the unchanged 253 CLEVR core, so this measures the readout's sampling window and
# nothing else. Training is untouched: PV2_0021-22 found that training AT the readout's window
# collapses, which is a different experiment from evaluating at a shorter one.
set -u
C=/work/USERS/tkim1/sw0255b/benchmark_codes
RUN=/work/USERS/tkim1/sw0255b/run
CFG=$C/configs/frontier.json
P=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
JOB=/work/USERS/tkim1/share/native64_core_253
EV=/work/USERS/tkim1/sw0255b/ev_tsweep
R=/work/USERS/tkim1/share/tsweep
export CUDA_VISIBLE_DEVICES=GPU-50cc59a2-8312-ff02-5889-1becf63c7cd4   # GPU 2
mkdir -p "$EV" "$R"
say () { echo "[$(date +%H:%M:%S)] $*"; }

for T in 512 256 128; do
  say "########## T=$T (settle $((T/2)))"
  for SP in train fixed320; do
    N=$([ "$SP" = train ] && echo 256 || echo 320)
    O=$EV/clevr_${SP}_T${T}
    if [ -f "$O/complete.json" ]; then say "  $SP already cached"; continue; fi
    $P -u $C/cache_any.py --code $C --config $CFG --run $RUN --job $JOB \
       --dataset clevr --split $SP --count $N --min-group 8 --steps $T --out $O \
       2>&1 | tail -2
  done
  say "--- core alone, T=$T"
  $P -u $C/score_cache.py --cache $EV/clevr_fixed320_T${T} --dataset clevr --split fixed320 \
     --code $C --config $CFG --run $RUN --out $R/core_only_T${T}.json
  say "--- core + head, 5 head seeds, T=$T"
  $P -u $C/headseed.py --code $C --config $CFG --run $RUN --dataset clevr \
     --train-cache $EV/clevr_train_T${T} --eval-cache $EV/clevr_fixed320_T${T} \
     --eval-split fixed320 --arm local_rgb_trust --epochs 2 --seeds 5 \
     --out $R/headseed_T${T}.json 2>&1 | tail -6
done

say "=== summary"
$P -u - <<'PY' | tee $R/SUMMARY.txt
import json, pathlib
R = pathlib.Path('/work/USERS/tkim1/share/tsweep')
K = ('fg_ari', 'foreground_iou', 'matched_object_iou')
# T=1024 is the existing measurement, from PV2_0137 / fit5/headseed_clevr.json
ROWS = [(1024, (0.8421, 0.7545, 0.6410), (0.8468, 0.7575, 0.6460), 8.0, 0.2001)]
for T in (512, 256, 128):
    co = R/f'core_only_T{T}.json'; hs = R/f'headseed_T{T}.json'
    if not co.exists():
        continue
    c = json.loads(co.read_text())
    core = tuple(c[k] for k in K)
    head = None
    if hs.exists():
        m = json.loads(hs.read_text())['metrics']
        head = tuple(m[k]['mean'] for k in K)
    ROWS.append((T, core, head, c['mean_regions'], c['predicted_foreground']))
print('CLEVR fixed320, the unchanged 253 core, one seed. settle = T//2 at every T.')
print('T=1024 is the existing measurement; the rest are this sweep.\n')
print(f'{"T":>5} {"core fg_ari/fg_iou/obj_iou":>30} {"+head (5 seeds)":>30} {"regions":>8} {"pred_fg":>8}')
for T, core, head, reg, pf in ROWS:
    h = '/'.join(f'{v:.4f}' for v in head) if head else '---'
    print(f'{T:>5} {"/".join(f"{v:.4f}" for v in core):>30} {h:>30} {reg:>8.1f} {pf:>8.4f}')
base = ROWS[0][1]
print('\ndelta against T=1024 (core alone):')
for T, core, head, reg, pf in ROWS[1:]:
    print(f'  T={T:<5} ' + '/'.join(f'{c-b:+.4f}' for c, b in zip(core, base)))
print('\ntruth foreground on this split is 0.2115.')
PY
say "T_SWEEP_DONE"

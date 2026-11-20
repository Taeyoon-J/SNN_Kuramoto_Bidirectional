# PV2_0139 — retraining Slot Attention on our render

**Status** planned, code written and not yet run
**Blocked on** a frontier login (the ControlMaster expired) and a budget decision

## Why

`BENCHMARK_PLAN_v4.md` calls this the largest remaining fairness gap, and PV2_0134 is why.
The published checkpoint, on our images:

- uses **6.99 of 7 slots on every frame** with near-equal shares
  (0.156 / 0.147 / 0.141 / 0.148 / 0.133 / 0.131 / 0.144) — it is tiling, not segmenting;
- shows the same pattern in the older centre-crop run, so this is **not** our preprocessing;
- was trained on the **original CLEVR render**, not `clevr_with_masks`, which its authors warn
  about;
- cannot be rescued by the background rule — all four rules tried give the **identical** FG-ARI,
  because they only touch patches the truth calls background and FG-ARI excludes those.

We currently beat it 0.8468 / 0.7574 / 0.6456 against 0.3747 / 0.2219 / 0.1165. **Beating a
model that is failing to transfer is not beating the model.** Until this row exists, the honest
headline is "we beat the published checkpoint transferred to our render".

## Why code had to be written

Only `model.py` and `utils.py` are present in `/work/USERS/tkim1/baselines/slot_attention`.
No training loop ships with them, so `code/sa_train.py` writes out the published
object-discovery recipe: L2 reconstruction, Adam, linear warmup then exponential decay, slots
resampled every forward pass.

```
base lr 4e-4, batch 64, 7 slots, 3 iterations,
warmup 10,000 steps, decay rate 0.5 every 100,000 steps
```

## The fairness decisions, made explicitly

**Same images.** `code/sa_dump_train.py` writes the **identical 10,000 TRAIN ids the 253 core
trains on** — same `RandomState(2553500)` subset of the same 35,000 parent pool, same 128×128
frames from `dataset.rgb`. Slot Attention sees neither more nor less data than we do, so the
only thing differing between the two columns is the model. Giving it the full 35,000 would be
more generous but would stop the row from being a same-data comparison; that variant can be run
separately if we want it.

**Budget is a flag, not a constant.** The published budget is 500,000 steps, which is days per
seed on a TITAN RTX. `--steps` makes a shorter run possible, and `--save-every` writes
checkpoints throughout so one run can be scored at several budgets without retraining and
resumes rather than restarts if interrupted.

**What is not acceptable:** reporting a short run simply as "Slot Attention". The step count,
the exposure count and the wall time go into `provenance.json` and **must appear in the
reported row**. `sa_train.py` says so in its own docstring so the rule travels with the code.

## Cost

At batch 64 on a TITAN RTX, measure the step rate before committing. Rough shape:

| budget | exposures | note |
|---|---|---|
| 500,000 steps | 32M | published; days per seed, over a week for three |
| 100,000 steps | 6.4M | a defensible reduced reference if labelled |
| 1,563 steps | 100,000 | matches *our* exposure count exactly, and is far too few for Slot Attention — listed only to show why exposure-matching would be the unfair choice in the other direction |

Recommended: **reduced budget, three seeds, labelled as reduced.** It makes the render matched,
which is the part that actually threatens the comparison, at a fraction of a week.

## Run

```bash
B=/work/USERS/tkim1/sw0255b
SA=/work/USERS/tkim1/baselines/slot_attention
PY=$B/../envs/miniforge3/envs/snn/bin/python
TF=$B/../envs/miniforge3/envs/tfgpu/bin/python
CFG="--code $B/benchmark_codes --config $B/benchmark_codes/configs/frontier.json --run $B/run"

$PY  $B/benchmark_codes/sa_dump_train.py $CFG --split train --out $B/sa/frames_train.npy
CUDA_VISIBLE_DEVICES=2 $TF $B/benchmark_codes/sa_train.py \
  --frames $B/sa/frames_train.npy --model-dir $SA --out $B/sa/retrain_seed0 \
  --steps 100000 --seed 0
```

Then score it exactly as the published checkpoint was scored — `sa_infer.py` against
`--checkpoint-dir $B/sa/retrain_seed0`, then `sa_bg.py` to choose the background rule on
validation — so both reference rows go through the identical path.

## Status of the pieces

| file | state |
|---|---|
| `code/sa_dump_train.py` | written, parses, **not run** |
| `code/sa_train.py` | written, parses, **not run** |
| step rate | **unmeasured** — needs a GPU |

Nothing here has touched a GPU. The step rate and therefore the real cost are unknown until it
runs, and the table above is a shape, not a measurement.

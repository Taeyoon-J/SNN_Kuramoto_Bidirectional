# PV2_0138 — 261 trained natively on Tetrominoes and Multi-dSprites, 10,000 × 10

**Status** running (launched 2026-11-05)
**Parent** PV2_0134 (transfer rows), PV2_0136 (the label-free recipe)
**Seeds** 0 only, by decision. One run per dataset, the two in parallel.

## What "261" means here

Not the peer's 261. The peer's is a **native16 core state lifted to grid 64** by `parent_index`
(`axis = arange(64)//4`, with omega/kappa/tau tied in 4×4 blocks and `direction_learner` built
by `index_select` on both axes of a 256×256). That is a resolution conversion of an existing
checkpoint.

Here 261 is defined as **the core trained from scratch, natively at grid 64, by 253's
unsupervised recipe — no masks, no teacher — plus the SW0260 readout.** No lifting.

There is no `--model 261`: the trainable variants are `{252, 253, 254, 240}`, because 261 is not
a core variant at all, it is 253's core plus a readout. So the core is trained with
`--model 253` and the 261 readout goes on afterwards.

## Why this run exists

Every Tetrominoes and Multi-dSprites number in PV2_0133–PV2_0137 comes from **one
CLEVR-trained core reused by transfer**, with only the head refit. That answers "how well does
CLEVR transfer". It does not answer "how good is the method when the core sees the target
domain", which is the stronger claim and the one this run measures.

It is also distinct from the three-seed per-dataset cores already on disk at
`/work/USERS/tkim1/sw0255/run/jobs/model253/*/seed{0,1,2}`, which are **3,500 unique × 10
epochs** (2,190 updates). This is the **matched 10,000 × 10** budget, a 2.86× increase, so for
the first time a per-dataset core is directly comparable with 253-on-CLEVR.

## Budget, verified rather than assumed

`SW_TRAIN_PRESET` is deliberately unset, which selects the default preset. Read back from each
run's own `binding.json`:

| | unique_TRAIN | len(train_ids) | subset_seed | parent pool |
|---|---|---|---|---|
| clevr seed 0 (reference) | 10,000 | 10,000 | 2553500 | 35,000 |
| **tetrominoes (this run)** | **10,000** | **10,000** | **2553500** | 35,000 |
| **multi_dsprites (this run)** | **10,000** | **10,000** | **2553500** | 35,000 |
| tetrominoes, older core | 3,500 | 3,500 | 2553500 | 35,000 |

Identical selection procedure and identical count to 253-on-CLEVR. Batch 16 × 10 epochs →
**6,250 updates**, matching seed 0 exactly.

## Commands

```bash
B=/work/USERS/tkim1/sw0255b
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
unset SW_TRAIN_PRESET
CUDA_VISIBLE_DEVICES=2 $PY $B/benchmark_codes/run.py \
  --config $B/benchmark_codes/configs/frontier.json \
  --output $B/run_261_tetrominoes    train --dataset tetrominoes    --model 253 --seed 0
CUDA_VISIBLE_DEVICES=3 $PY $B/benchmark_codes/run.py \
  --config $B/benchmark_codes/configs/frontier.json \
  --output $B/run_261_multi_dsprites train --dataset multi_dsprites --model 253 --seed 0
```

Logs `core261_tetrominoes.log`, `core261_multi_dsprites.log`. Measured **29.5 s/update
(51.1 h)** and **30.5 s/update (52.9 h)**; in parallel on GPUs 2 and 3, about **2.2 days**.

## Steps after the cores finish

1. Cache activity (`features`, `gamma`, `regions_253`, `regions_all`) with `cache_any.py`
   at `--min-group 8`, on train / validation / test.
2. Fit the 261 readout — `(prior + 2·tanh((sim − mean)/std))` masked to the locality radius,
   `prior_scale` 4.0 and `residual_scale` frozen, `content_encoder` frozen. This is what
   `fit5.Readout` implements, verified against SW0260's form and against `native_routing`.
3. Score under the patch contract at grid 64 on the **same 320 evaluation images** the existing
   transfer rows use, so the two columns are directly comparable. Note these datasets have
   **1,000** validation and 1,000 evaluation ids available; extending both columns to the full
   1,000 is a follow-up, and must be done for *both* or neither.

## The comparison this produces

| dataset | core trained on | fg_ari | fg_iou | obj_iou |
|---|---|---|---|---|
| Tetrominoes | CLEVR, transferred (PV2_0134) | 0.8580 | 0.7885 | 0.8173 |
| Tetrominoes | **itself, 10,000 × 10 (this run)** | pending | pending | pending |
| Multi-dSprites | CLEVR, transferred (PV2_0134) | 0.6974 | 0.8177 | 0.7628 |
| Multi-dSprites | **itself, 10,000 × 10 (this run)** | pending | pending | pending |

Both rows will carry the 261 readout fitted the same way, so the only difference between them is
what the core trained on.

## Note

CLEVR seeds 1 and 2 of the 253 core were stopped to free the GPUs for this, by decision. They
reached 20 and 25 updates of epoch 1; `run_s1/` and `run_s2/` are kept so they resume rather
than restart. The 3-seed requirement is therefore still open.

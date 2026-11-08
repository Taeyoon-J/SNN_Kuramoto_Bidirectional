# PV2_0148 — a `cont5000` preset: continue a CLEVR core on another dataset

**Status** implemented and smoke-tested; **not yet run for real, and not yet pushed** — it
edits the peer's vendored engine, so it is flagged before landing.
**Purpose** take the finished 10,000 × 10 CLEVR grid-64 core and continue it for 5,000 × 10 on
Tetrominoes, then continue *that* for 5,000 × 10 on Multi-dSprites — **sequential and
cumulative**, one core that has seen all three, with the intermediate checkpoint pulled down
between stages.

Parallel is not an option here and that distinction matters: running the two continuations side
by side would produce **two separate branch models**, not one core that learned both. Sequential
chaining is what gives a single cumulative core, so one GPU is enough.

## What was blocking it

**Three** things, all now solved. Two of them would have fired only at the *second* stage, after
the first had already spent 5–9 h.

**1. There was no 5,000 preset.** `data.py` accepted only the default (10,000), `cont15000` and
`union25000`, and raised `ValueError` on anything else.

**2. "Warm start on a different dataset" looked absent** — but it is not. `engine.py:20` has
`WARM = PRESET=='cont15000'`, and `WARM` already does exactly what is needed: it relaxes the
binding assertions at `:100` and `:118` so a checkpoint from another dataset loads, and at
`:123-129` it resets `next_epoch`/`next_offset` when the resumed counters do not match the new
schedule. So the weights carry over while the data changes. `cont15000` simply never pointed it
at another dataset.

**3. Two guards hardcoded the single-stage case.** `engine.py:185` asserts the final
`updates`/`seen` equal module constants, and `updates`/`seen` are carried forward from the
resumed checkpoint (`:122`), so a fixed 9,380 / 150,000 is correct for stage 1 and wrong for
every stage after. And `engine.py:145` guards warm-start entry with
`updates==6250 and seen==100000`, which admits only the base core. Both were found by
fabricating a stage-1 checkpoint instead of waiting 9 h to discover them.

## The change

Additive only; no existing preset's behaviour moves. Full diff in `code.patch`, backups on the
server at `*.py.bak`.

- `engine.py`: `WARM = PRESET in ('cont15000','cont5000')`, and a new branch
  `PRESET_UNIQUE, PRESET_EPOCHS = 5000, 10` with cumulative
  `PRESET_UPDATES, PRESET_SEEN = 6250+313*10, 100000+50000` → **9,380 / 150,000**.
  313 updates per epoch is 5,000/16 = 312.5 rounded the way `cont15000` rounds 937.5 to 938.
- `engine.py`: the stage increment is a constant (`PRESET_ADD_UPDATES, PRESET_ADD_SEEN =
  313*10, 50000`) and the **expected totals are computed from what was resumed**, so the chain
  works at any depth. `training_report.json` records the real cumulative `seen` and the stage
  increment rather than a fixed constant.
- `engine.py`: the warm-start entry guard now admits, **for `cont5000` only**, the base core or
  the base core plus a whole number of cont5000 stages. `cont15000`'s strict
  `updates==6250 and seen==100000` is untouched.
- `data.py`: `cont5000` takes `first[:5000]` — the first 5,000 of the **same**
  `RandomState(2553500)` draw the default preset uses, so the run sees a strict subset of the
  10,000 and the ids stay reproducible.

## Chaining verified without spending the 9 h

`chaintest.py` fabricates a checkpoint at each stage's entry point and runs the engine far
enough to print its computed targets:

```
stage 1: resumed  6250/100000 -> targets  9380/150000   entering stage 1
stage 2: resumed  9380/150000 -> targets 12510/200000   entering stage 2
```

Both pass. Before the fixes, stage 2 aborted with `AssertionError: (9380, 150000)`.

## Smoke test — it works

```
unique_TRAIN 5000   len(train_ids) 5000
composition: first 5,000 of the default 10,000 subset; warm start from a 10,000x10 core
```

**Warm start confirmed by the loss, not by inspection:** `positive_spike` starts at **3.68**,
where training Tetrominoes from scratch starts it at **15.37**. The weights carried.

```bash
mkdir -p OUT/jobs/model253/<ds>/seed0
cp share/native64_core_253/{initial.pt,resume.pt} OUT/jobs/model253/<ds>/seed0/
SW_TRAIN_PRESET=cont5000 python run.py --output OUT train --dataset <ds> --model 253 --seed 0
```

## The normalization question, measured rather than argued

`initial.pt` carries CLEVR's feature statistics, and `model.py:73` standardises with them and
then **clamps to ±3**. The worry was that another dataset's features sit at a different scale,
get pushed outside ±3, saturate, and lose their gradient. Measured on 64 images per dataset
through the same encoder:

| dataset | own std | ratio to CLEVR's 0.2478 | **% clamped with CLEVR's stats** | % clamped with its own |
|---|---|---|---|---|
| CLEVR | 0.2462 | 0.99× | 0.00% | 0.00% |
| Tetrominoes | 0.1675 | 0.68× | **0.17%** | 1.66% |
| Multi-dSprites | 0.3033 | 1.22× | **0.04%** | 0.01% |

**Keep CLEVR's normalization.** The loss is negligible, and on Tetrominoes CLEVR's statistics
actually clamp *less* than that dataset's own (0.17% against 1.66%), because its features are
narrower and dividing by a smaller std widens them past the clamp. I had raised this as a design
decision "where the results differ"; measuring it shows they do not.

## Cost on a rented RTX 4090 — 3,130 updates per dataset

A correction first: the 2.8× spread we measured on frontier is **machine load, not dataset**. A
single uncontended forward costs the same on both (1.75 s CLEVR, 1.74 s Tetrominoes), and
CLEVR seed 0's clean full run recorded **10.66 s/update** (66,602 s / 6,250) where the recent
contended runs read 29–33 s/update.

| | per stage | both stages | cost |
|---|---|---|---|
| frontier, uncontended basis (10.66 s/upd) | 9.3 h | 18.5 h | $0 |
| frontier, current load (29.4 s/upd, measured) | 25.6 h | 51 h | $0 |
| **RTX 4090, one GPU** | **5–9 h** | **10–18 h** | **$4–8** |

**One GPU is enough** — the stages are sequential by construction, so a second GPU would sit
idle.

The case for renting is the **dedicated machine**, not raw speed: frontier's 2.8× spread is
contention, which a rental removes, and the 4090's higher clock helps on top because this
workload is clock- and CPU-bound rather than bandwidth-bound (the H200 lesson).

**Measure 20 updates immediately after renting** — that turns the 5–9 h range into one number in
about a minute.

Upload is ~760 MB (inputs 250 MB + core 513 MB). **Test the link with ≥1 GB first**: a host that
read 200 MB at 14.5 MB/s sustained only 1.9 MB/s.

## Evaluation design — the question a cumulative core raises

A core that has seen all three datasets has to be scored on all three, because the real question
is whether continuing cost it what it already had:

| after stage | score | the number to watch |
|---|---|---|
| 1 (CLEVR + Tetrominoes) | Tetrominoes **and CLEVR** | does CLEVR's 0.8421 survive? |
| 2 (+ Multi-dSprites) | all three | do 0.8421 and stage 1's Tetrominoes figure both survive? |

The intermediate checkpoint is pulled down between stages, so each can be scored at that point.
About 30 min per dataset with `score_any.py` and the existing cache path.

## Before this lands

It modifies `vendor/full_benchmark/{engine,data}.py` in **three** places, which is the peer's
bundle. Every change is additive or scoped to `PRESET=='cont5000'`; no existing preset's
behaviour moves, and `*.py.bak` holds the originals on the server. It should be agreed before
pushing, and the peer should see `code.patch`.

## Reproduce

```bash
python3 norm_effect.py   # the clamping measurement
python3 profile1.py      # forward cost per dataset
python3 clevr_rate.py    # wall-clock rates from each run's own report
```

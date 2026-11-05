# PV2_0140 — the project's first genuine 3-seed table, from cores already on disk

**Status** planned, code written and not yet run
**Blocked on** a frontier login, and it should queue behind `PV2_0138` rather than contend
for the same GPUs

## The gap this closes, and the one it does not

Everything this project has reported is **seed 0**. The contract judges success on 3-seed
training means of all three metrics with per-seed values and standard deviations alongside, and
no such table has ever been produced here.

There are three real training seeds already on disk:

```
/work/USERS/tkim1/sw0255/run/jobs/model253/{tetrominoes,multi_dsprites}/seed{0,1,2}
    updates 2190   images_seen 35000   unique_TRAIN 3500   epochs 10   resume.pt present
```

Read back from each run's own `training_report.json`, so these are three independent trainings,
not one checkpoint scored three times.

**What they support.** "A 253-style core trained on its own dataset at 3,500 × 10, plus the 261
readout" — as a 3-seed mean with standard deviation, on all three metrics.

**What they do not support, and must never be used for.** The headline claim is *one
CLEVR-trained core reused on other datasets*. That core exists at **seed 0 only**. These cores
were each trained on their own dataset, at a smaller budget, so putting them in the
generalisation column would answer a question nobody asked and would misstate the result.
`BENCHMARK_PLAN_v4.md` says the same thing; this record repeats it because the temptation is
obvious and the mistake would be invisible in the final table.

## Why run it at all, then

Two reasons, both honest.

1. It is the first time this project produces the **reporting format the contract requires** —
   per-seed, mean, std, three metrics, never combined. If that format is wrong, better to find
   out on a side claim than on the headline.
2. It gives a real measurement of **seed spread** for this architecture on these datasets. Every
   seed-0 number we hold is currently a point estimate with no idea of its width. Knowing the
   standard deviation tells us whether differences like Tetrominoes' +0.0006 FG-ARI are
   meaningful at all — and my expectation is that it will show several of our reported deltas
   are inside the noise.

## Guards in the aggregator, and the one that was missing

`code/agg3.py` refuses to emit a mean when the results are not what an N-seed mean requires.
The guards are **tested**, not assumed — `tests/run_guard_tests.sh` feeds it fixtures and
checks each refusal actually fires:

| case | behaviour |
|---|---|
| three distinct runs | accepted |
| **one run copied three times** | **refused** |
| two results when three expected | refused |
| results spanning two datasets | refused |

The second one was **missing until the test was written.** The README claimed the aggregator
rejected "one checkpoint evaluated three times", but that sentence lived only inside the
*message* of the count check — nothing implemented it. Fed three copies of one result, the
aggregator averaged them cleanly and emitted a "3-seed mean" with standard deviation
**0.0000**, which is precisely the artefact the contract forbids. It is now checked three
ways: identical run identity, non-distinct seeds, and all metrics numerically identical across
results.

## Cost

Per seed per dataset: 256 + 160 + 320 = 736 rollouts at ~3.1 s ≈ 38 min of caching, then about
a minute of head fitting. Six combinations ≈ **3.8 h on one GPU**, under 2 h on two. Cheap
relative to anything else outstanding.

## Precondition

Same one as `PV2_0138`: `cache_seed2.py` hardcodes `jobs/model253/clevr/seed0`, so if
`cache_any.py` does too it would load the CLEVR core whatever `--run` says and cache it under
six different names. `code/seeds3.sh` refuses to start unless `cache_any.py` exposes a flag
pointing at a specific checkpoint. **Unverified** — ssh expired before `cache_any.py` could be
read.

## Run

```bash
# after PV2_0138's cores finish
nohup bash /work/USERS/tkim1/sw0255b/seeds3.sh 2 > /work/USERS/tkim1/sw0255b/seeds3.log 2>&1 &
```

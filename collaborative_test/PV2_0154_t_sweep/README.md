# PV2_0154 — how short can the rollout be? Half is free; a quarter is not; and shortening it saves nothing

**Status** completed (second run; the first was wrong, see below). CLEVR only, the unchanged 253
core, one seed; ± on the head rows is head re-initialisation variance over 5 seeds.

**Question.** T, the rollout length, is 1,024. Two motives for shortening it: (a) does a shorter
sampling window still support the readout, and (b) would it make every remaining experiment
cheaper, since caching at ~11 s/frame is the project's bottleneck.

**The existing caches cannot answer (a).** `cache_any.py` stores `activity_features(trace)`, and
`head.py:12-23` collapses the time axis — the settled window is averaged into 32 bins — so a
shorter T cannot be recovered by truncation. The rollout has to be re-run per T.

**`settle` follows T.** It was hardcoded at 512 of 1,024, the second half. A fixed 512 is
impossible for T < 512, so the sweep keeps the rule rather than the number: **settle = T // 2**.
`bins=32` needs `settle % 32 == 0`, which holds down to T = 64.

## Result — CLEVR fixed320, core alone and with the label-free head

| T | core fg_ari / fg_iou / obj_iou | + head (5 seeds) | regions | pred fg |
|---|---|---|---|---|
| **1024** | 0.8421 / 0.7545 / 0.6410 | 0.8468 / 0.7575 / 0.6460 | 8.0 | 0.2001 |
| **512** | **0.8402 / 0.7594 / 0.6340** | 0.8437 / 0.7628 / 0.6387 | 9.8 | 0.1936 |
| 256 | 0.8159 / 0.7358 / 0.6084 | 0.8233 / 0.7473 / 0.6179 | 13.3 | 0.1851 |
| 128 | 0.5796 / 0.4988 / 0.3985 | 0.7326 / 0.7117 / 0.5298 | 12.5 | 0.1148 |

Core alone, against T = 1024:

| T | Δ fg_ari | Δ fg_iou | Δ obj_iou |
|---|---|---|---|
| 512 | **−0.0019** | **+0.0049** | −0.0070 |
| 256 | −0.0262 | −0.0187 | −0.0326 |
| 128 | −0.2625 | −0.2557 | −0.2425 |

**Halving the rollout is free.** FG-ARI moves −0.0019, which is below every noise scale measured
in this project, and Foreground IoU actually *improves* by 0.0049. A quarter costs 0.02–0.03.
An eighth collapses.

Truth foreground on this split is 0.2115.

## Answer to (b): no. Shortening T saves almost nothing

| T | s / frame |
|---|---|
| 1024 | 10.8 |
| 512 | 10.0 |
| 256 | ~8.8 |
| 128 | ~8.7 |

**An 8× cut in rollout steps buys 19%**, and the 1024 → 512 step that is free in accuracy buys
**7%**. The rollout is not the bottleneck; the per-image fixed cost is — encoder, graph
generation, the partition merge, and I/O. The premise that motivated this experiment was wrong,
and it was my premise: I wrote that "its cost is linear in T, so ~1.4 s/frame at 128". It is not.

**So T = 512 is the right default for anything new** — it is free in accuracy and 7% cheaper —
but it does not change what is affordable.

## Two things the sweep shows that were not the question

**1. Shorter windows fragment more.** Regions per image rise 8.0 → 9.8 → 13.3 as T falls from
1,024 to 256, against a truth of 6.36 (`PV2_0157`). A shorter settled window gathers less
synchrony evidence, so groups that would have merged stay separate. At T = 128 the count falls
back to 12.5 only because the foreground itself has shrunk — predicted foreground drops to
**0.1148 against a truth of 0.2115**, a severe under-claim, which is the first time this readout
has under-claimed foreground at all.

**2. The head's value is a measure of how broken the core is, confirmed on a new axis.**

| T | head's Δ fg_ari |
|---|---|
| 1024 | +0.0047 |
| 512 | +0.0035 |
| 256 | +0.0074 |
| 128 | **+0.1530** |

At T = 128 the head lifts FG-ARI from 0.5796 to 0.7326 — **33× its contribution at full length**.
`PV2_0147` found exactly this on a different axis (the head repairs a broken core and barely
touches a healthy one, 104× on the fragmented Multi-dSprites core against 0.3–2% of patches on
the CLEVR core). Here the same pattern appears by degrading one healthy core instead of
comparing two different cores, which is a cleaner demonstration of it than PV2_0147 had.

## The first run was wrong, and how

The first pass reported **exactly 0.0000** on all three metrics at every reduced T, with zero
regions. Not a decline — a collapse, which is what prompted checking rather than reporting.

`features` were healthy at every T (mean ≈ 0.5, std ≈ 0.3, no zeros), so the spikes were fine;
only `regions_253` was empty. Cause: `cache_any.py` called
`spike_synchrony_components(..., settle=512, ...)` with **512 hardcoded** — a third hardcoded
`settle` beyond the two in `model.py`. At T ≤ 512 that leaves no settled window, the foreground
comes back empty, and `partition.py:40` returns all-background by design.

This was my error, and specifically the error my own patch warned about: `patch_steps.py`'s
docstring says "T is hardcoded in three places that must move together" and then moves two. The
fix threads `settle` through from the model output; the default-T path was re-verified
**bit-identical** (features and `regions_253` both, max diff 0.0) before rerunning, because the
scaling-curve run on another GPU shares this file. The bad log is kept as
`t_sweep_WRONG_settle512.log`.

## What this licenses

**Licenses:** "the readout needs about 512 settled steps; at 256 it loses 0.02–0.03 and at 128 it
fails. Shortening the rollout does not make the pipeline meaningfully cheaper."

**Does not license** retraining at a shorter T. `PV2_0021-22` found that training *at* the
readout's window collapses; this experiment changes evaluation only, on a core trained at 1,024.
One seed, CLEVR only.

## Reproduce

```bash
bash /work/USERS/tkim1/t_sweep.sh          # T = 512, 256, 128; settle = T//2
# patch_steps.py opens total_steps in model.py and --steps in cache_any.py, both defaulting
# to today's 1024, and threads settle through to activity_features and
# spike_synchrony_components.
```

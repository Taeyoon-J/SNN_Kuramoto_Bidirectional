# Benchmark plan v7

Supersedes v6. Written after `PV2_0155`, `PV2_0158` and `PV2_0159`, and after re-reading
`PV2_0136`, which already contained the answer to three of the four questions the standing goal
asks and had never been carried into a plan document.

**Everything in v5 §1–§2 (protocol, best-baseline-per-cell, per-model budget verdict) stands.**
v6 §2–§4 stand. This document answers the four standing questions in §1–§4, corrects two things
v6 got wrong in §5, and replaces v6 §7–§8 with §6–§7.

---

## 1. Why 252 lost to 253, when the data was identical

Source: `PV2_0136`, read from both runs' own `training_report.json` and `training.jsonl`.

The budgets were the same to the image:

| | 252 | 253 |
|---|---|---|
| updates | 6,250 | 6,250 |
| images seen | 100,000 | 100,000 |
| unique train images | 10,000 | 10,000 |
| epochs | 10 | 10 |

**So the data is not the explanation, and "same dataset size, different result" is answered by
what the gradient did, not by how much data there was.** First 50 updates against last 50:

| component | 252 | 253 |
|---|---|---|
| `phase_primary` | 7.2389 → 7.0511 (−2.6%) | 4.6355 → 2.4556 (**−47.0%**) |
| `positive_spike` | 15.6905 → 10.4814 (−33.2%) | 15.3876 → 2.5839 (**−83.2%**) |
| `gradient_norm` | 47.96 → 247.27 (**+416%**) | 6.08 → 6.27 (+3.2%) |

253 drove both objectives down with a flat gradient norm — a converging run. 252 moved both
barely while its gradient norm quintupled — a run pulling against itself.

The parameters confirm it. 252's **core received 3.7× the gradient and moved 0.62× as far**; its
encoder received **65×** the gradient and moved *less* in relative terms. More gradient
producing less motion is cancellation. The one part that did move was the head: 177% mean
relative change, one tensor 566%, sitting on a core that moved 7%.

**Rule.** Equal data budgets do not imply comparable runs. Report `gradient_norm` trend and
parameter movement per unit of gradient beside every training comparison; they identified this
failure without the evaluation at all.

---

## 2. Why the supervised variant scored *below* the unsupervised one

Same source, and it is structural rather than a tuning accident. Supervision enters 252 by two
routes and neither can improve what is scored.

**Route A — the mask teacher is detached.** `loss.py:63-67`:

```python
features = activity_features(output['trace'])[0]
if model.variant == 252: features = features.detach()
```

`detach()` severs the gradient between head and trace. The mask labels train the head and
**cannot reach the core at all**. That term improved 52.9%, which is exactly why the head moved
177% — all of it went there. At evaluation the score is read from the core's spikes, so the
improvement is spent on a component the metric never sees.

**Route B — the gamma teacher does reach the core, and went the wrong way.**

| | first 50 | last 50 | change |
|---|---|---|---|
| `gamma_anchor` | 0.00016 | 0.01591 | **+9,907%** |
| `frequency_denoising` | 0.11854 | 0.13893 | +17.2% |

The term carrying label information into gamma ended **62× worse** than it started, while
consuming gradient the core objective needed.

**252 bought a better head on a worse core, and the score is read from the core.**

The symptom was visible in its own evaluation curve: Foreground IoU was **bitwise identical
(0.2021225) at all five checkpoints** — not a plateau but a constant, because the readout
designated no background and `prediction != 0` covered the whole image. FG-ARI rising 0.4454 →
0.5239 underneath that is not partial progress; it is two metrics improving inside a
segmentation that never separated figure from ground.

**Rules.**
1. A supervised term that reaches only a detached head is not supervision of the model. If the
   score is read from spikes, the loss needs a gradient path to the state updates that produce
   spikes.
2. A metric constant to seven decimals across checkpoints is a bug report, not a plateau.
3. Do not quote 252's numbers against 253's: 252's curve was scored on 160 images under the
   `historical_fixed64_patch` contract, 253's on 320 under the patch contract at grid 64. The
   diagnostics above are the comparable evidence.

---

## 3. A third case of the same disease, found this round

`PV2_0159` ran the identical readout on `component_gates` instead of `component_spikes`. **Every
mask is reproduced exactly** — 40 images, three metrics, 16 significant figures. Because
`s_ia = g_i·e_ia` and the firing indicator `e` is 1 in **99.992%** of settled steps, `s` tracks
`g` at correlation **0.9997**, and `g` is a deterministic function of the Kuramoto phase.

The spiking layer is not selecting anything; it transmits the phase gate. The mask is computed
from `s`, which is what is cached and scored, but `s` **is** `g`. This is the same failure mode
as §2 in a different place: **a component that is in the diagram and in the name, but carries no
gradient-bearing or information-bearing role in what is measured.**

The same round also showed, by per-patch circular shift, that *relative timing* is necessary:
FG-ARI .8648 → .0735, regions 7.8 → 562, while Foreground IoU falls only .803 → .736 because
the foreground test is a mean and a rotation does not change a mean. The timing claim survives;
the *spike* claim does not, and the paper now says so.

**Rule.** For every named component, run the ablation that removes it while holding everything
else fixed. If the output does not change, the component is not part of the result, whatever
the architecture diagram says.

---

## 4. Two corrections to v6

**v6 §5 scaling reading is withdrawn.** v6 said FG-ARI peaks at 14,000 and turns down 0.0010.
At nine points (`PV2_0155`) 17,000 scores .8470, above the 14,000 value. There is no turn:
FG-ARI is flat inside one seed's noise from 12,000 on. The *conclusion* survives and is
stronger — Foreground IoU (+.0210) and matched-object IoU (+.0169) are still climbing at 18,000.

**v6 §1 made object count a required metric without its caveat.** `PV2_0158` found the caveat:
**every fixed-`k` baseline scores 100% exact count on Tetrominoes**, CtCAE included at FG-ARI
0.0799, because Tetrominoes contains exactly three objects in every image and "always answer
three" is exactly right without grouping anything. Count accuracy is **degenerate wherever the
true count is constant**. It stays required, but it is only *informative* on datasets whose
count varies, and every report must say which case it is in.

---

## 5. The transfer claim is stronger than "zero-shot" says, and we were not stating it

`PV2_0158`'s fairness work turned up the asymmetry the paper now records in Appendix E.

| | cluster count `k` | background mapping |
|---|---|---|
| **Ours** | — | `f = 0.01`, fixed on **CLEVR validation only**, transferred unchanged |
| **Baselines** | **each target dataset's own validation** | **each target dataset's own validation** |

And the selection does not land on one value: CAE++ uses `k` = 11 / 11 / **7** across CLEVR /
Tetrominoes / Multi-dSprites, CAE 11 / 11 / **4**, and the background rule differs per model and
per dataset.

**"Not trained on the target" and "not tuned on the target" are different claims, and only we
satisfy the second.** A benchmark that says "zero-shot" without separating them understates its
own result and invites the reviewer to assume the weaker reading.

---

## 6. How benchmark testing should go from here

Ordered by what the evidence above says costs least and buys most.

1. **Score the partition and the count as two separate questions, and say when the count is
   degenerate.** Three partition metrics + count, per dataset, with a line stating whether the
   true count varies on that dataset.
2. **Report the composite, not a single rival.** Best-baseline-per-cell across all evaluated
   baselines; no single architecture attains that row, which makes it the honest bar. Our nine
   margins against it are +.0410 to +.3261 at seed 0.
3. **State the tuning split for every free choice of every model**, ours included (§5). One
   table, one line per model per dataset.
4. **Offer every post-processing rule to every baseline before claiming a margin.** `PV2_0158`
   did this and found one real baseline gain (AKOrN conv, CLEVR Foreground IoU .4219 → .4267)
   which is now in the composite row.
5. **Publish the component ablations, not only the scores.** The per-patch shift and the
   gate-substitution (§3) are what let a reader tell which part of the model is load-bearing.
   A benchmark of oscillatory/spiking models without them cannot distinguish the mechanism from
   the pipeline around it.
6. **Keep diagnostics that catch a broken run before the metric does**: gradient-norm trend,
   parameter movement per unit of gradient, predicted-foreground fraction, and a check that no
   metric is constant across checkpoints (§1, §2).
7. **Do not design around rollout length.** `PV2_0154`: an 8× cut in `T` buys 19%; per-image
   fixed cost dominates. `T = 512` is the default, `T = 1024` for reported numbers.

### On "Slot Attention must be beaten"

It is beaten, at seed 0, on every cell of all three datasets, twice over: against our matched
budget retraining and against its own released weights at ~320× our exposure budget (margins
0.377–0.720). That is the strongest single-rival statement available and it is not the headline
claim — the composite row in (2) is, because it is harder.

### The limitation that no amount of further benchmarking fixes

**Every number in this project is a single training seed.** Core seeds 1 and 2 were declined on
2026-11-09. `CLAUDE.md` sets the success criterion as 3-seed means of all three metrics, so
**the stated criterion is not met and cannot be met from the current runs.** What partially
covers it: `PV2_0150` measured a baseline reseed at FG-ARI spread 0.1339 on CLEVR, 0.0535 on
Tetrominoes, 0.0094 on Multi-dSprites, and our margins (0.041–0.326) exceed all three. That is
evidence about *baseline* variance; it is not an error bar on our core and must never be
presented as one. Every report says "at seed 0" or does not go out.

---

## 7. Order of work

1. **A readout that uses the firing events** `e` rather than the gated trace `s` (`PV2_0159`
   next proposal). This is the one experiment that would make "spike" load-bearing. Readout
   change only; runs on existing caches.
2. **Apply the component floor after the spatial split** instead of before, which removes the
   single-patch fragments at source and makes `PV2_0158`'s rule unnecessary. Core change —
   analyse, then one small ablation.
3. **Core seeds 1 and 2**, if the scope decision is ever revisited. Nothing else closes the gap
   to the stated criterion.
4. **A second AKOrN (conv) seed**, 7 minutes; it owns our narrowest margin.
5. Rotating Features and SynCx remain unrun, both oracle-assisted in their published form.

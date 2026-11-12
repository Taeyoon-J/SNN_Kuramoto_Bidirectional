# Benchmark plan v6

Supersedes v5. Written after `PV2_0154`–`PV2_0157`, which produced four results that change what
the benchmark must report rather than only what it reports.

v5 fixed the comparison: best-baseline-per-cell, a per-model budget verdict, and three required
diagnostics. It was written before the object-count measurement, the three-dataset rows for the
continued cores, the rollout sweep and the scaling curve. Each of those changes a rule.

**Everything in v5 §1 (the protocol) and §2 (best-baseline-per-cell, narrowest cell, per-model
budget verdict) stands unchanged.** This document adds to §3 and replaces §4–§6.

---

## 1. A fourth required metric: object count

`PV2_0157` is the reason for this version. Our three contract metrics score the *partition* and
none of them answers "how many objects did the model think there were". That turned out to hide
a large effect:

| | predicted count | truth | exact match |
|---|---|---|---|
| 253 core, CLEVR | 8.04 | 6.36 | 0.175 |
| 253 core **+ head**, CLEVR | **12.90** | 6.36 | **0.035** |
| 253 core, Tetrominoes | **21.07** | **3.00** | **0.000** |

**The head raises the predicted count on every row without exception and cuts exact-match three-
to five-fold.** Its contribution to the three contract metrics is +0.003 to +0.026 — real,
reproducible over five head seeds, and reported as a gain in `PV2_0146` and `PV2_0153`. None of
those three metrics can see what it costs. A benchmark that reports only them says the head is
free, and it is not.

On Tetrominoes the gap between the two readings is total: FG-ARI 0.8580, our best cell anywhere,
and **not one image of 320 with the right object count**.

**Requirement.** Every reported configuration carries object-count accuracy — exact match, and
mean predicted count against the truth's — beside the three metrics. `objcount.py` in
`evaluation_howto/` computes it from the same masks that are scored, so there is no separate
mechanism to validate.

**And the wording changes with it.** "The model finds the objects" is not supported on
Tetrominoes. "The model groups the patches correctly" is. Reports must use the second form unless
the count supports the first.

### Why this is structural, and belongs in the write-up

`model.py` and `loss.py` contain **no decoder and no reconstruction term** — zero matches for
`decoder`, `reconstruct`, `deconv`, `upsample` — and for variant 253 `self.head` is `None`. The
objective is `primary + 5.*spike` over PLV structure and spike synchrony (`loss.py:21`).

Every baseline in the comparison reconstructs the image: Slot Attention's spatial-broadcast
decoder, the CAE family's complex-valued autoencoder MSE, AKOrN and ItrSA's autoencoding. A
reconstruction loss penalises splitting one object in two, because both halves must then
reconstruct the same texture independently. **Our objective contains no such pressure**, so
nothing stops one tetromino becoming seven mutually synchronous fragments. The count errors are a
predicted consequence of the design, not an implementation fault — and the absence of a decoder
is also what makes the model cheap to train. Report both halves of that trade.

---

## 2. Count accuracy and matched-object IoU are not the same axis

They moved in **opposite** directions across three cores on CLEVR (`PV2_0157`):

| core | predicted count (truth 6.36) | count MAE | matched_object_iou |
|---|---|---|---|
| cont5000(Tetrominoes) | **6.65** | **1.99** | **0.6207** (worst) |
| 253 base | 8.04 | 2.88 | 0.6410 |
| cont5000(Multi-dSprites) | 9.90 | 4.20 | **0.6671** (best) |

Monotone and inverted. **A benchmark cannot use either as a proxy for the other**, and an
explanation that invokes one must be checked against the other. `PV2_0152` attributed a
matched-object IoU loss to over-merging; the core in question has the *most accurate* count of
the three, so it was not merging objects away. That mechanism is withdrawn, and the general
lesson is in this rule.

---

## 3. Report a core on all three datasets before comparing it

`PV2_0153` called a core "the best configuration this project has" on the strength of **two**
datasets. `PV2_0156` scored the missing third and it lost two of nine cells; the claim was
withdrawn, and the *other* branch turned out to be the better core under the stated
FG-ARI-first priority.

**Requirement.** No core enters a comparison table until it is scored on all three datasets. A
partial row is reported as partial, never as a headline. This is cheap to satisfy — one 320-frame
cache per missing cell, about an hour — and it is the rule the project broke most recently.

---

## 4. Default rollout length is 512, and shortening it saves nothing

`PV2_0154`: on CLEVR with the model fixed, `T=512` costs **−0.0019** FG-ARI against `T=1024` and
*improves* Foreground IoU by 0.0049. `T=256` costs 0.02–0.03; `T=128` collapses by 0.26 and
under-claims foreground for the first time (0.1148 against a truth of 0.2115).

**Requirement.** `T=512` is the default operating point for anything new; `T=1024` for reported
numbers until a seed says otherwise. `settle = T/2` — the readout reads the second half — and that
rule, not the number 512, is what travels.

**And the cost motive is dead.** Per frame: 10.8 s at `T=1024`, 10.0 at 512, 8.8 at 256, 8.7 at
128. An 8× cut in rollout steps buys **19%**. The rollout is not the bottleneck; per-image fixed
cost is (encoder, graph, merge, I/O). Do not plan experiments around shortening `T` to afford
them.

**One more confirmation worth keeping.** The head's value tracks how broken the core is — head
ΔFG-ARI is +0.0047 at `T=1024` and **+0.1530 at `T=128`**, 33×. `PV2_0147` found this by comparing
two different cores; this finds it by degrading one, which is the cleaner demonstration.

---

## 5. In-domain data is not the most useful data

Two independent measurements now say the same thing:

| | result |
|---|---|
| `PV2_0155` | +5,000 CLEVR images raise CLEVR FG-ARI **+0.0038**; +5,000 **Tetrominoes** images raise it **+0.0084** — 2.2× more, same budget |
| `PV2_0156` | continuing on **Tetrominoes** improves **Multi-dSprites** more (+0.0168) than continuing on Multi-dSprites does (+0.0058) |

Both are off-diagonal: the dataset that helps is not the dataset being improved. **`cont5000`
should not be described as domain adaptation**, and a benchmark that frames continued training
that way will mis-measure it.

### What the scaling curve says so far (six of nine points)

| images | fg_ari | fg_iou | obj_iou |
|---|---|---|---|
| 10,000 | 0.8421 | 0.7545 | 0.6410 |
| 12,000 | 0.8449 | 0.7607 | 0.6461 |
| **14,000** | **0.8469** | 0.7652 | 0.6524 |
| 15,000 | 0.8459 | **0.7702** | **0.6534** |

**The three metrics saturate at different rates.** FG-ARI peaks at 14,000 and turns down 0.0010;
Foreground IoU and matched-object IoU do not turn and are still rising. More in-domain data stops
improving *how* patches are grouped well before it stops improving *where the boundaries fall*.

**Not yet a convergence claim.** One 0.0010 decrease at one seed is the shape of a curve, not
saturation. 16,000–18,000 settle it. A benchmark reporting a scaling result must say which of the
three metrics it refers to, because they do not agree.

---

## 6. Seeds — unchanged in substance, and now a recorded decision

Core seeds 1 and 2 **will not be run** (decided 2026-11-09). Every number in this project is
seed 0 and every report says so. This is a scope choice, not an oversight, and it is the single
largest limitation on everything above.

What partially covers it: `PV2_0150` measured a baseline reseed at FG-ARI spread 0.1339 on CLEVR,
0.0535 on Tetrominoes, 0.0094 on Multi-dSprites, and our margins over the field (0.134–0.333)
exceed all of them. That is evidence about *baseline* variance. It is not an error bar on our own
core and must never be presented as one.

---

## 7. Order of work

1. **Finish the scaling curve** (16,000–18,000, ~21 h). It is the only running item and it decides
   whether §5's saturation reading holds.
2. **Decide whether `cont5000(Tetrominoes)` replaces 253 as the reported core.** It is measured,
   not adopted (`PV2_0156`). FG-ARI rises on all three datasets; the cost is CLEVR matched-object
   IoU −0.0203 against our own previous core, which is still +0.2354 over the best baseline. Both
   the FG-ARI-first priority and maximise-the-weakest-cell select it.
3. **A second AKOrN (conv) seed**, 7 minutes. It is the one baseline that is not budget-limited
   and it owns our narrowest margin (Tetrominoes Foreground IoU, +0.0345).
4. **Protocol B**, labelled per model. No baseline's authors released weights except Slot
   Attention, so every protocol-B row costs a full training run: AKOrN ~3.5 h on our 10,000 or
   ~24 h on the full 70,000; the CAE family ~18 h each.
5. Rotating Features and SynCx remain unrun. Both carry figures above ours on Multi-dSprites
   (0.84 and 0.82), and **both are oracle-assisted** — SynCx's evaluator reads the per-image true
   object count, and the 0.84 attributed to Rotating Features is SynCx's reproduction, not its
   authors' own number.

---

## 8. What the final report says, and in what order

1. **The headline**: the three metrics **and object-count accuracy**, per seed with mean and
   standard deviation, beside the best-baseline-per-cell composite, on the same image ids.
2. **Every baseline's own row**, with its published figure and its budget verdict.
3. **The diagnostics table**: predicted foreground against truth, own-metric agreement,
   background rule and the split that chose it, **and the object count**.
4. **The narrow cell, named.**
5. **The claim, scoped.** With seeds: "a CLEVR-trained core, frozen, beats the best of eight
   baselines in every metric cell on three datasets at an identical data budget, two of which it
   never trained on — while counting objects badly." Without seeds, the same sentence with
   "at seed 0" in it.
6. **The config, split manifest version, evaluation contract version, code commit, reproduction
   commands, checkpoint identifiers, known limitations** — and the evidence that the score came
   from classifier masks on spikes, not from theta or PLV.

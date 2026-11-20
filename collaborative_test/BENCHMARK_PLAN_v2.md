# Benchmark plan v2 — what this paper can actually claim

Supersedes `BENCHMARK_PLAN.md`, whose comparison family does not survive a literature
check. Written from measurements, with each table labelled by what it can and cannot claim.

---

## 0. The constraint to design around

On a head-to-head leaderboard we are not competitive with the slot family, and the gap is
not small:

| Tetrominoes / Multi-dSprites / CLEVR, pixel ARI-FG | |
| --- | --- |
| Slot Attention | 99.5 / 91.3 / 98.8 |
| IODINE | 99.2 / 76.7 / 98.8 |
| MONet | — / 90.4 / 96.2 |
| SynCx | 89 / 82 / 59 |
| CtCAE | 84 / 56 / 54 |
| **ours** | **92.8 / 30.6 / 66.7** |

All unsupervised. CLEVR is not comparable as listed — theirs is CLEVR6 at 128×128, ours is
full CLEVR, ten objects at 240×320 — but Tetrominoes is identical in conditions and we lose
by 6.7 points.

**A paper framed as "we beat the baselines" cannot be written from these numbers.** What
follows is framed around what the measurements do support.

---

## 1. The strongest result we hold is about the metric, not the model

We re-ran Slot Attention's published checkpoint under our conditions, scoring all three
metrics rather than the one the field reports:

| CLEVR, our conditions, patch protocol | FG-ARI | fg-IoU | obj-IoU | predicted foreground |
| --- | ---: | ---: | ---: | ---: |
| Slot Attention | **0.8102** | 0.1151 | 0.1025 | **0.8531** |
| ours | 0.7603 | **0.3373** | **0.6598** | 0.1497 |

**Slot Attention calls 85% of the image foreground, against a true 11.8%, and still scores
0.8102 FG-ARI.** FG-ARI is computed over true-foreground cells only, so it cannot see
over-painting. Our matched-object IoU is **6.4× higher**.

This is a measured claim about the field's standard metric, on the field's most-cited model.
Published tables report FG-ARI alone, so the behaviour is invisible in them.

**Table 1 — the metric critique.** Every model we can run, all three metrics, one protocol.
This is the table where our numbers look best **and** where the contribution is real
regardless of who wins.

Cost: we must run the comparison models ourselves. We have one point (Slot on CLEVR). CAE,
CtCAE, RF and SynCx have public code; each needs a run and a three-metric scoring pass.

---

## 2. What a straight comparison table can say

**Table 2 — unsupervised comparison, cited numbers.** Three datasets, pixel protocol,
FG-ARI, against published figures. We state the loss.

| dataset | split | seeds | grid | ceiling | ours |
| --- | --- | ---: | ---: | ---: | ---: |
| Tetrominoes | TEST 1000 | 3 | 32 | 0.9345 | **0.9282 ± 0.0170** |
| CLEVR | TEST 1000 | 3 | 32 | 0.8406 | 0.6668 ± 0.0506 |
| Multi-dSprites | TEST 300 | 1 | 64 | 1.0000 | 0.3061 |

Claimable: **above the synchrony-binding line on Tetrominoes** (SynCx 89, CtCAE 84).
Not claimable: anything against the slot family.

The restriction to the synchrony line has to be argued in the text, not assumed — these are
phase-based binding models and ours is one, so it is a fair family to be compared within,
but a reviewer will still ask about Slot Attention and the answer must be in the paper.

---

## 3. The ablations are our strongest material

This is the part worth building the paper around. **Nine candidate causes of our
Multi-dSprites failure were eliminated by measurement**, which is an unusually complete
account of why an architecture fails on a dataset.

**Table 3 — what moves the score and what does not.** One dataset, one metric family, one
variable at a time:

| variable | range tested | effect |
| --- | --- | --- |
| patch grid | 32 → 64 | ceiling 0.5619 → 1.0000; score 0.2080 → 0.2817; **fraction of ceiling fell** |
| readout threshold × min group | 12 combinations | **the default was the best of all 12**; the peer's own setting came last |
| merge stage (ported, verified) | on/off | groups 65 → **151**; score unchanged |
| sprite overlap | measured | fewer mixed cells than CLEVR (3.05% vs 6.26%) |
| object/background contrast | measured | **stronger** than CLEVR (13.0× vs 6.2×); per-image r = −0.039 |
| encoder capacity | 224 → 29,128, reconstruction | **0.2658 → 0.2193**, worse |
| encoder objective | reconstruction → masks | 0.2193 → 0.4252 |
| encoder objective | reconstruction → **recon + total variation** | 0.2193 → **0.4617, unsupervised** |

Two of these are results in their own right:

- **capacity is not the lever, the objective is** — the identical architecture scores worse
  with more parameters when trained by reconstruction
- **a total-variation prior beats mask supervision** — 0.4617 unsupervised against 0.4252
  with our masks and 0.4574 with the peer's

**Table 4 — the grid as a controlled variable.** Ceiling and score together, per grid, per
dataset. This makes a methodological point the field does not currently make: **a
patch-based method's number is bounded by a quantisation ceiling that nobody reports.** Our
Tetrominoes 0.9282 is 99.3% of what grid 32 can express, and the grid-35 run (in progress)
removes the bound entirely.

**Table 5 — supervision as a controlled variable.** Same architecture, same core, same
readout, four encoders: reconstruction, reconstruction+TV, our masks, the peer's masks.
Bounds what supervision is worth in this architecture, which is a question the field asks
and rarely answers.

---

## 4. The architectural analysis

`PROPOSAL_0.9_v3` derives from the equations that **this family of model cannot drive two
oscillators apart**: coupling is non-negative after the ReLU, the phase lag is bounded by a
hardcoded `alpha_scale = 1.0` so |α| < 57.3° where repulsion needs 90°, and a pair's stable
state is Δ = α. Anti-phase is not reachable.

That explains why nine routes that changed *when* the model merges all failed, and it is a
statement about Kuramoto-based binding generally, not only our implementation. It is
currently **unvalidated** — the ablation is designed (`alpha_scale` past π/2, lag driven by
geodesic/Euclidean ratio) and not run.

Worth including only if that ablation runs. An equation-level claim with no experiment is a
remark, not a result.

---

## 5. What to run, in order of value per hour

| # | run | cost | what it buys |
| --- | --- | --- | --- |
| 1 | finish grid 35 Tetrominoes | in progress | Table 4's point, and possibly a better headline |
| 2 | TV on TEST, weight fixed on validation, 3 seeds | ~1 day | promotes 0.4617 from exploratory to reportable |
| 3 | TV on Tetrominoes and CLEVR | hours | whether the prior generalises or is Multi-dSprites-specific |
| 4 | Multi-dSprites seeds 1–2 | ~25 h | the only single-seed row in Table 2 |
| 5 | CAE / CtCAE / RF / SynCx run by us, three metrics | days | **Table 1**, the strongest contribution |
| 6 | the `alpha_scale` ablation | ~1 day | makes §4 publishable |

**1–3 before 4.** A better model changes every row; a third seed changes one cell's error
bar.

**5 is the expensive one and the most valuable.** Without it Table 1 has a single data
point and the metric critique rests on Slot Attention alone.

---

## 6. What this paper is

Not "a new state of the art". The honest shape, from what is measured:

> A spiking Kuramoto architecture for object binding, evaluated under a three-metric
> protocol that exposes a failure mode of the field's standard metric; with a systematic
> account of what limits it, including a quantisation ceiling that patch-based methods do
> not report, and an unsupervised boundary prior that matches mask supervision.

Each clause is backed by measurement already in hand or one run away. The leaderboard claim
is the only thing being given up, and it was not available.

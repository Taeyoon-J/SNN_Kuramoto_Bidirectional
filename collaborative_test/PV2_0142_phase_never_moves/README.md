# PV2_0142 — on Tetrominoes and Multi-dSprites, the 253 recipe never moves `phase_primary`

**Status** completed (analysis of eight runs' own training logs)
**Found while** resuming `PV2_0138` after the Nov 6 reboot
**Bearing on** what the `PV2_0138` per-dataset cores can be expected to produce

## The observation

253's core objective has two terms. On CLEVR, both fall hard. On these two datasets, one of
them does not move at all — in **eight independent runs**.

Measured first-50 against last-50 updates from each run's own `training.jsonl`:

| run | updates | `phase_primary` | `positive_spike` | grad-norm median |
|---|---|---|---|---|
| **253 on CLEVR** (finished) | 6,250 | 4.636 → 2.456 **−47.0%** | 15.388 → 2.584 **−83.2%** | **3.57** |
| Tetrominoes seed 0 | 2,190 | 2.887 → 2.906 +0.7% | 8.673 → 4.928 −43.2% | 4.2e5 |
| Tetrominoes seed 1 | 2,190 | 2.875 → 2.809 −2.3% | 5.783 → 4.433 −23.3% | 1.6e4 |
| Tetrominoes seed 2 | 2,190 | 2.632 → 2.600 −1.2% | 9.916 → 7.026 −29.1% | 1.9e5 |
| Multi-dSprites seed 0 | 2,190 | 3.562 → 3.648 +2.4% | 10.499 → 10.224 −2.6% | 7.7e7 |
| Multi-dSprites seed 1 | 2,190 | 5.506 → 5.116 −7.1% | 5.755 → 5.089 −11.6% | 1.1e8 |
| Multi-dSprites seed 2 | 2,190 | 3.024 → 3.077 +1.8% | 10.800 → 10.286 −4.8% | 2.3e7 |
| Tetrominoes, 10,000×10 (66%) | 4,311 | 2.623 → 2.662 +1.5% | 15.370 → 10.256 −33.3% | 1.1e6 |
| Multi-dSprites, 10,000×10 (66%) | 4,130 | 2.621 → 2.532 −3.4% | 13.709 → 13.701 **−0.1%** | 3.4e5 |

`phase_primary` moves by at most 7% and often the wrong way. On Multi-dSprites `positive_spike`
barely moves either. Pre-clip gradient norms run four to eight orders of magnitude above
CLEVR's, which is the cancellation signature documented for 252 in `PV2_0136`.

## Judging it at 66% is fair, and that was checked

The obvious objection is that the running cores are only two thirds done and 253's gains might
come late. They do not. Measured at matched fractions of the same 6,250-update schedule:

| fraction | 253 on CLEVR | Tetrominoes (new) | Multi-dSprites (new) |
|---|---|---|---|
| 25% | phase −28.4%, spike −42.7% | +2.3%, −0.2% | −1.5%, −2.1% |
| 50% | −45.8%, −75.2% | −0.4%, −24.7% | −5.7%, −0.8% |
| 66% | **−46.1%, −81.4%** | +1.5%, −33.3% | −3.4%, −0.1% |
| 100% | −47.0%, −83.2% | — | — |

253-on-CLEVR had essentially finished learning by 66%; it gained only 0.9 and 1.8 further
percentage points in the last third. So the comparison stands.

## Two readings, and the scores will decide between them

**(a) The recipe fails on these datasets.** The gradient norms and the flat objectives look
exactly like 252: lots of gradient, no motion, which is terms cancelling. If this is right, the
`PV2_0138` cores will be weak and the per-dataset column will come out *below* the
CLEVR-transfer column.

**(b) There is nothing left to learn on that term.** Note where `phase_primary` *starts*:
2.6–3.6 on these datasets, against 4.64 on CLEVR — close to where CLEVR *ends* (2.46). These
are simpler scenes, so the random-init phase structure may already be adequate and the term is
near its floor from the first update. Under this reading the cores are fine and the flat curve
is uninformative.

The two readings predict opposite things, and `PV2_0138`'s scores will separate them:

- per-dataset core scores **well** → reading (b), the term was already satisfied;
- per-dataset core scores **poorly** → reading (a), the recipe does not transfer.

I do not currently know which it is, and the table above does not settle it.

## Why it matters either way

Our generalisation claim is that a **CLEVR-trained** core carries to datasets it never saw
(Tetrominoes 0.8580 FG-ARI, Multi-dSprites 0.6974, both seed 0). If reading (a) holds, then
training *on* these datasets does not work, and the CLEVR-transfer route is not merely
convenient — it is the only route that produces a core at all. That would strengthen the claim
while narrowing what the method can be said to do.

It also offers a plain explanation for why the peer's 3,500×10 per-dataset numbers were low.

## Reproduce

```bash
python3 old3500.py     # the six finished 3,500x10 runs
python3 stage.py       # matched-fraction comparison against 253-on-CLEVR
python3 trend261.py    # current state of the two running cores
```

All three read the runs' own `training.jsonl` on frontier; nothing is recomputed from
checkpoints.

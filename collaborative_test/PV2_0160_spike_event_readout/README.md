# PV2_0160 — can the mask be read from the threshold crossings? No, not from this checkpoint

**Status** completed · **negative result**
**Parent** `PV2_0159` · **Comparison** the same rollouts, affinity source swapped
**Data** clevr_with_masks, split manifest v1, **first 20 reserved images** · **Contract** v2
**Base commit** `41b0fa9` · **Seed** 0 · frontier GPU 2 · no retraining

## Why this was run

`PV2_0159` established that the reported readout is numerically a phase readout: the identical
readout on `component_gates` reproduces every mask to 16 significant figures. `CLAUDE.md`
requires the final mask to come from a classifier on **spikes or membrane** and forbids a phase
readout as the final mask. That was documented and not resolved. This is the attempt to resolve
it: build the affinity from the actual threshold crossings `e` instead of the gated trace
`s = g·e`, with nothing retrained.

`e` is 1 in 99.992% of settled steps, so it carries almost no variance as it stands. The sweep
raises the firing threshold **at inference only**, read from the membrane trace the normal
rollout already produces. The foreground set is taken from the control arm and held fixed, so
the only thing that changes is the signal the grouping is computed from —
`spike_synchrony_components` takes foreground activity and affinity components as separate
arguments, which makes that isolation exact.

## 1. The membrane sits far above its own threshold

The trained threshold is `vth = 0.06` (read from `wrapped.core.membrane_layer.vth`). Raising it:

| threshold | ×1 | ×1.5 | ×2 | ×3 | ×5 |
|---|---|---|---|---|---|
| firing rate | .99988 | .99942 | .99799 | .97888 | **.866** |

**Five times the trained threshold still leaves 87% of steps firing.** The 99.992% figure in
`PV2_0159` is not a near-miss, it is where training put the operating point.

## 2. Sweeping the firing rate directly

Threshold set per component as the quantile of `V` over the settled window that gives the target
rate. 20 reserved CLEVR images.

| arm | firing rate | fg_ari | fg_iou | obj_iou | regions |
|---|---|---|---|---|---|
| events `e` | .866 | .4073 | .7349 | .3758 | 176.0 |
| events `e` | .500 | .4960 | .6165 | .4058 | 115.8 |
| **events `e`** | **.300** | **.5622** | .6150 | **.4473** | 85.2 |
| events `e` | .200 | .5290 | .5975 | .4243 | 93.0 |
| events `e` | .100 | .3968 | .5780 | .3285 | 132.9 |
| events `e` | .050 | .2096 | .4960 | .1958 | 183.6 |
| **control, gated trace `s`** | — | **.8798** | **.8145** | **.7264** | **7.45** |

There is a real interior optimum near a 30% firing rate, and it is **not enough**: the best
spike-event arm recovers **64% of the control's FG-ARI** and **62% of its matched-object IoU**,
and emits **11× the regions** (85.2 against 7.45).

## 3. Verdict

**The direction is closed for this checkpoint.** The grouping information is in the graded
amplitude of the gate, not in the binary firing events, and no inference-time threshold recovers
it. `CLAUDE.md`'s requirement cannot be satisfied by re-reading the existing core; it needs a
differently trained one.

The cause is structural and was predictable from the objective. The spike loss is defined on
`s = g·e` (`loss.py:21`, `primary + 5·spike`), and `s` is dominated by `g`. **Nothing in the
objective ever put pressure on `e`** — no rate term, no sparsity term, no loss defined on the
crossings — so the membrane was free to saturate above threshold, and it did. The model learned
to use the gate and to leave the spike mechanism switched on.

## 4. One thing worth keeping from the negative result

At a 30% firing rate, using **only binary threshold crossings and no gate amplitude at all**,
the readout still reaches fg_ari .5622 / fg_iou .6150 / obj_iou .4473. Those are well above
Slot Attention's CLEVR row (.2872 / .2220 / .0801) and above the best-baseline composite on two
of the three cells (.6326 / .4267 / .3853 — ahead on fg_iou and obj_iou, behind on fg_ari).

**This is not a table claim.** It is 20 images against the baselines' 320, and the control on
these 20 scores .8798 where the 320-image control scores .8421, so this subset is easier. What
is established is the *internal* comparison in §2. A 320-image run would be needed before any
of it goes beside Table 1.

## Commands

```bash
$PY spike_readout.py --code benchmark_codes --config configs/frontier_staged.json \
  --run .../recovered/run --dataset clevr --split fixed320 --count 20 \
  --mults 1.0,1.5,2.0,3.0,5.0        # multiples of the trained threshold
$PY spike_readout.py ... --mults 0.5,0.3,0.2,0.1,0.05   # values <= 1 are target firing rates
```
Script in `evaluation_howto/spike_readout.py`; results `metrics_threshold.json`, `metrics_rate.json`.

## Limitations

20 images, one seed, CLEVR only. The quantile threshold is global per component over the whole
settled window; a per-patch or adaptive threshold was not tried and could do better. The
foreground set was inherited from the control, so this measures the grouping signal only and
says nothing about whether `e` could also carry foreground detection.

## Next proposal

The experiment that would make "spike" load-bearing is now a **training** change, not a readout
change: define the spike term on `e` rather than on `s`, or add a firing-rate target that keeps
the membrane near threshold. That is a core change under `CLAUDE.md`'s rules — analyse the
equations first, validate with a small ablation, one variable. Until it is run, the paper should
keep saying what Section 3.3 and Appendix F3 now say: the timing is the gate's.

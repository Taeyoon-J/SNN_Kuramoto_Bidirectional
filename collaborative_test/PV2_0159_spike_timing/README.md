# PV2_0159 — the grouping is carried by relative timing; and the spikes are the gate

**Status** completed
**Parent** `PV2_0137` (253 core) · **Comparison** the same rollouts, one factor changed at a time
**Data** clevr_with_masks, split manifest v1, first 40 reserved images · **Contract** v2
**Base commit** `4b111cf` · **Seed** 0 · frontier GPU 2

## Hypothesis under test

The paper's own: *when oscillators representing patches of the same object synchronise, their
downstream neurons produce similar spike patterns*, and that similarity is what produces the
masks. Two measurements, plus one control that was not asked for and changes the reading.

## Validity of the ablations

`Q` is recomputed here from `component_spikes` rather than taken from the model, so a modified
trace can be pushed through the same arithmetic. On the unshifted control the recomputed `Q`
agrees with the model's own `q` to **6.7 × 10⁻⁶** maximum absolute difference over all 4096²
entries. The ablations therefore differ from the control in exactly one thing.

## 1. How different are different objects' spikes

Ground truth is used **only to sort patch pairs into groups** — never to set a threshold, a
count, or a cluster. Mean over 40 images:

| patch pair | affinity `Q` |
|---|---|
| inside one object | **0.891 ± 0.111** |
| different objects | **0.047 ± 0.048** |
| object vs background | 0.099 ± 0.106 |

Nineteenfold separation, with the across-object value near the floor of the measure.

## 2. Per-patch circular shift — the direct test

Each patch's four component traces are rolled by one offset drawn uniformly over the readout
window, independently per patch. **Rate, waveform and within-patch structure survive; only the
alignment between patches is destroyed.** Graph, foreground threshold, component floor,
background heuristic and greedy merge untouched.

| | fg_ari | fg_iou | obj_iou | regions |
|---|---|---|---|---|
| control | .8648 | .8028 | .6925 | 7.8 |
| **per-patch shift** | **.0735** | .7358 | **.0325** | **562.3** |
| gates instead of spikes | .8648 | .8028 | .6925 | 7.8 |

FG-ARI falls by 0.79 and the region count rises 72×, while **Foreground IoU falls only 0.067**
— the foreground test is a mean over the window and a rotation does not change a mean, so the
*detection* survives and the *partition* shatters. This is the cleanest demonstration in the
project that the grouping information is in relative timing, not in which patches are active.

It also happens to be a worked example of the metric asymmetry recorded in the paper's
Appendix E: FG-ARI and matched-object IoU see the partition, Foreground IoU sees the set.

## 3. The control that was not asked for: the spike trace is the gate

Third row above replaces `component_spikes` with `component_gates` and changes nothing else.
**Every mask is reproduced exactly** — 40 images, all three metrics identical to 16 significant
figures (`0.8647904747997799` both).

The reason is in the traces. With `s_ia = g_i · e_ia`:

| quantity | value |
|---|---|
| fraction of settled steps with `e = 1` | **0.99992** |
| `corr(component_spikes, component_gates)` | **0.9997** |
| `max |spikes − component_spikes.mean(0)|` | 0.000000 (patch `spikes` is just the 4-component mean) |

The membrane is above threshold in essentially every settled step, so the spiking layer passes
the phase gate through rather than selecting anything, and the 0.008% of silent steps never
change a thresholded decision.

**Consequence.** The timing shown necessary in §2 is the timing of the *oscillatory gate*.
`CLAUDE.md` requires the final mask to come from a classifier on spikes or membrane and forbids
a phase readout as the final mask. The masks *are* computed from `s`, which is what is cached
and scored — but `s` is numerically `g`, and `g = (1 + sin θ̄(t−2))/2` is a deterministic
function of the Kuramoto phase. **The readout is a phase readout wearing a spike's clothes, and
the paper now says so** (Section 3.3 and Appendix F3).

This does not touch the benchmark numbers: those masks are real and were scored by the same
function as every baseline. What it touches is the mechanism claim and the word "spike" in the
title.

A further implication, measured earlier and consistent: `g_i` is shared by all four components,
so with `e ≈ 1` the four `ρ_a` are near-identical and `Q = Π_a max(ρ_a,0) ≈ ρ⁴` — the
four-component structure contributes little.

## Commands

```bash
# frontier, GPU 2
$PY spike_timing.py --code benchmark_codes --config configs/frontier_staged.json \
  --run /work/USERS/tkim1/sw0255b/recovered/run \
  --dataset clevr --split fixed320 --count 40 --out spike_timing_clevr.json
```
Result: `artifacts.json`. Script in `evaluation_howto/spike_timing.py`.

## Limitations

40 images, one seed. The shift ablation destroys cross-patch alignment but also any slow
common drift; a phase-randomisation control that preserves the power spectrum would separate
those, and was not run. The gate control used `component_gates` directly; it does not test what
a readout designed to use threshold crossings could do.

## Next proposal

A readout that uses the firing events `e` rather than the gated trace `s` — for example
correlating inter-spike intervals, or lowering `ϑ_a` until `e` is sparse — is the experiment
that would make the title's "spike" load-bearing. It is a readout change, not a core change,
and can be measured on the existing caches.

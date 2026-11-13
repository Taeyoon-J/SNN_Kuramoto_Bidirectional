# PV2_0161 — core change analysed, not made: give the firing events something to do

**Status** planned (analysis only; no code changed, no run started)
**Parent** `PV2_0160` · **Base commit** `4dae3f6` · **Contract** v2 · split manifest v1

`CLAUDE.md` requires a core change to be analysed before it is made — what in the equations or
the state updates changes — and then validated with a small ablation. This is that analysis.
Nothing has been changed.

## Why a core change is now the only route

`PV2_0159`: the readout is numerically a phase readout — running it on `component_gates`
reproduces every mask to 16 significant figures, because `e` is 1 in 99.992% of settled steps.
`PV2_0160`: the inference-time fix fails — the best spike-event arm reaches FG-ARI .5622 against
the control's .8798 on the same images, and five times the trained threshold still leaves 87% of
steps firing. The saturation is where training put the operating point, so it has to be changed
at training time or not at all.

## What is actually wrong, found in the loss

`full_benchmark/loss.py:14-21`:

```python
criterion = UnsupervisedS2NetLoss(spike_rate_weight=0., spike_smooth_weight=0.,
    spike_diversity_weight=0., structural_weight=0., plv_bimodality_weight=6., ...)
phase = phase_locking_value(output['trace']['theta'], settle=512, combine='mean')
primary, _ = criterion(plv=phase)
spike,   _ = criterion(plv=output['q'])
return primary + 5.*spike, ...
```

Two separate facts, and both must change.

**1. The terms that would constrain firing exist and are switched off.**
`snn_kuramoto_bidirectional/loss_function.py` defines `spike_rate_loss(spikes,
target_rate=0.1)` at line 8, whose body is `loss = (rate - target_rate)**2` (line 20) — exactly
the firing-rate target whose absence `PV2_0160` identified. The class defaults are
`spike_rate_weight=1.0` and `spike_target_rate=0.1` (lines 795, 806), alongside
`spike_smooth_weight` and `spike_diversity_weight` at 0.1. **`core_objective` overrides all
three to zero.**

**2. Even un-zeroed, the term could never fire.** Line 929 reads
`add("spike_rate", spikes is not None, ...)`. `core_objective` calls `criterion(plv=...)` and
never passes `spikes`, so the term is skipped regardless of its weight. Turning the weight on
alone would change nothing — a trap worth recording.

And what the objective calls the "spike" term is `criterion(plv=output['q'])`: the same four
PLV-shaped terms (bimodality, balance, coherence, collapse) applied to `Q`. `Q` is built from
`s = g·e`, `s ≈ g`, so **the spike term is a second phase term**. That is the whole explanation
of `PV2_0159` in one line: the objective contains two phase losses and nothing about firing.

## What changes in the equations

Add to the objective

```
L_rate = w · ( mean_{i,a,t∈settled} s_ia(t) − r* )²
```

with `s_ia(t) = g_i(t)·e_ia(t)` as already defined. No state update changes; no new state.

**Gradient path.** `membrane_layer.py:77-78` computes `inputs_ = mem − v_th` and
`spike = act_fun_adp(inputs_) * g_wave_t`, where `act_fun_adp` is a Heaviside with a
multi-Gaussian surrogate derivative. So ∂L_rate/∂s → ∂s/∂e via the surrogate → ∂e/∂V → the
membrane update (λ_i, ξ_a, ϑ_a through q_a) → H → the dendritic branch (U, b, β, η) → the gate.
It reaches every parameter the spiking path owns, which is precisely the set that the present
objective leaves unconstrained.

**Why it should move the operating point.** `g ∈ [0,1]` with mean ≈ 0.5 fixed by the sinusoid,
so `mean(s) = mean(g·e)` can only be driven to `r* = 0.1` by making `e` zero most of the time.
That pulls `V` down toward `ϑ`, which is the saturation `PV2_0160` measured.

**The risk, stated before the run.** Lowering the firing rate also lowers the amplitude of `s`,
and the existing `Q(s)` structure — the thing that currently scores .8421 — is built from that
amplitude. The change could buy an informative `e` at the cost of the result. That is the
trade the ablation has to measure, and it is why `w` and `r*` are the one variable group.

## Cost, measured

From `scale_curve2.log`: 630 updates take 5 h 42 m to 6 h 03 m, i.e. **≈33 s per update** on one
frontier GPU. A full retrain at 6,250 updates is **≈57 hours**. A warm-start continuation of
600 updates with the term on is **≈5.5 hours**.

## Proposed ablation, if it is authorised

One variable group: `(spike_rate_weight, spike_target_rate)`. Warm-start from the 253 checkpoint,
600 updates, arms `w ∈ {0 (control), 1.0}` at `r* = 0.1`. Success criterion stated in advance:

* **Mechanism** — settled firing rate falls below 0.9 and the `Q(e)` affinity becomes
  non-degenerate (same-object vs different-object separation above zero). This is what the
  change is *for*.
* **Cost** — the three contract metrics on CLEVR validation, read with the normal `s` readout,
  against the control arm.

If the mechanism moves and the cost is small, it justifies a from-scratch run. If the firing
rate does not move in 600 updates, a warm start is the wrong instrument and the question needs
the full 57 hours or a larger `w`.

## Not started

This is an analysis. No file has been edited, no job submitted. The decision to spend 5.5 GPU
hours on a core change is the user's, and the paper's current wording (Section 3.3, Appendix F3)
is correct for the model as it stands either way.

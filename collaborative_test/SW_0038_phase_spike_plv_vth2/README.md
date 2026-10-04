# SW_0038: phase objective plus component-spike PLV at vth2

Status: prepared, not started. Run only if an idle GPU and time remain after
SW_0037's first validation.

Peer `patch_v2:c76492d` reports a large gain from adding the PLV family to
product-combined per-component spike synchrony while retaining the phase loss.
Its render, split, graph and reference differ from ours, so none of its scores
are pooled here. We port only this loss structure.

Matched seed-0 comparison:

- `control`: phase PLV, vth2, spike auxiliary weight 0.
- `spike0p25`: identical, with component-spike PLV weight .25.
- `spike5`: identical, with component-spike PLV weight 5.

Both use our learned graph top-k32/spatial decay .55; no peer geodesic graph,
classifier thresholds, data split or foreground target. The existing four PLV
terms and phase path remain active. The auxiliary product synchrony puts actual
component spikes on the gradient path. Evaluate under the SW_0034–0036 fixed
validation suite; loss value alone cannot select a model because the peer found
lower final loss associated with worse task metrics across three seeds.

`run.sh GPU_ID VARIANT SEED` trains any variant. After a checkpoint is
complete, `evaluate.sh GPU_ID VARIANT SEED` applies the same full320 stage
diagnostic and selected classifier suite as SW_0037.

The `.25` dose was added only after the first `spike5` epochs showed an
unweighted spike loss near 13 versus phase loss near 3.2: weight 5 contributes
about 65 and dominates the objective, while weight .25 contributes about 3.25.
This matched parallel ablation separates the transferred mechanism from its
loss scale; no metric was inspected to choose the dose.

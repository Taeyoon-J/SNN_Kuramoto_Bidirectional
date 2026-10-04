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

## Seed-0 full320 result

All three 40-epoch checkpoints completed. Fixed spatial k10 stage readouts:

| variant / signal | FG-ARI | foreground IoU | object IoU | binary event rate | constant fraction |
|---|---:|---:|---:|---:|---:|
| control membrane | .415576 | .226701 | .195412 | .061485 | .164218 |
| control spike | .407526 | .228462 | .194464 | .061485 | .164218 |
| spike .25 membrane | .405577 | .228304 | .194326 | .227556 | .043170 |
| spike .25 spike | .416059 | .228732 | .190776 | .227556 | .043170 |
| spike 5 membrane | .437617 | .225843 | .199737 | .367379 | .000388 |
| spike 5 spike | .435727 | .226240 | .198260 | .367379 | .000388 |

The additive spike objective produces a dose-dependent recovery over the
phase-only vth2 control in FG-ARI and, at weight 5, object IoU. It also restores
more nonconstant event activity. Yet every variant remains below the existing
membrane-spatial ARI lead `.494639`, and no row dominates the prior Pareto
candidates on all three metrics. Do not expand this exact configuration to
seeds 1/2.

The spike5 launcher log ends with a shell `get-density` error only after epoch
40 and checkpoint saving. The already-running shell script was overwritten in
place to add the `.25` variant, changing bytes that bash read after Python
returned. The Python command, checkpoint, and full validation completed; the
current committed launcher is syntax-checked. Future live launchers must be
copied to a new path rather than overwritten.

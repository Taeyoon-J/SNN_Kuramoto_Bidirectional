# SW_0038: phase objective plus component-spike PLV at vth2

Status: prepared, not started. Run only if an idle GPU and time remain after
SW_0037's first validation.

Peer `patch_v2:c76492d` reports a large gain from adding the PLV family to
product-combined per-component spike synchrony while retaining the phase loss.
Its render, split, graph and reference differ from ours, so none of its scores
are pooled here. We port only this loss structure.

Matched seed-0 comparison:

- `control`: phase PLV, vth2, spike auxiliary weight 0.
- `spike5`: identical, with component-spike PLV weight 5.

Both use our learned graph top-k32/spatial decay .55; no peer geodesic graph,
classifier thresholds, data split or foreground target. The existing four PLV
terms and phase path remain active. The auxiliary product synchrony puts actual
component spikes on the gradient path. Evaluate under the SW_0034–0036 fixed
validation suite; loss value alone cannot select a model because the peer found
lower final loss associated with worse task metrics across three seeds.

# Proposed next test: detach event credit only in the spike auxiliary

Status: design only. No runner, preflight, training, or queue is authorized by this document. The current work window ends at 2026-10-07 10:37:49 UTC. Resume requires the user's later instruction; nothing should launch automatically after the pause.

## Question and evidence

Test whether surrogate gradient credit through the binary event factor harms FG-ARI when the actual forward events are nearly saturated. SW0104's 16-image validation diagnostic found nearly identical actual-spike and forced-always-on readout metrics, while binary-event-only predictions produced zero groups. This supports gate-amplitude dominance on that subset, not a global causal claim or a replacement classifier.

Event-factor gradients nevertheless remained nonzero. Their Kuramoto directions opposed gate-factor gradients in all four inspected seed2 training batches, with cosines from approximately -.47 to -.98. This motivates one controlled gradient intervention. It does not prove that removing those gradients improves segmentation. The strict gradient-sum check retained four seed2 family/batch failures; its largest relative residual was 1.835e-5, and the largest absolute residual among failed checks was 6.63e-5. Do not describe the numerical identity check as a complete pass.

The recorded event occupancy and constant-unit statistics span the full 64 training frames or full 1024 validation frames. Affinities and losses use their registered settle32 or settle512 windows.

## Single intervention

Keep the entire actual core forward and inference readout unchanged:

```python
actual_component_spikes = event * gate
```

Only the spike auxiliary uses:

```python
aux_component_spikes = event.detach() * gate
aux_affinity = spike_synchrony_affinity(
    aux_component_spikes.mean(dim=1),
    components=aux_component_spikes,
    settle=32,
)
loss = existing_phase_primary + 5.0 * existing_criterion(plv=aux_affinity)[0]
```

The coefficient remains the old 5.0. There is no new coefficient calibration or sweep. The phase primary, positive-product criterion, density/bimodality/coherence/collapse weights, raw gate mapping, membrane threshold, and classifier are retained.

Detach only the captured event tensor at this auxiliary output multiplication. Do not replace or globally detach `act_fun_adp`, detach the internal reset/recurrent spike state, alter previous-spike inputs to the dendrite, or substitute forced-always-on activity. The original event, gate, dendrite, membrane, and recurrent ancestors remain intact. This tests explicit output-factor credit; it does not remove every event or gate path from the network.

Capture the actual live event and actual passed membrane gate, with component history shape `[B,D,N,T]`, rather than inferring events by division through a possibly zero gate. An optional tracing/helper implementation must preserve numerical order, initialization, RNG behavior, and checkpoint keys/shapes. Default auxiliary behavior stays legacy full credit. No extra trainable parameter is introduced.

## Matched recipe

Start each candidate from its whole SW0095 final core, source seeds0/1/2. Reuse the completed SW0097 `positive_frozen` controls after checking their manifests.

- 256 Adam updates, batch16, the exact 4,096 ordered training IDs per matched control.
- Shuffle117/118/119; actual training64 steps, settle32.
- Core LR3e-5, gradient clip1, unchanged registered priors and positive-product objective.
- Encoder and graph frozen. No masks, counts, teacher pair labels, new reconstruction, anchors, or connectivity losses.
- The candidate has the same source SHA, data order, exposures, update count, optimizer, and evaluation contract as its corresponding control.

The intervention removes the spike auxiliary's direct gradient into dendritic and membrane parameters; the phase primary does not reach those parameters either under this recipe. They should therefore remain bitwise at the source state. This is an expected consequence of the selected gradient path, not a second architecture change. Kuramoto/input parameters remain eligible to update through the unchanged phase primary and gate-factor auxiliary. The actual neuronal forward still runs at training and inference.

## Required preflight after a future resume

For every source seed, on actual registered training inputs:

1. Strict checkpoint loading passes with identical state-dict names/shapes. Metadata capture adds no checkpoint tensor or parameter. Graph feedback and pulse coupling remain disabled.
2. Actual `event*gate` exactly reconstructs `last_component_spikes`; folded batch/component ordering and delay are verified.
3. At the same checkpoint and input, full-credit and event-detached auxiliary histories, classifier affinities, and scalar loss values match exactly. Legacy tracing-off/on actual core outputs match exactly. The intervention changes backward credit only.
4. The candidate helper's gradient matches the direct reference expression `event.detach()*gate`; it is the SW0104 gate-credit construction, not its event-credit construction. Keep any predefined numerical check failures explicit instead of relaxing tolerances after execution.
5. Losses and trainable-parameter gradients are finite and nonzero where expected. A real update changes eligible upstream core tensors while encoder, graph, dendrite, and membrane tensors remain bitwise fixed. Record gradient norms by family; do not mistake nonzero surrogate gradients for demonstrated binary timing information.
6. A full-batch GPU memory/update check passes on a genuinely free authorized device. No control or completed candidate is silently repeated.

## Evaluation and decision

Evaluate all three candidates on validation1320-1639, original batch8, actual1024 inference steps/settle512, threshold.50, minimum group2, largest connected component background, and the established patch metrics. Predictions use the original actual `event*gate` component spikes. There is no per-image readout choice or gate-only final mask.

Record all seeds and paired per-image differences against SW0097. To justify one further full70k continuation on the FG-ARI track, require:

- Three-seed FG-ARI mean at least .01 above the matched control mean, with at least two positive seed deltas.
- Foreground IoU at least .25358913 and matched-object IoU at least .25693698, preserving .05 margins over the matched Slot means.
- A paired image-bootstrap FG-ARI difference interval supporting positive improvement. Resample the same image indices across all three seeds; this does not establish seed-level generalization.

Inspect event occupancy/variance, actual-versus-forced-on affinity and endpoint differences, and object connectivity/fragmentation alongside the metrics. A performance gain dominated by gate amplitude may advance the user's FG-ARI objective but cannot be presented as newly demonstrated binary-event binding. Useful tradeoffs remain recorded under `SELECTION_POLICY`; a failed material FG-ARI gate receives no automatic coefficient or seed-specific follow-up.

Do not read reserved holdout90640-90959 to select this setting. Full70k expansion and any eventual independent comparison require their own frozen recipe and evidence audit. The ultimate goal remains the three-seed actual spike/membrane mask comparison above matched Slot on all three metrics; this short continuation alone does not finish it.

## Why this choice precedes event-credit-only training

The other direction, `event * gate.detach()` in the auxiliary, would redirect explicit output credit toward the event factor while retaining the same numerical forward. It does not itself solve the measured positive activation margins and saturation, and restoring crossings by inference-only threshold changes already failed in SW0083. It therefore remains unselected here. The single proposed test first isolates the surrogate-credit hypothesis without a new rate target, threshold calibration, gate variant, or neuronal dynamics change.

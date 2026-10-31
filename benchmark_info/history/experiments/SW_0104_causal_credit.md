# SW_0104_causal_credit

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## next_test_design.md

<details>
<summary>next_test_design.md</summary>

Source SHA256: `e5220dbff2203c690f871da855fbff9af106c40a7c8ee85859eb92564e7a57a0`

````markdown
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

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `317303e0942b6b45782dd37649f73c37419a699c389d3641fb095498972e22b8`

````markdown
# SW0104 explicit event/gate causal-credit audit

This is a read-only diagnostic on the three matched SW0097 frozen cores. The membrane module's actual `act_fun_adp` callable was wrapped temporarily to capture the live logit `x` and binary event `e`, while hooks captured the actual per-step gate `g`, dendrite input, membrane output and Kuramoto phase. The wrapper returned the original `e` unchanged and all hooks/callables were restored in `finally`. Graph feedback and pulse coupling were confirmed disabled. `e*g` exactly reconstructed each historical component spike tensor.

On the first four registered B16/T64/settle32 training batches per seed, without GT or optimizer, the existing five-times positive-product spike loss was evaluated three ways at the same forward values: full `e*g`, gate-credit `e.detach()*g`, and event-credit `e*g.detach()`. The three loss values matched exactly in all 12 batches. The parameter-gradient sum check used the fixed 1e-5 relative and max-absolute tolerances. It passed all family/batch checks for seeds0 and1; seed2 failed 4 checks (Kuramoto family in batches0,2,3, and the `other` family in batch0). The largest relative residual was `1.835e-5`. The largest Kuramoto absolute residual was `3.82e-5` in a different batch; the largest absolute residual among the failed parameter-family checks was `6.63e-5` for the `other` family in seed2 batch0. Tolerances were not relaxed. The remaining gradient norms, cosines, and captured theta/h/membrane/x/g node gradients are in `results_archive/train_credit.json`.

Training-side event statistics describe the full 64-frame trace; the spike-loss computation itself uses settle32. Event occupancy was above `.998` in every batch, and the fraction of per-unit event traces constant over all 64 frames was high across seeds. At validation, event occupancy across all 1024 frames was `.99885/.99998/.99721` and the per-unit always-on fraction was `.86163/.99756/.70764`. These are full-window saturation statistics, not settle512-only event estimates.

The fixed IDs1320–1335 validation comparison built all actual, gate-only and binary-event predictions for all seeds before reading labels. The actual `e*g` readout reproduced every archived SW0097 per-image metric exactly. Mean FG-ARI/foreground IoU/object IoU was `.810841/.561339/.642008` for actual spikes, `.810753/.561461/.641527` for gate-only `g` activity, and `0/0/0` for binary-event-only readout, which produced zero groups. Actual-vs-gate q≥.50 edge agreement was `.99853/1.00000/.98864` by seed; actual-vs-event agreement was `.39182/.56799/.74697`. The readout-only alternatives did not change the original forward or checkpoint.

The gate-only classifier is mathematically the positive scalar gate Pearson correlation raised to the fourth power. Its separately calculated scalar-formula check differed from the component classifier by `5.54e-6` to `6.62e-6` in float32, above the recorded strict `1e-6` check; the exact classifier readout itself is unchanged and the discrepancy is retained. This output-factor decomposition measures the explicit event/gate multiplication while preserving the model's existing internal ancestry; it does not establish a global causal allocation or prove that one factor is the only bottleneck.

The binary-event trace was nearly constant across time. With the existing Pearson classifier, zero-variance event traces yield zero similarity, explaining the zero-event-readout scores and groups; this diagnostic readout is not a proposed model. The gate-only readout is likewise diagnostic, not a trained candidate.

No training, threshold change, full320 follow-up or weights were produced. Failed attempts and logs are preserved alongside the result JSON files. The user's hard stop remains 2026-10-07 10:37:49 UTC; no work should be queued beyond it, and the parent owns the goal-pause action.

````

</details>

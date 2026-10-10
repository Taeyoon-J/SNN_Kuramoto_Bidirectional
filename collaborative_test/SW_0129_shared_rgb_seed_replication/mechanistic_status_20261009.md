# Mechanism and scaling status after the seed-1 replication

Neither contribution from every gating stage nor improvement with dataset size is established. A reconstruction-credit improvement alone cannot establish either claim.

## What the existing implementation actually gates

`MembraneLayer.forward` computes the usual leaky update and reset, and preserves the previous membrane only when `g_wave_t == 0`. The native sinusoidal gate is continuous: `(1 + sin(delayed_phase_mean))/2`. It is generally nonzero; therefore a small gate normally reduces the emitted spike amplitude but does not reduce the membrane integration update. This code observation identifies a possible mismatch between phase-dependent integration and the implemented dynamics. It does not prove that replacing the recurrence improves segmentation.

The SW0108 seed-0 intervention measured binary-event occupancy about 0.999785 before forcing events on. Under this condition the emitted trace `event * gate` nearly equals the continuous gate. Forced-on events and removed dendritic/membrane retention did not reduce the evaluated FG-ARI materially. Conversely, disrupting upstream coupling/drive caused collapse. These are frozen-model interventions, not retrained architecture ablations, and cannot establish an independent contribution from every stage or from their training gradients.

SW0126 made events vary through history-centered thresholds but degraded native QCC performance; SW0127 showed degradation even before training the history adapter. Repeating sparsification or replacing the classifier without preserving a native source comparison is not supported by those results.

## What more data has shown

The existing equal-update seed-0 SW0109 comparison gave FG-ARI 0.822080, 0.819832, and 0.818679 at 2,500, 10,000, and 70,000 unique images. This does not establish a scaling benefit or a statistically reliable decrease. Fixed updates also change repeat exposures per image. Both matched-compute and adequately trained learning curves are needed before attributing the result to dataset size.

SW0129 uses 4,096 unique additional training images from the registered 70k pool, starting from SW0095. It is a continuation pilot, not a new complete 70k training result. Seed 2 failed its unchanged assignment-use preflight and was not trained. Seed-1 endpoint evidence must stay separate from selected historical seed 0 and must not be reported as a three-seed mean.

## Peer check

The peer `patch_v2` branch remains at `697eeb67dc031325b63153e678cea8cbc602da31`, verified by fetch in this cycle. PV2_0053 reports GT-audited appearance/position geometry, not an improved trained score. Our SW0110 already tested an XY graph route without FG-ARI improvement. PV2_0048 also found worse scores with more unique images at fixed compute, with fewer exposures per image; its full-epoch large-data follow-up was cancelled. Neither peer result proves a solution to our scaling requirement.

The next architecture decision is under separate review: preserve useful native event/phase information while making the downstream temporal mechanism testably functional. No new architecture training, recipe change, or full-data expansion has been launched by this note.

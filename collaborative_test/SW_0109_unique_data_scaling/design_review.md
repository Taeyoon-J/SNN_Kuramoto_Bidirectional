# Design review: peer evidence, SW0109 limits, and one next experiment

Reviewed peer `origin/patch_v2` commit697eeb6, especially PV2_0048/0051/0052/0053. This document records design only. It launches no jobs, changes no runner, and does not supersede SW0107/0108/0109. Reserve90640-90959 remains unread.

## What the evidence supports

PV2_0048 reports declining FG-ARI as distinct images increase at fixed approximately240k presentations. Exposures per image decrease and the large-data loss is still descending. This establishes a failure under that compute/architecture/training recipe; it does not prove that scaling cannot help with more computation or a different model. PV2_0051's GT-based within/between statistics describe the saved encoder endpoints. They cannot establish what additional optimization would have learned. Its corrected validation-row provenance is an important check to retain.

PV2_0052's tiny gain beyond1024 frames applies to those checkpoints and its different classifier contract. Keep our1024/settle512 endpoint; no longer-window sweep follows from this result.

PV2_0053 usefully identifies minimum-distance object pairs hidden by average feature separation. Its object identities and coordinate-weight comparisons use GT. Treat them as feature-geometry diagnostics, not predictions, certified impossibility of separation, or permission to import the GT-favored weight1 into our model.

Our `ImageConditionedGraph.forward` already adds a grid-distance penalty, and with registered settings uses three feature-aware geodesic relaxation steps, BEFORE top-k. Thus position is already available as a pairwise connectivity prior. What is absent is XY in the node embedding `normalize(projection(gamma))`. Explicit coordinate features would change that embedding; they would not introduce the first spatial information into this model.

## What SW0109 can establish

The three nested2500/10000/70000 pools share audited2500-image source cores, fixed encoder/graph, fresh Adam, exactly4375 updates and70000 additional exposures. This is a controlled comparison of downstream data diversity at fixed additional compute, conditional on inherited pretraining. It is not from-scratch scaling of every component. Smaller pools receive28/7 passes; the full pool receives1. Fewer repetitions are part of the fixed-budget question, not proof that the larger pool is intrinsically less learnable.

Because the encoder and graph are frozen, native gamma and its graph node-feature geometry do not improve with N. Any improvement must arise in eligible downstream core parameters. A failed curve cannot rule out encoder/graph feature learning at scale. Conversely, fixed-prefix training loss improvement does not guarantee improvement in thresholded actual-spike connected components.

Report all three seeds/counts at the fixed4375 endpoint, actual prefix unique counts, all three metrics, and the inherited exposure provenance. Bootstrap the same sampled320 image indices across ALL three seeds before averaging seed differences. Independent image resampling per seed loses their shared image correlation. These intervals describe image uncertainty conditional on these three checkpoints, not population-level seed uncertainty.

The nine conditions can run up to four concurrent independent workers when four devices are genuinely free, one worker per device. Current authorization is GPUs0-3; three free devices permit three workers. Serial execution is unnecessary scientifically. Preserve dependency order: validated per-seed preflight before its training, training before its evaluation; do not require unrelated conditions to finish. Reserve a device before its CUDA process registers, check owners at launch and during work, and stop only an owned worker if a foreign owner appears. Do not restart a live coordinator to widen it without auditing its state. Luna owns implementation and partial-output handling.

## Proposed SW0110: coordinate information in graph node features

One paired256-update test after SW0108/0109, with source selection fixed in advance to the completed SW0097 `positive_frozen` cores for seeds0/1/2. Use the registered native8-channel gamma and encoder unchanged. Candidate graph embedding is:

```text
h[n] = legacy_projection(gamma[n])       # original16-dimensional embedding
z[n] = normalize(h[n] + P * xy[n])       # P has shape16x2
```

XY are fixed normalized patch centers in[-1,1]. Preserve the existing spatial/geodesic priors, top32, temperature, gain, symmetry, and all neuronal dynamics. Coordinates enter ONLY this graph node projection, not gamma preprocessing, oscillator sensory drive, theta initialization, a mask head, or the classifier. The live learned graph still drives the actual Kuramoto-to-dendrite-to-membrane-to-spike rollout.

Initialize P to zero, without RNG consumption. This gives an exact legacy starting function rather than choosing a coordinate weight from the peer's GT diagnostic. On the first4 registered TRAIN batches require finite, nonzero P gradients under the actual old loss; the gradient check approves or rejects this initialization and never tunes its scale or direction. If it is inert, stop this recipe instead of adjusting initialization using validation. P itself learns the coordinate scale; there is no fixed-weight or temperature sweep.

Freeze every legacy graph parameter in both arms. Both arms train the same previously eligible downstream core parameters under phase primary plus5x positive-product actual-spike affinity. Candidate additionally trains P, LR3e-5; control keeps P zero. Use fresh Adam, the same256xB16 ordered4096 training IDs, shuffle117/118/119,64/32, and the registered clip1 recipe. Record the extra parameter/gradient contribution and clipping; candidate core gradients must not be discarded or replaced by position-only gradients. This tests a new positional feature route without repeating unrestricted graph adaptation from SW0097. Persist P and its configuration explicitly; candidate loading must reconstruct the augmented graph before strict loading, rather than drop new weights or substitute a graph at scoring time.

First check zero-P graph/Q/core output equivalence to legacy on actual TRAIN data, absence of changes to native gamma/drive, finite source gradients, checkpoint round-trip, and a real batch memory/update check. Then complete all three paired runs and score full320 with original B8/T1024/settle512/.50/min2/largest-background masks from actual component spikes. Do not use a16-image metric to select a coordinate setting. Predictions are frozen before GT scoring.

Require candidate FG-ARI mean at least+.01 against BOTH its new matched continuation control and archived SW0097, at least two positive seed deltas against both, positive paired-image difference intervals, and IoUs at least Slot+.05. Report graph changes, same-instance fragmentation and cross-instance merges after prediction as diagnostics. Coordinate variation may fragment one object or background, and the old global objective may not teach correct binding; an FG failure stops this single recipe without a position-weight/LR sweep.

A success would support explicit graph-node coordinate information for this fixed recipe. It would not prove every neuronal stage contributes or that full70k scaling is solved. Those require SW0108's physical-rollout evidence, SW0109's fixed-budget curve, and later matched full70k plus frozen independent confirmation. If SW0108 shows graph coupling has no material effect on actual masks, prioritize interpreting that failure before attributing a positional graph intervention to the spike pipeline.

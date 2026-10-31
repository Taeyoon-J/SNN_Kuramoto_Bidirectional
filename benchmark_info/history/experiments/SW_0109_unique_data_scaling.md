# SW_0109_unique_data_scaling

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## design_review.md

<details>
<summary>design_review.md</summary>

Source SHA256: `488405b8ebae370b80e7af161528fac3257f9cba18dc1af1a6d0e35f48ff74c7`

````markdown
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

````

</details>

## peer_review_20261008.md

<details>
<summary>peer_review_20261008.md</summary>

Source SHA256: `31cb90991a18c7f2dce3181cdbfc742e8447ce1a2abc42339044f8bb2564caef`

````markdown
# Peer evidence reviewed before the next scaling study

Source: origin/patch_v2 commit697eeb6, fetched2026-10-08. Read
PV2_0048_data_scale, PV2_0051_data_quality, PV2_0052_window and
PV2_0053_position_diag. Their scores use a different evaluation contract;
none replaces our native320 Slot reference.

PV2_0049_slot_draws was also reviewed. Peer Slot inference now uses three
seeded draws per training checkpoint and verifies identical labels on a
repeated same-seed run. Its large third-training-seed degradation explains
the changed three-seed reference. Our existing Slot SW0092 uses fixed
inference seed0; report that distinction instead of importing peer means.

At approximately240k presentations, peer unique-image counts6000/20000/40000
gave FG-ARI .6779/.5983/.4790. Repetitions per image also fell40/12/6.
This demonstrates degradation at that fixed compute budget, not that more
data cannot help with sufficient optimization. Their within/between-feature
diagnostics describe those fitted encoders; they do not resolve an unrun
longer-training counterfactual. SW0109 likewise must report its result as
matched-compute downstream scaling from inherited2500-image cores.

Peer PV2_0053 finds a close-object-pair separation issue in appearance-only
features. Appending coordinates reduces its GT-defined unseparable-image
fraction. This is a diagnosis, not a trained mask-score improvement. Our
graph already has a spatial-distance prior; explicit coordinate features
would be a distinct proposed change requiring a controlled actual-spike
evaluation. Inspect our own feature geometry before adopting the change.

Peer PV2_0052 gains only about.0005 FG-ARI from extending inference1024 to
4096. Avoid another expensive window sweep without evidence that our
current checkpoints have a sampling bottleneck. Our SW0098 longer-training
window already failed the fixed three-seed FG-ARI comparison.

Current causal gate evidence remains pending corrected SW0108 baseline
reproduction. Test-time neutralization measures checkpoint sensitivity;
matched retraining ablations are still required for a stronger necessity
claim in the paper. No GT-derived teacher or classifier setting is adopted.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `e5849e83f8777b4ffacc4f21a84d51667c17fde49b3691c96881b0c959b3f83a`

````markdown
# SW0109 unique-data scaling

This local runner prepares three matched seeds at 2,500, 10,000, and 70,000 unique source-image counts. Every condition receives 70,000 B16 exposures, uses the audited seed-matched pre-large-data core, fresh Adam at `3e-5`, 64/32 train/settle windows, the registered phase-plus-five-positive-actual-spike objective, and a frozen encoder and graph. Each smaller-pool cycle is independently shuffled without replacement; no partial cycle or dropped batch is used.

The registered inputs are `data/SW_0090_large_unique_scale/gamma_train_70000.pt` and `data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt`. The runner validates their existing manifest schemas, exact ID segments, encoder/preprocessing identities, and file hashes. It does not regenerate these caches. Before training, each seed must pass four real B16 throwaway Adam updates for each of the three pool sizes, with finite nonzero gradients and an unchanged learned graph. The exclusive coordinator can run up to four independent workers, one per genuinely free GPU. Each seed's baseline and training wait for its validated preflight; each endpoint evaluation waits for its own completed, validated training output. Foreign GPU owners are not shared or stopped.

The primary endpoint evaluates all 320 registered validation images with the fixed B8, T1024/settle512, `.50` threshold, minimum group size 2, largest-component background readout. It retains all per-image metrics, recomputes aggregate means, reports the registered 70k-versus-2500 FG-ARI gate and all IoUs, and uses a paired-image bootstrap conditional on these three trained cores.

Status: authorized source deployment and Frontier CPU validation are complete; Python compilation, protocol parsing and all thirteen scheduler/manifest/preflight/summary tests pass. The coordinator was verified alive and waits for SW0108 completion; the exact SW0106 reviewed queue dependency is already complete with no active worker. All24 tasks are queued; no SW0109 GPU preflight, training or evaluation has started. Independent conditions will run on available exclusive GPUs after their validated prerequisites finish. Runtime snapshots are evidence of the observed state, not a claim that the registered endpoint is complete.

## Checkpoint-only diagnostic evaluation (2026-10-09)

The old queue waiting state is stale: its recorded PID287264 was absent. Three seed0 training attempts each reached4375 finite updates and saved final core/history, then exited rc1 with KeyError4096 while serializing the manifest. Missing training manifests and markers remain missing; no training retry or completion certification was performed. The original runner bytes are recoverable exactly from Git83f99c94c99050b58bd2866c8a647338b2de6c2a (SHA08a06572a429fe78888184e66852e5e845c79713b342d70e524be4960cb6ca91).

A separate checkpoint-only diagnostic queues the three existing final cores into fresh N2500/N10000/N70000 directories, using unchanged fixed320 B8 T1024/512 .50/min2/largest-background spike readout. Local and actual server focused tests passed9/9; current runner/evaluator/core SHA and actual validation cache contract passed before launch. Supervisor1425494 and children1425545/1425562/1425580 were verified live on GPUs0/1/3. Occupied GPU2 was excluded. No automatic retries or output overwrites.

Scores pending. These are older frozen-encoder/graph recipe checkpoints initialized from pre-large-data sources, not SW0097 or the forthcoming adaptive model. Seed0 results cannot establish three-seed scaling, and checkpoint evaluation does not repair missing historical training provenance. See `recovered_checkpoint_eval_protocol.json`, `recovered_checkpoint_eval.py`, `results_archive/recovery_inventory_readonly_20261009.json` and `results_archive/recovered_checkpoint_eval_launch_state_20261009.json`.

## Completed checkpoint diagnostic (2026-10-09)

All three evaluations passed rc0 and full320 provenance/metric validation; supervisor1425494 is absent. FG/foregroundIoU/objectIoU: N2500 .822080/.734862/.650085; N10000 .819832/.730208/.648823; N70000 .818679/.730915/.654075. Equal4375updates/70000additional exposures. No demonstrated FG scaling gain, slight objectIoU increase at70k. One historical source seed only; no current97/adaptive-model or three-seed scaling claim. All original training metadata failures remain disclosed. Full per-image reports and paired-image CIs are archived in recovered_checkpoint_eval_results_20261009.json and recovered_checkpoint_eval_summary_20261009.json. Running paragraphs above are historical and superseded.

````

</details>

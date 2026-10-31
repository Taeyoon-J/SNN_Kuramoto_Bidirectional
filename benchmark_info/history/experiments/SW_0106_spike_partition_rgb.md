# SW_0106_spike_partition_rgb

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## design.md

<details>
<summary>design.md</summary>

Source SHA256: `aa6d54f1b455f06600ef72867320246feb54fb6f3ffa5713ef46ba895e9f12cf`

````markdown
# SW0106: teach the actual spike partition through image explanation

Status: approved design, runner, and owner-aware queue. The queue waits for SW0105's terminal state and valid full320 evaluations for all three seeds, then for a genuinely free GPU. It does not share or interrupt another user's GPU. SW0106 uses SW0095 rather than assuming SW0105 improves FG-ARI.

## Question and controlled change

Can RGB explanation improve grouping when its credit reaches the encoder and graph only through the actual spike partition? SW0094's naive joint training lowered full320 FG-ARI by .012579 relative to its frozen control. SW0090's continuing global synchrony objective also improved its training loss while FG-ARI deteriorated. Those findings motivate an assignment objective, not another encoder unfreeze under only global density/bimodality targets.

Use one paired seed0 pilot. Both arms train the registered encoder, learned graph, and whole core. Both retain the original phase primary plus 5.0 times the positive-product actual-spike auxiliary. Candidate adds reconstruction credit through spike assignment; control trains the same decoder but disables that credit. There is no graph teacher, mask/count supervision, mask decoder, phase-center assignment, analytic pooled-color reconstruction, or feature-preservation anchor.

## Assignment and decoder

Use actual component spikes `[B,4,256,64]`, settle32, and the production `spike_synchrony_affinity` to obtain live `Q`. Its per-component positive Pearson product remains unchanged. Obtain detached patch labels from the exact production classifier: threshold .50, minimum group2, largest component background. Include its background label0 and every returned foreground label in one-hot `H[N,K]`; discarded singleton patches stay background. Match production traversal/tie handling rather than implementing a different symmetric threshold rule.

`K` is one plus the number of predicted foreground groups. It is not fixed to the true count or capped at11. This preserves the existing classifier even when it predicts too many groups. Iterate/chunk per image so an unusually large K does not inflate an entire padded batch. With K=1, reconstruction trains the decoder but produces zero assignment credit; record those images without inventing foreground slots.

For each image, with all component membership/count tensors detached:

```python
Q0 = Q * (1 - identity)                 # remove self-credit only in surrogate
count = H.sum(dim=0)
other_count = (count[None, :] - H).clamp_min(1)
A = (Q0 @ H) / other_count
P = softmax(A / 0.10, dim=-1)          # sole assignment driver is actual Q
W_candidate = H + (P - P.detach())     # forward exactly H
W_control = H                         # no reconstruction assignment credit
z = (W.T @ F.detach()) / W.sum(dim=0)[:, None].clamp_min(1e-8)
```

`F[N,8]` is the existing standardized/clipped native gamma from the registered encoder. Detach F only inside this content pool. F remains live on the encoder-to-graph/core path. Consequently new reconstruction gradients reach encoder/graph/core only through `Q -> W`, including W's effect on both content pooling and reconstruction mixing. Decoder slot contents may contain image appearance; they cannot choose masks.

The same shared MLP applies to every slot and patch: concatenate z8 with normalized patch-center coordinates x/y; linear10->64, ReLU, linear64->64, ReLU, linear64->3, sigmoid. It predicts RGB only, with no mask logits or slot identity embedding. Reconstruct each patch with `sum_k W[n,k] * decoder(z[k], xy[n])`. The target is the exact mean RGB of its 8x8 pixels in the 128x128 input, in [0,1]. Reconstruction loss is plain mean squared error over256 patches x3 channels, then equally averaged over images. Column permutation changes neither the model nor the represented partition.

Forward reconstruction uses the exact scored hard partition. The straight-through derivative is biased and cannot split a completely collapsed K=1 image. This is a source-continuation experiment, not a proof that reconstruction solves object count or same-colored-object separation. Decoder inference masks are never substituted for actual spike masks.

## Paired recipe and calibration

Start both arms from the whole SW0095 seed0 final core, registered encoder, and identical fresh decoder. Keep raw gate, full event credit, preprocessing scalar mean/std/clip, graph top32 and spatial/geodesic settings, neuronal dynamics, and registered losses unchanged. Use exact shuffle117 training IDs from the established 70k pool, first4096 without replacement; batch16, 256 joint updates, actual64/32. Encoder LR3e-6; graph and remaining core LR3e-5, Adam. SW0097 is a frozen reference, not the new matched joint control.

First perform a fixed32-batch decoder-only warmup on the first32 batches of that sequence. Freeze the entire source network; cache detached H/F and RGB targets. Both arms start joint training from the same resulting decoder state. No extra unique images or GT are introduced. Decoder LR3e-4, Adam, gradient clip1. The warmup makes calibration use a decoder that has learned some RGB meaning rather than arbitrary random output derivatives.

After warmup, calibrate one lambda on the first4 training batches, before updates: `.25 * median(norm(old_joint_gradient) / norm(reconstruction_joint_gradient))`. Compute each norm over the unique union of trainable encoder/graph/core parameters, without counting graph parameters twice. Require finite nonzero reconstruction credit; stop on invalid calibration rather than replacing a zero denominator or tuning lambda. Retain this single seed0 lambda for later seeds, with no validation or coefficient sweep.

Candidate joint gradients use `old_loss + lambda * reconstruction`; control joint gradients use `old_loss`. Decoder gradients in both arms use unweighted reconstruction. Use `autograd.grad` with explicit parameter lists to avoid accidentally doubling/scaling decoder gradients. Clip the joint parameter union to1 and decoder parameters separately to1; use separate optimizer states. Preserve the same optimizer, decoder warmup, initial states, IDs, exposure order, and update count in both arms.

## Data preparation and preflight

Build a derived contiguous uint8 RGB cache with a sequential/chunk-aware HDF5 pass. Cache only the exact training pool IDs0-999 plus1640-70639, in existing pool-index order; preserve all128x128x3 bytes. Size is about3.44GB. Record source identity, shape/dtype, the exact ID mapping/list hash, and block hashes; compare every written source block byte-for-byte before marking complete. Read-only mmap, bounded workers, pinned batches, and prefetch should remove SW0094's repeated random compressed-HDF5 bottleneck. Do not cache or inspect reserved90640-90959. Validation1320-1639 is prepared separately with the existing prediction/evaluation contract. No new image normalization or preprocessing fit is allowed.

Before the GPU pilot:

1. Check strict source checkpoint loading and unchanged core architecture. Registered source RGB-to-gamma matches the established cache at its existing2e-5 tolerance. Candidate/control inputs and initial actual forward, Q, labels, old loss, and reconstruction values match exactly. H equals production predicted patch labels bitwise. Warmup leaves source encoder/graph/core state bitwise unchanged.
2. Check finite W/P/loss/gradients and real reconstruction gradients into Q, encoder, graph, and upstream core on each first4 batch. Log norms by family, selected-edge graph gradients, clipping, K, and K=1 coverage. Top-k membership is discrete: gradients update selected graph values/features; do not claim every topology decision is differentiable. Control reconstruction gradients into those parameters must be absent/zero.
3. After warmup, on those same4 TRAIN batches, freeze original H predictions before any GT. Compare reconstruction with original H versus one within-image random row permutation of H (seed106; preserve component sizes, recompute pooled z). Require mean shuffled MSE minus original MSE >0 and finite nonzero Q credit to proceed. This is a weak assignment-usage check, not an object-quality claim. Slot-column permutation must preserve the result within a declared1e-6 floating tolerance. Do not alter the decoder, temperature, or masks after seeing the check.
4. Benchmark a complete real batch backward/update and worst observed K on a genuinely free GPU. Record throughput and memory before authorizing training. Keep defaults legacy; disable the entire optional reconstruction path for existing evaluation/checkpoint loading.

## Evaluation, expansion, and limits

Run both seed0 arms for their complete fixed256-update budgets; evaluate both full320 validation images using actual `event*gate` spikes, original1024/settle512, batch8, threshold.50, minimum group2, largest component background, and unchanged patch metrics. Use regenerated gamma from each trained encoder with frozen registered preprocessing. Score no P/W/decoder-derived prediction. Report paired deltas against the new joint control and archived SW0097 seed0.

Extend to seeds1/2 only if seed0 FG-ARI is at least .01 above both its joint control and SW0097, and both IoUs retain Slot+.05 margins. Carry forward all seeds/results; no successful-seed selection. Later paired seeds use118/119, their own SW0095 sources and fixed32-batch decoder warmups, and the seed0 lambda.

Three-seed promotion to one matched full70k continuation requires mean FG-ARI at least .01 above both joint controls and SW0097, at least two positive seed deltas against both references, foreground IoU >=.25358913 and object IoU >=.25693698, and a paired image-bootstrap FG difference interval supporting improvement against both references. The bootstrap does not establish seed generalization. At the fixed endpoint, record reconstruction/Q gradients, H row-scramble usage, foreground/group counts, and grouping errors; no readout or GT-driven setting changes.

If promoted, freeze this recipe before a complete70000-image continuation for all3 paired seeds. Candidate/control receive the same full-pool order and additional update budget; the efficient RGB cache serves both. Do not silently substitute the frozen SW0097 pilot for that joint training control. Full70k and eventual independent evidence remain required; the pilot does not finish the user's three-metric objective. A failed mechanism or FG gate stops this recipe rather than spawning temperature, coefficient, decoder-capacity, or seed sweeps.

````

</details>

## followup_queue_design.md

<details>
<summary>followup_queue_design.md</summary>

Source SHA256: `e9a723ede5a07b99b666cb9c7625e2cfb9eaccac364e360371cfcde9a5bb5203`

````markdown
# Registered conditional follow-up queue

The user explicitly requested that all necessary follow-up tests be registered in a server queue on 2026-10-07 (local date). This extends the existing seed0 pilot; it does not restart or duplicate live coordinator PID4012262. All stages use exclusive GPUs0-3, at most one worker per device, with no foreign process sharing or termination.

The independent follow-up coordinator waits for the existing seed0 coordinator to finish and validates both complete320-image evaluations before applying the existing seed0 expansion gate. A failed gate stops this recipe and preserves every result; it does not automatically change temperatures, losses, thresholds, checkpoints, or seeds.

1. Seed0: existing shared decoder preflight, paired candidate/control256-update pilots, and both full320 evaluations.
2. If the registered seed0 gate passes: each of seeds1/2 receives its own shared32-batch decoder warmup and preflight, followed by paired256-update training and full320 evaluation. Both arms of each seed share the same warmup artifact and optimizer state. All seeds reuse the seed0 lambda.
3. If the registered three-seed pilot gate passes: freeze the recipe and queue all six candidate/control full70000 continuations. Each starts from its own completed pilot core, encoder, decoder, and BOTH saved Adam states. Missing optimizer state is an error; it is never silently reset or reconstructed. Each receives exactly4375 updates of batch16, one complete without-replacement pass over the same70000 RGB images, using shuffle117+seed. Candidate and control see identical orders. Lambda, learning rates, gradient clipping, losses, readout, and preprocessing remain fixed.
4. Evaluate all six full70000 endpoints on the original320 validation images with separately regenerated gamma from each trained encoder. If candidate three-seed means exceed matched Slot means on all three metrics, freeze model/encoder/preprocessing/checkpoint-shard/training-protocol/evaluator hashes before unlocking reserve90640-90959.
5. Run frozen reserve evaluation for all three candidate and matched Slot endpoints and all three prespecified full controls. Our readout remains actual event-times-gate spikes with B8/T1024/settle512/.50, minimum group2, largest-component background. Slot uses its original B1/11-slot/3-iteration checkpoint, inference seed0, perimeter background, native normalization and no crop. Use the same modal8px16x16 patch metric definitions. No reserve images, masks, or gamma are read before the freeze.

The pilot three-seed promotion gate is unchanged: mean FG-ARI at least+.01 against both the new joint controls and SW0097, at least two positive seed deltas against both, foreground IoU>=.25358913, object IoU>=.25693698, and deterministic paired-image bootstrap intervals supporting positive FG differences against both references. Preregister10000 replicates, RNG seed1060106,95% percentile intervals, resampling common image IDs after averaging seed-paired differences. Bootstrap intervals describe image uncertainty and do not establish seed generalization.

Final success requires actual reserve three-seed means strictly above matched Slot on FG-ARI, foreground IoU, and matched-object IoU. Report all seeds, image intervals, seed spread, and training/pretraining exposure differences. A validation-only win does not complete the broader goal. If reserve confirmation fails, preserve and report that result; do not tune against the reserve.

Server jobs use immutable task records and source/artifact hashes. Errors, invalid outputs, lost state, or a foreign owner appearing on an active GPU stop the affected recipe without automatically repeating work. The queue is separate from goal-mode state; goal-mode pauses are never inferred from ordinary GPU waiting.

The user's additional SW0107 request is an independent sibling branch in this same owner-aware queue: exact unfiltered official CLEVR70000 CPU preparation, real-batch preflight, all three current-best frozen-graph model continuations, and our fixed320 evaluation. A SW0106 promotion rejection stops only SW0106 descendants, while SW0107 still runs. Unexpected runtime failures preserve partial outputs and stop for diagnosis; they are not promotion rejections.

## Independent SW0107 official-domain branch

The same queue also runs SW0107 regardless of whether an SW0106 promotion gate passes. At queue startup, while SW0105/SW0106's seed0 work may still be GPU-bound, it uses CUDA-disabled subprocesses to freeze the three completed SW0097 positive-frozen source checkpoints and registered encoder/statistics before exporting and encoding exactly the unfiltered first70,000 `clevr:3.1.0` train examples (no initial skip), with the established native crop/resize and `[0,1]` RGB contract. No masks are requested. Before training each seed it runs a real first shuffled B16,64/32 forward/backward and a throwaway optimizer update, verifying finite gradients and bitwise-frozen graph; failed preflight stops SW0107 without updates or retries. If the SW0106 seed0, three-seed, or full-validation promotion gate rejects, the queue preserves that result and proceeds to SW0107 preflights. It applies one additional raw-gate, phase-plus-5x-positive-product actual-spike pass per source seed, with graph and encoder frozen, and evaluates all three on the unchanged320 native validation IDs. Each seed, its matched SW0097 result, and the matched Slot epoch10 means are reported. This is an additional cross-domain continuation from already-trained source cores, not a from-scratch comparison. SW0107 never reads the reserved90640–90959 data.

````

</details>

## posthoc_replication_design_20261009.md

<details>
<summary>posthoc_replication_design_20261009.md</summary>

Source SHA256: `6f981b74758a8a29cd6dee7d9792605016a98fb8b2caf67447dbd81bb0fa1f68`

````markdown
# SW0129 design decision: exact shared-decoder replication

SW0128 completed successfully as a rejection-only diagnostic. All six source-seed/distance-bin checks failed. Confident positive spike relations are not reliable object identity targets, so that self-teacher recipe is closed without threshold tuning or training.

The next hypothesis is explicitly selected retrospectively from SW0106, not a reversal of its original failed expansion decision. Seed0 showed FG-ARI and matched-object-IoU improvement over source97 with positive paired-image intervals. Its original +.01 FG margin failed and remains recorded. SELECTION_POLICY allows retaining informative improvements; replication on both previously untested seeds is needed before interpreting this selected result as generalizable.

Replicate the original SW0106 shared RGB decoder with actual-spike hard-QCC straight-through assignment credit and its detached-assignment control. This is different from SW0122's analytic color-mean objective and SW0126's newly learned binder. The decoder supplies training loss only; inference masks remain native actual spike QCC. Do not use the rejected SW0128 relations, target instance counts, or GT masks in training.

Freeze each seed's SW0095 source, original 4096 training IDs and shuffle117+seed,32xB16 decoder-only warmup, seed106 decoder initialization, shared seed0 lambda23250.431374718348,256xB16 updates,T64/settle32, core/graphLR3e-5, encoderLR3e-6, decoderLR3e-4, and both Adam states after warmup. Both arms share all quantities except assignment-loss credit. Preserve original106 files/artifacts. Regenerate native validation gamma from each trained encoder and evaluate fixed320,B8,T1024/512,.50,min2,largest-component background. Reuse hash-bound original seed0 paired results instead of retraining selected seed0.

Preregister the new three-seed continuation gate before seed1/2 results: meanFG strictly above both source97 and matched controls; at least two seedFG improvements against each; shared-image paired bootstrap10000 replicates with frozen RNG129 has positive95% lower bounds for both; foregroundIoU>=Slot+.05 and objectIoU>=Slot+.05. Report all three metrics and all seeds, even on rejection. Bootstrap covers image uncertainty, not training-seed certainty. No temperature,lambda,learning-rate,seed or epoch search after scoring. No70k expansion before this gate passes.

A positive replication would establish evidence for reconstruction assignment credit under this recipe. It would not establish independent event credit, usefulness of every gating, increased performance with unique-data count, or untouched-split superiority. Those remain separate requirements after a useful candidate. Pretraining and upstream70k exposures must be disclosed in any scaling comparison; varying only continuation data does not establish from-scratch data scaling.

Sol6.1 design review approved the exact replication scope on2026-10-09. The peer patch_v2 head was rechecked and remains697eeb67dc031325b63153e678cea8cbc602da31. No new peer experiment changes were available for incorporation.

````

</details>

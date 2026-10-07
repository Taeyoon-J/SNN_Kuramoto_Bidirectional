# SW0106: teach the actual spike partition through image explanation

Status: approved design and preparation. Training requires a genuinely free authorized GPU and root's runner/preflight review. All GPUs were occupied by other users at the latest ownership check. This document does not authorize an automatic queue or interrupt other users. SW0105 has completed training but its evaluation is pending; SW0106 uses SW0095 rather than assuming SW0105 improves FG-ARI.

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

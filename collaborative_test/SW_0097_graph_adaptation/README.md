# SW0097 — matched graph adaptation on all source seeds

SW0095 validation seed0/1/2 FG-ARI is .8241/.7413/.7616; inherited graph
parameters remain frozen. SW0094 tested graph learning only on seed0, where
the80-image FG-ARI dropped while foreground/count quality improved. That
single stronger source does not establish whether weaker source seeds can
benefit. Test this hypothesis with a matched per-seed control.

For each seed, start both arms from its **whole SW0095 final core**, encoder
and preprocessing fixed. Train4,096 existing training images,256 updates of
batch16, shuffle seed117/118/119 matched within each pair. Both arms use
positive classifier-aligned spike loss, core LR3e-5 and all existing priors.
Only difference: graph generator frozen versus trainable. No new data.
Graph learning includes its projection/temperature/coupling/spatial/geodesic
parameters; all changes are recorded. Real CPU backward checks for all six
seed/arm pairs precede GPU launch. Gradients must be finite and graph gradients
nonzero in the adaptive arm. Failed runs are diagnosed rather than restarted.

Full320 validation IDs1320–1639 for each arm, original batch8,1024/512 spike
classifier,.50 threshold,largest-component background,16x16 modal patch
targets. No test/holdout scores select model settings. Current95 unchanged
validation scores remain an additional baseline, not the causal control.
Compare graph adaptation against each seed's equally trained frozen arm;
report all seeds and means, including adverse results.

Total downstream exposures per seed become144,096; graph pretraining and
encoder provenance remain inherited. This is a short structural pilot on
models already trained on the complete70k pool, not a fresh70k pass. Only
expand graph learning to another complete70k pass if this matched pilot
supports improved object separation. Retain IoU tradeoffs per SELECTION_POLICY.
GPU0/1 only; do not share GPUs with other users. Holdout90640–90959 stays unread.

The shared SW0094 runner accepts optional explicit source checkpoint and
shuffle seed. Existing SW0094/95 defaults remain unchanged; their completed
recorded runs are not relaunched. Never describe this recipe as a rerun of95.

## Completed matched comparison

All six runs finished with256 finite updates and original batch8 full320
validation. For each seed, both arms have identical source SHA/training IDs.
Frozen graph parameters stayed unchanged; adaptive graph parameters changed.

| Arm / seed | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| Frozen0 | .824868 | .719872 | .655663 |
| Graph0 | .820099 | .716549 | .649673 |
| Frozen1 | .742658 | .590283 | .584964 |
| Graph1 | .736645 | .585398 | .581440 |
| Frozen2 | .761519 | .421056 | .563985 |
| Graph2 | .751176 | .423373 | .558600 |
| Frozen mean | **.776348** | **.577070** | **.601537** |
| Graph mean | .769307 | .575107 | .596571 |

Graph adaptation loses FG-ARI in all3 seeds and all3 mean metrics versus
matched frozen continuation. Seed2's small foreground-IoU gain does not rescue
grouping. Do not expand this exact adaptive-graph objective to a full70k pass.
This is evidence about this LR/objective/continuation recipe, not a requirement
to keep graph parameters fixed forever. Raw manifests/history/evaluations are
archived for every condition; all-arm means are in `results/summary.json`.

Frozen short continuation raises the validation FG-ARI mean by only .000703
over SW0095, too little to establish improved generalization. Do not consume
another independent holdout on this tiny gain. Next controlled training will
address short-training versus long-evaluation time windows while preserving
encoder/graph/classifier and the same validation-only selection protocol.

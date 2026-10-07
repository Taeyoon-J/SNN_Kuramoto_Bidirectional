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

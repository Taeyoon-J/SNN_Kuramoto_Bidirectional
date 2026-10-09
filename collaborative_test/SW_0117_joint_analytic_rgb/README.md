# SW0117 joint analytic RGB pilot

SW0117 is a separate registration of the same seed-0 paired pilot as SW0116: the whole matched SW0097 positive-frozen source, jointly trainable core/legacy graph/registered encoder, fresh Adam, 256 updates of batch 16, exact 4096-ID order, and unchanged phase plus five-times actual spike-product objective. The candidate adds the unchanged SW0115 analytic RGB loss. No model, loss, coefficient, data, optimizer, or readout recipe was changed from SW0116.

The prior SW0116 registered preflight stopped at a failed guard; its code and artifacts remain preserved. SW0117 clarifies two distinct comparisons. First, regenerated gamma is checked against the registered cache (maximum absolute difference at most `2e-5`), while cached-versus-live old-loss difference must be at most `2e-5` and production labels/H must match exactly. Individual cache-versus-live rollout trace differences are recorded as input-sensitivity evidence, not treated as an implementation tolerance. Second, an independent source-core reference receives the exact same regenerated gamma detached; its full traces, old loss, labels, and hard H must equal the live joint core exactly. This second comparison isolates implementation identity from input differences.

The training tasks include the external dependency `sw0117_source_endpoint_reviewed`, and the runner independently enforces it before creating a training output. The reviewed source endpoint report is SHA-bound to the source core, registered encoder/statistics, and its full 320-image evaluation contract. Its cached-gamma scores are `.8248677211561721 / .7198719167110539 / .6556628993752277`; regenerated live-encoder gamma scores are `.8251401617079835 / .7196062474701879 / .656379507803011`, with maximum gamma difference `7.152557e-7`. The fixed source was reproduced on IDs 1320–1639 with batch 8, 1024/512 steps, threshold `.50`, and no ground truth in prediction. The registered promotion gate still compares the candidate against both original SW0097 and the matched control. A missing or altered source report blocks training before any output directory is created.

All other preflight guards, the fixed 320-image evaluation contract, and the preregistered promotion gate are unchanged from the copied SW0116 protocol. A failed guard stops the recipe without tuning or seed expansion.

Local CPU tests passed 15/15, including source-report hash/schema/metric rejection and proof that training cannot create its output without the reviewed evidence. Python compilation, protocol JSON parsing, and whitespace checks pass. No server deployment, real-data preflight, training, commit, or push is included.

## Completed seed-0 pilot (2026-10-08)

The preceding paragraph records the predeployment state. Both deployed arms subsequently passed real-data preflight, completed 256 updates, and were evaluated on all 320 fixed validation images. The control scored `0.8168938447 / 0.7196492583 / 0.6481461872`; the analytic candidate scored `0.8262101307 / 0.7229130104 / 0.6595478231`.

Candidate-minus-control FG-ARI was `+0.0093163` (paired-image bootstrap 95% CI `[0.0021404, 0.0166691]`). Candidate-minus-original-source was only `+0.0013424` (CI `[-0.0058533, 0.0085446]`). The registered promotion gate failed; this recipe will not expand to more seeds or coefficient tuning. Full compact provenance and all three paired metric comparisons are in `seed0_paired_summary_20261008.json`.

Gradient norms were usually modest (candidate median `190.4`) but had extreme outliers (maximum `446,360,736`). This is a separate numerical-conditioning concern requiring diagnosis; neither gradient connectivity nor the small score increase proves semantic binding by the RGB objective.

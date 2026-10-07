# SW0098 ??longer training rhythm window

SW0097 graph adaptation failed under the original64/32 training window.
SW0094 temporal diagnostics found same-object connectivity loss despite
retained pair AUC, and short/long-window affinity differences. Test whether
training on256 steps with128-step settling better preserves the long-window
spike classifier than the original64 steps/32 settling. This is a hypothesis,
not a claim that window mismatch explains all degradation.

All three whole SW0095 final cores; encoder and graph frozen; identical
positive-product loss weights, LR3e-5,4,096 training IDs,256 updates x16 and
shuffle117/118/119. Reuse SW0097 positive_frozen controls: exact same source,
shuffle, images, optimizer updates and original full320 batch8 evaluation.
Do not retrain already completed controls. Only temporal training window
changes; longer differentiation/computation and gradient dynamics are part
of this intervention. Inference stays1024/512,.50 spike classifier,16x16
modal patch scoring,largest-component background and original batch8.

Real CPU backward/update preflights for all3 source seeds and a GPU full-batch
single-update memory check precede the pilot. GPU0/1 only when free; no
sharing another user's GPU, no automatic restart of a failed output.
Use validation1320??639 only. Holdout90640??0959 remains unread.
This is4,096 additional images on models trained on the entire70k pool;
expand to another full70k pass only if the controlled three-seed pilot supports
a meaningful grouping improvement. No extra training data or count labels.

Three real-source CPU backward/update checks passed. The actual batch16,
256/128 GPU check also passed; peak Torch reserved memory was3.465GiB.
The runner asserts actual component-history length, not only a configuration
value. Raw preflight manifests/history/completion markers are in `preflights/`.
Defaults64/32 remain unchanged for earlier recipes; no completed run is repeated.


## Completed validation results

All three registered seeds completed 256 updates at the actual 256/128
training window, with finite recorded losses/gradients, graph and encoder
frozen, and 320/320 valid images for all three patch metrics. Each source SHA,
training ID list, shuffle seed, batch, and optimizer update count matches its
SW0097 positive_frozen control. Inference used the unchanged batch8, 1024/512
spike classifier at threshold .50 on IDs1320-1639. Raw small records are in
`results/`; model weights remain in `trained_models/` on the server.

| Seed | FG-ARI | Foreground IoU | Matched-object IoU |
|---:|---:|---:|---:|
| 0 | .820852 | .723376 | .648856 |
| 1 | .740108 | .590483 | .584255 |
| 2 | .759277 | .423988 | .558246 |
| Mean | **.773412** | **.579282** | **.597119** |
| SW0097 matched frozen mean | .776348 | .577070 | .601537 |
| SW0095 unchanged mean | .775645 | .574029 | .599779 |
| Slot mean | .774933 | .203589 | .206937 |

Against the matched SW0097 controls, the mean changes are -.002936 FG-ARI,
+.002212 foreground IoU, and -.004418 matched-object IoU. Per-seed FG-ARI
falls for all3 seeds; foreground IoU rises in all3; matched-object IoU falls
in all3. Against unchanged SW0095, SW0098 improves foreground IoU but lowers
both grouping metrics. Its mean FG-ARI also misses the Slot mean. This
validation-only result does not meet the all-metric target. Holdouts remain
untouched; do not retune on them.

`results/summary.json` records the per-seed provenance and comparisons.

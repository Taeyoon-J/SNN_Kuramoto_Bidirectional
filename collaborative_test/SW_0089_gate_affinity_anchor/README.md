# SW0089: classifier-aligned gate-affinity preservation

SW0085 localized the first material loss in joint training to the gate, and
SW0088 showed that freezing the downstream rhythm core limits the damage but
still trades foreground and object grouping metrics. SW0089 adds a label-free
teacher loss at that exact interface.

For each component, the settled gated trace is centered and normalized in time.
Its patch-pair correlation is clamped at zero and multiplied across components,
exactly matching the fixed spike classifier's affinity. The loss is MSE between
the current learned encoder/graph affinity and the original SW0072 affinity on
the same image. It preserves the classifier-relevant temporal relationship,
rather than freezing raw activation amplitudes or graph edges.

All arms retain SW0088's frozen downstream core, seed1, 2,500 training images,
five epochs, graph/encoder LR `3e-5`, and reconstruction weight `.3` (the SW0088
setting with the largest object-IoU gain). H/I/J use coarse affinity weights
1/10/100. A one-update real-data preflight verifies the exact loss, nonzero
encoder/graph updates, and bitwise-fixed non-graph core before each run. The
fixed pilot remains IDs1320-1351, T256/settle64, vth `.06`, spike-CC `.35`.

## Result

The affinity anchor prevents a complete drift but still produces a metric
tradeoff. Weights 1/10/100 score `.675377/.454694/.512301`,
`.701279/.481901/.512234`, and `.699688/.475970/.516505`, versus the fixed
SW0072 seed1 pilot `.708757/.468105/.509881`. Weight100 improves foreground IoU
and matched-object IoU by `+.007865/+.006624`, while FG-ARI falls `.009069`.
No arm improves all three, so this direction is not promoted to full validation.

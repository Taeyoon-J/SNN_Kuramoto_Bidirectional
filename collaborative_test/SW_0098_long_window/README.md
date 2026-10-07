# SW0098 — longer training rhythm window

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
Use validation1320–1639 only. Holdout90640–90959 remains unread.
This is4,096 additional images on models trained on the entire70k pool;
expand to another full70k pass only if the controlled three-seed pilot supports
a meaningful grouping improvement. No extra training data or count labels.

Three real-source CPU backward/update checks passed. The actual batch16,
256/128 GPU check also passed; peak Torch reserved memory was3.465GiB.
The runner asserts actual component-history length, not only a configuration
value. Raw preflight manifests/history/completion markers are in `preflights/`.
Defaults64/32 remain unchanged for earlier recipes; no completed run is repeated.

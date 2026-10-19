# SW0233: frozen frequency component diagnosis

Read accepted SW0223 caches and original source checkpoint omega/freq_gain on CPU.
Decompose class-balanced within-object effective-frequency variance into static,
sensory and cross-covariance terms. Validate all three source/prediction protocols
before reading the same fixed16 GT. No model forward, update, mask change or GPU.
This is descriptive evidence; it cannot prove a gating contribution or causal
bottleneck. Covariance can cancel static variance, so variance ratios are not
causal attribution. Original frozen experiments remain unchanged.

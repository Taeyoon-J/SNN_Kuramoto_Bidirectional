# SW0156 native32 signed coupling

This local candidate changes one inference-time interaction: it retains the source model's positive graph and original lag/cost preparation, then uses a signed interaction matrix for phase lag multipliers. The candidate interaction is `A0 + 1e-6 - 0.5 U`; the matched control is `0.5 A0 + 1e-6`, retaining the same native epsilon floor. Here `A0` is the source positive symmetric graph before that floor. The original positive graph plus floor remains the pulse-coupling graph.

`U` is built from the same SW0143 T graph logits: use the source top-k indices per row (including any diagonal selected by that source rule), scatter the checkpoint's learned `exp(log_coupling_gain) / K` on each directed selected entry, then symmetrize once. It does not use the native softmax weights and does not apply absolute value, ReLU, inverse-cost, or row normalization to the signed matrix. The candidate/control continue to compute alpha and cost from positive `A0`.

The first registered screen is one seed1 TRAIN RGB image, native32, 1024 integration steps and 512 settling steps. Before interpreting signed interactions, beta zero must reproduce the SW0143 T source traces, actual-spike Q, and QCC labels bit-for-bit. Candidate/control traces must remain finite; negative interaction mass, runtime, and memory are descriptive. No masks or ground truth are read for this screen.

Only after the source-only screen is reviewed would fixed320 seed1 candidate and control predictions be prepared. All predictions must be frozen and validated before scoring opens masks. The registered expansion gate is at least +0.01 FG-ARI over both same-seed source T and the matched control, with positive paired-image 95% confidence-interval lower bounds for both comparisons. A failure closes this fixed recipe; it does not trigger a sweep.

Peer commit `7d094e8`'s PV2_0092 signed-coupling result remains a pending hypothesis, not a reproduction or result for this experiment. SW0156 is a separate local implementation and has no outcome yet.


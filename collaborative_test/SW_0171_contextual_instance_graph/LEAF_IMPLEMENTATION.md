# SW0171 pure encoder and objective implementation

2026-10-15. Contextual encoder and leave-one-out instance objective are implemented.
Root and the independent reviewer each passed all 12 CPU contracts using the
actual native CNNFeatureEncoder, encode_native32 and graph projection APIs.
The zero-output residual preserves initial feature maps and gamma exactly.
The original base kernels stay frozen; the contextual branch remains trainable
jointly with the graph/core in the registered forthcoming training comparison.
Constructor seeding preserves caller RNG. Hidden residual credit is expected to
appear after the initially zero final layer changes; tests check this behavior.

The auxiliary loss uses the actual graph projection output captured exactly once,
with native normalization, live leave-one-out prototypes and class-balanced CE.
Tests cover gradients, background/instance-label exchangeability, singleton
queries versus negative prototypes, zero-loss images and uniform exclusion counts.
These CPU checks prove implementation behavior, not object segmentation gains.

No SW171 GPU model experiment or optimizer update has run at this stage. The
separate CPU-only original-source TRAIN512 cache producer was observed live as
PID3736173; a launch or lock alone is not proof it completed. It reuses verified
source inputs and will save canonical full modal labels separately from feasible
source-support targets. It does not initialize the failed SW169 model or change
the dataset. Dataset/cache arrays remain on Frontier and are not published.

Preflight source parity, lambda calibration, actual eight-update comparison,
fixed320 performance and all broader gates still require runtime evidence.
SW165 remains the incumbent. Source files and generated outputs retain their
existing separate canonical/centralized locations.

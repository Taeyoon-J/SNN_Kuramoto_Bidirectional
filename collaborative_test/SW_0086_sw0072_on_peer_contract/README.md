# SW0086: SW0072 on the peer contract

This is the requested cross-contract transfer test. The current best SW0072
seed0/1/2 cores are evaluated without retraining on the peer's
`clevr_with_masks` validation rows 6000-6299 and contract-v1 targets. The input
is the peer's aligned 10,000-row gamma tensor; its shape is `[10000,8,256]`, the
target names match the split manifest, and the selected target foreground
fraction is `.117604`.

Every model and classifier setting remains the registered SW0072 setting:
T1024/settle512, membrane threshold `.06`, spike connected components at
threshold `.35`, largest-component background, graph spatial decay `.35`,
geodesic 3/1.5/2/.5/16, factorized Kuramoto, and raw gating. No threshold is
selected on peer labels and no target informs a prediction. Seed0 is the
original SW0055 source core; seeds1/2 are the SW0072 retrained downstream cores.
Their six graph tensors were verified bitwise equal before the run.

The resulting three-seed mean will be compared to the peer contract's published
single-checkpoint Slot reference `.6195/.1241/.0920`. This is a transfer
diagnostic, not a replacement for either branch's fixed-contract headline and
not a matched-training comparison.


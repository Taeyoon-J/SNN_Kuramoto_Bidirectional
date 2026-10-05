# SW0073 - reconstruction-held joint features on the frozen trained graph

SW0072 establishes that a stable trained graph is causal: freezing the seed0
graph lifts the fixed three-seed mean to `.781407/.478038/.616746`. SW0068
showed that joint encoder/core training with the generic binding objective
causes harmful co-adaptation. Peer PV2_0039 now supplies the missing constraint:
phase-slot RGB reconstruction prevents the encoder from making every patch
alike while the PLV losses shape synchrony.

This experiment ports that loss into our aligned HDF5 path while retaining the
exact SW0072 graph. Seed1 first compares reconstruction weights `0.3` and `1.0`
at encoder LR `3e-5`; all other data, core loss, optimizer, epochs, and the
32-image contract remain fixed. The graph must remain bitwise unchanged. An
arm advances only if it exceeds SW0072 seed1 on all three patch metrics.


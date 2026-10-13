# SW0153: native gamma ordering audit

The native SW0135 encoder emits `gamma[B, 8, 1024]`: axis 1 indexes the eight learned feature channels, while axis 2 indexes the 32×32 spatial patches. Those axes have different meanings.

A global channel permutation is an arbitrary feature reparameterization only when every consumer is reindexed consistently. In the registered source core, both `gamma_channel_proj.weight` and the learned graph's `projection.weight` consume the channel basis. Reordering gamma channels and the corresponding columns of both matrices should preserve projected drive, graph adjacency, and downstream dynamics up to floating-point roundoff. Shuffling gamma channels while leaving frozen weights fixed is a different intervention and can change both drive and graph.

A patch permutation instead reassigns image locations among the 1024 nodes. It is not a channel reparameterization: the graph retains its registered spatial/geodesic structure. The probe therefore unpermutes patch outputs back to original image positions before comparing Q and QCC labels.

`audit_source.py` is a read-only, TRAIN-only optional probe. It uses the first `--count` IDs in the registered seed-1 TRAIN order, no masks or GT, and reports actual-Q off-diagonal MSE and QCC label-change fractions for canonical gamma, a fixed random naive channel shuffle, a TRAIN-mean-ordered naive channel shuffle, a consistently reindexed global channel permutation, and per-image patch permutations. It snapshots/restores model tensors and CPU/CUDA RNG state. The default count is four to bound native1024 rollout work; `--count` is limited to 16. It writes one create-once JSON only when explicitly invoked.

This audit does not choose a preferred channel order, alter checkpoints, establish a new model, or claim that any observed Q/mask change improves segmentation. No label ordering is selected using ground truth.

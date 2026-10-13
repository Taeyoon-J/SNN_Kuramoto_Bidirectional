# SW0154 native64 resolution comparison: preparation only

This folder currently contains CPU-only spatial mapping and direct 2x2 modal-pooling helpers. A row-major native16 node maps to its exact 4x4 block of native64 children. Node-leading parameters repeat by that mapping; pairwise parameters use it on both axes. The Slot baseline is pooled from its saved native128 labels directly to64 with the production modal converter, preserving its stored label IDs.

This is not a native64 model or a completed comparison. Sol61's graph, spatial units, top-k, gamma pooling, and physical constants must be frozen before a model/resource probe is added. In particular, no 4096-node tensor allocation or GPU execution is part of this preparation. Slot pixel pooling alone does not establish comparable native model dynamics or scores.

# Source preservation helper preparation

Prepared locally on 2026-10-10 from the conditional [design](../../source_preservation_design.md). This is a loss and provenance-validation library, not a registered experiment or a complete training implementation. No teacher cache, rollout, calibration probe, server deployment or new GPU training has run.

The helper computes float32 off-diagonal actual-Q MSE with a detached teacher. CPU fixtures check exact values, zero identity loss, live student gradients, unchanged inputs, malformed affinities, cache identity and grouping, and the four-batch TRAIN-only calibration rule. Actual checkpoint-to-cache and rollout-to-loss integration remain unverified.

Root independently reproduced 5/5 tests with CUDA hidden, OMP_NUM_THREADS=1, and C:/Users/SangWoo Kim/anaconda3/python.exe; import numpy before unittest loads the test module. The default local Python failed with duplicate OpenMP initialization; no KMP_DUPLICATE_LIB_OK workaround was used. Both implementations remain identical to the tested private originals.

- source_preservation_loss.py SHA256: 28d6fd98bc7be9d57b7a72a8af9d6bb76e063eb88a8ffbf02291e807d1f8ea0a
- test_source_preservation_loss.py SHA256: 12708f91c163120411f2fe1abdd8b6f85d5d59525ab20942c0b6cc673a7ad962

Admission and full implementation depend on the pending SW0139 cross-view and SW0140 causal endpoint results. The incumbent and all running protocols are unchanged.

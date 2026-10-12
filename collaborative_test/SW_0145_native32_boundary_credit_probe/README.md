# SW0145 native32 boundary-Q credit probe

This prospective diagnostic runs one disposable source0 native32 update using the first16 registered TRAIN images. Its target is an exponential relation derived from RGB seam energy between adjacent 4x4-pixel patches. The scale is fixed from all4096 source0 TRAIN images. It then reports boundary-Q and detached source-Q-preservation gradient norms on the same first64 TRAIN images. Before the update, it checks that actual Q on the first16 matches the saved initial source-Q baseline exactly.

The command requires CUDA and writes one create-only JSON: `python -m collaborative_test.SW_0145_native32_boundary_credit_probe.run --device cuda:0 --output collaborative_test/SW_0145_native32_boundary_credit_probe/results_archive/source0_boundary_credit_probe.json`. It saves no model or optimizer state, reads no GT, and always reports `training_admission=false`. A result cannot establish that the relation improves segmentation or justify a training run.

CPU unit tests exercise RGB seam geometry, actual-spike affinity gradients, detached source-Q targets, and registered row-count guards. They do not replace the actual GPU source/resource probe.

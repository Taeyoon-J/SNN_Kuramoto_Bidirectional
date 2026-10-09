# SW0120: Degenerate-trace gradient guard

SW0120 tests a narrow backward-only intervention in the positive four-component spike-affinity path. For centered traces whose norm is at or below the existing `1e-8` clamp, the normalized trace is detached in backward. Forward affinity, phase loss, all loss coefficients, partition RGB, production hard labels, prediction, and evaluation remain unchanged.

The registered seed-0 pilot reuses the SW0097 source core and live encoder, the exact SW0117 data order and optimizer schedule, and the immutable SW0117 lambda. It first requires the actual update-3 replay proof, which established bitwise equality of forward Q, losses, and production labels/H; exact valid-row direct gradients; zero degenerate-row direct gradients; and a finite combined-gradient reduction. The first failed primary-gradient replay is preserved: recomputing the primary objective on a second phase graph showed small encoder autograd repeat variation, so the accepted proof retains the original primary tensor instead of relaxing tolerances.

The actual four-batch GPU preflight passed, and the seed-0 pilot completed all 256 updates with return code 0. The supervisor validated the training artifacts and started the unchanged fixed 320-image evaluation. Seed expansion remains disabled pending seed-0 results review.

Across the completed training, the guarded candidate's preclip gradient norm had median 173.3702, p99 1321.7225, and maximum 3694.9094. The matched SW0117 candidate had median 190.4081, p99 180347181.6, and maximum 446360736. Both histories contain 256 finite gradient norms. This establishes suppression of extreme gradients in this pilot, not improved segmentation or a three-seed result.

The completed candidate core SHA256 is `447b4b8c273fdb049af026ca841fbe7256d9079b10239e8e14758838575a29e5`; its encoder SHA256 is `34f6319025c8366cb102eabb8df94abff5a54222122ebcbcd5e291d161fffdc2`. The current state and comparison anchors are recorded in `../best_model_tracking_20261008.json`.

Run the local synthetic suites from the repository root with `python -m unittest collaborative_test.SW_0120_degenerate_trace_gradient_guard.test_affinity collaborative_test.SW_0120_degenerate_trace_gradient_guard.test_run -v`.

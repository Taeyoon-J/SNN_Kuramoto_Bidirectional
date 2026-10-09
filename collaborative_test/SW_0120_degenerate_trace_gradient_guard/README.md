# SW0120: Degenerate-trace gradient guard

SW0120 tests a narrow backward-only intervention in the positive four-component spike-affinity path. For centered traces whose norm is at or below the existing `1e-8` clamp, the normalized trace is detached in backward. Forward affinity, phase loss, all loss coefficients, partition RGB, production hard labels, prediction, and evaluation remain unchanged.

The registered seed-0 pilot reuses the SW0097 source core and live encoder, the exact SW0117 data order and optimizer schedule, and the immutable SW0117 lambda. It first requires the actual update-3 replay proof, which established bitwise equality of forward Q, losses, and production labels/H; exact valid-row direct gradients; zero degenerate-row direct gradients; and a finite combined-gradient reduction. The first failed primary-gradient replay is preserved: recomputing the primary objective on a second phase graph showed small encoder autograd repeat variation, so the accepted proof retains the original primary tensor instead of relaxing tolerances.

The current status means only that the local update-3 proof is available and passes its contract. It is not evidence that the 256-update pilot passed preflight or completed. Seed expansion is disabled pending seed-0 results review.

Run the local synthetic suites from the repository root with `python -m unittest collaborative_test.SW_0120_degenerate_trace_gradient_guard.test_affinity collaborative_test.SW_0120_degenerate_trace_gradient_guard.test_run -v`.

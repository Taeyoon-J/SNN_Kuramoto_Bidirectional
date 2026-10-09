# SW0120: Degenerate trace gradient guard

This folder contains a standalone spike-affinity helper and synthetic tests. Its forward computation matches the existing positive four-component spike-affinity calculation. The only intended difference is backward behavior for centered component traces whose norm is at or below the existing `1e-8` clamp: their normalized trace is detached, preventing the clamp-scale derivative from propagating through a degenerate norm.

No model, classifier, loss, readout, density target, or training recipe is changed here. SW0117's source, objective, and frozen lambda remain unchanged. Passing these synthetic tests does not establish model-level equivalence or authorize training; the actual update-3 guarded-Q/L/H equivalence and gradient-drop checks remain required first.

Run the focused suite from the repository root with `python -m unittest collaborative_test.SW_0120_degenerate_trace_gradient_guard.test_affinity -v`.

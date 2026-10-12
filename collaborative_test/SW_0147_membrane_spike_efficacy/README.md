# Conditional membrane-to-spike efficacy candidate

Status: local CPU leaf verification only; no server deployment, calibration, GPU training, or segmentation score. Admission depends on completed SW0148 results under the unchanged SW0146 recipe, as specified in [the conditional design review](../SW_0146_native32_boundary_q_pilot/conditional_membrane_efficacy_review.md).

The candidate adds four zero-initialized bounded coefficients to actual native event-times-gate spikes. A membrane-margin term modulates spike efficacy between 0.5 and 1.5 times the native amplitude. The emitted spike remains the value used by recurrence/reset and the final trace. At zero initialization, the tested native rollout is reproduced exactly. This is not a replacement for the event rule or evidence that every gate is necessary.

Six CPU tests passed with CUDA hidden and NumPy imported before Torch to avoid the local duplicate OpenMP runtime failure. Evidence and source hashes are in `results_archive/local_cpu_20261012/`. They verify the 64-node/64-step synthetic rollout, bounded amplitudes, recurrence/trace consistency, invalid-input rejection, first-step coefficient credit, and second-step added membrane credit. The live and detached-credit variants have the same forward values from identical inputs and coefficients; the isolated added-margin gradient appears only in the live variant. The rollout also checks a nonzero membrane-parameter gradient difference while retaining native surrogate credit in both variants.

These tests establish feasibility for the fixture. They do not establish actual native32 TRAIN calibration, production source-Q parity, dataset-scale preflight, better FG-ARI, data scaling, or gating necessity. All of those require separate empirical evidence. No training runner or queued GPU experiment is registered for this candidate yet.

To reproduce the local suite from the repository root with CUDA hidden and one OpenMP/MKL thread:

```sh
python -c "import numpy; import unittest; suite=unittest.defaultTestLoader.loadTestsFromName('collaborative_test.SW_0147_membrane_spike_efficacy.test_efficacy'); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(0 if result.wasSuccessful() else 1)"
```

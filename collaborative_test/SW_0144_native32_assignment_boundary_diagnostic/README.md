# SW0144 native32 assignment/boundary diagnostic

This small CPU diagnostic reads the completed SW0142 seed1 masked-spike cache and its warmed binder, then examines the first64 registered TRAIN rows. It reports the mass-weighted fraction of predicted nonbackground P labels split across multiple 4-connected components, and compares actual-spike Q affinity on visible neighboring patches with RGB seam energy and the descriptive target `exp(-e/s)`.

The scale `s` is the mean seam energy over these same64 rows. It is descriptive, not a training calibration. The cache contains float16 spikes generated from masked RGB, so this does not test clean-input parity, gradients, causal direction, GT quality, or a candidate objective. The program performs no training, prediction, GT read, or artifact overwrite.

Run from the repository root with `python -m collaborative_test.SW_0144_native32_assignment_boundary_diagnostic.run --output collaborative_test/SW_0144_native32_assignment_boundary_diagnostic/results_archive/first64_assignment_boundary.json`. The output is create-only. CPU tests: `python -m unittest collaborative_test.SW_0144_native32_assignment_boundary_diagnostic.test_run -v` with CUDA hidden before importing Torch.

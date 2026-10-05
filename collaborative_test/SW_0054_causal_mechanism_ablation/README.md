# SW0054 - separated carrier/mask causal ablation

This validation-only diagnostic uses the aligned seed0 low-LR epoch25 checkpoint from SW0053:
`trained_models/SW_0050_sample_diversity_s0_w0_lr0p0003_epoch25/checkpoints/epoch_25.pt`.
Every condition reruns from reset dendrite/membrane state on identical gamma and fixed readouts.

Under the checkpoint's asserted `gate_mode=raw`, the dendritic drive is `sin(current theta) × delayed raw mask`, while `g_wave` is the delayed mask. Thus the former combined-gate intervention confounded carrier and mask changes. The revised conditions separate them:

* `gate_perm_s0/s1/s2`: local carrier × permuted delayed mask; the membrane/spike gate uses the same permuted mask.
* `carrier_perm_s0/s1/s2`: permuted carrier × local dynamic mask; membrane gate stays local.
* `gate_mean`: local carrier × per-image/per-region temporal-mean mask; membrane gate uses that mean.
* `carrier_mean`: per-image/per-region temporal-mean carrier × local dynamic mask; membrane gate stays local.
* `K0`: exactly zero inter-region Kuramoto stiffness; gamma, frequency parameters, and theta initialization remain unchanged.

For every gate/carrier arm, graph output must be bitwise equal and theta allclose to normal. The comparison refuses pulse feedback, graph feedback, nonzero theta-init noise, or non-raw gate mode. Readouts include phase, mask, carrier, h-wave, membrane, and gated-spike distance-controlled AUC; fixed spike-CC (.35) and membrane×Gaussian spectral (sigma1.5/k10) metrics; predicted object count and foreground fraction; spike event rate; membrane temporal variance and absolute activity. Target masks are used only after prediction for AUC and scores.

The four-image short preflight is bound to checkpoint, gamma, manifest, HDF5 metadata, and relevant code hashes. Results refuse overwrite and occupied GPUs. Stage-1 selection can be kept small before a chosen candidate receives the full long run:

```bash
bash collaborative_test/SW_0054_causal_mechanism_ablation/preflight.sh GPU_ID
bash collaborative_test/SW_0054_causal_mechanism_ablation/evaluate.sh GPU_ID short 32 pilot
bash collaborative_test/SW_0054_causal_mechanism_ablation/evaluate.sh GPU_ID long 320 full
```

All six tensor unit tests and a one-image/full-condition CPU smoke test pass
against the real checkpoint, gamma, manifest, and HDF5 assets. The smoke test
is a code-validity check, not validation evidence. GPU preflight and the
32-image pilot remain pending.

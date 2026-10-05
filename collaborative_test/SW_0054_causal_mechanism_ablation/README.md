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

After the count-32 pilot JSON is written, validate and summarize changes versus
`normal` with:

```bash
python collaborative_test/SW_0054_causal_mechanism_ablation/summarize.py \
  trained_models/SW_0054_causal_mechanism_ablation/seed0_epoch25_short_n32_pilot_T256_settle64.json
```

The tool checks checkpoint SHA/path, IDs/count/window/condition set, finite
metrics, and graph/theta invariants. It writes JSON and Markdown beside the
input and refuses to overwrite either unless `--overwrite` is passed. Deltas
and activity ratios are descriptive; it deliberately makes no claim that an
effect is explained by activity-scale collapse.

All six tensor unit tests and the real-asset preflight passed. The 32-image
seed0 pilot also completed and passed the current report validator. On that
small pilot, normal spike-CC scored `.6685/.6920/.4899`; local-carrier with a
permuted gate scored `.0055/.0880/.0278` while spike event-rate and membrane
variance ratios stayed near 1 (`1.0005` and `.9990`). Permuting the carrier
instead preserved most of the score (`.6637/.6798/.4823`), while setting
Kuramoto coupling to zero scored `.2377/.4388/.2212` with activity scale
largely preserved. This supports gate identity and inter-region coupling as
causal factors in this pilot and suggests carrier permutation is largely
redundant under this condition. These are 32-image, seed0 pilot results, not a
full-validation or general mechanism claim. Raw and validated summary JSON
are under `results/seed0_epoch25_short_n32_pilot*`.

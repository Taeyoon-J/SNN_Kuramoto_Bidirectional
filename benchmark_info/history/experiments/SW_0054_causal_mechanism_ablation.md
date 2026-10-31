# SW_0054_causal_mechanism_ablation

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `5fa130c458419d6d5c33eae4a61de5bfc8ba3901b0e6ccb023fd8150990ad07c`

````markdown
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

````

</details>

## results/seed0_epoch25_short_n32_pilot_summary.md

<details>
<summary>seed0_epoch25_short_n32_pilot_summary.md</summary>

Source SHA256: `8a77f39ae436105e45d27f79134661d037d36558ea60a645ab64aa3b968e53eb`

````markdown
# SW0054 count-32 pilot summary

Checkpoint SHA-256: `d30147fbd460a702689cb59fea13dd4d36d70b4a7b460ffd020382b2334e8c85`
Protocol: IDs 1320-1351, T256/settle64, pilot condition set, GT-free prediction.

Deltas are intervention minus normal. Activity values are ratios to normal.

| Condition | Readout | FG-ARI delta | FG IoU delta | Matched IoU delta | Spike-rate ratio | Membrane-variance ratio | Membrane-abs ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| gate_perm_s0 | spike_cc | -0.674001 | -0.603998 | -0.462112 | 1.0005366608234825 | 0.9989897642671285 | 1.0005336768039321 |
| gate_perm_s0 | membrane_spatial | -0.177434 | 0.007768 | -0.152926 | 1.0005366608234825 | 0.9989897642671285 | 1.0005336768039321 |
| carrier_perm_s0 | spike_cc | -0.004832 | -0.012223 | -0.007555 | 1.0005622390821556 | 1.0006399194047328 | 1.0006340722493436 |
| carrier_perm_s0 | membrane_spatial | -0.064457 | 0.006144 | -0.061322 | 1.0005622390821556 | 1.0006399194047328 | 1.0006340722493436 |
| gate_mean | spike_cc | -0.666685 | -0.563001 | -0.477229 | 1.011894372891679 | 0.30244141120499757 | 0.9809648732241784 |
| gate_mean | membrane_spatial | -0.128889 | 0.003613 | -0.084994 | 1.011894372891679 | 0.30244141120499757 | 0.9809648732241784 |
| carrier_mean | spike_cc | 0.005710 | -0.001696 | -0.006463 | 1.0120255217934606 | 0.5076867052549898 | 1.0145205274168942 |
| carrier_mean | membrane_spatial | -0.008959 | 0.003037 | 0.014262 | 1.0120255217934606 | 0.5076867052549898 | 1.0145205274168942 |
| K0 | spike_cc | -0.430772 | -0.253211 | -0.268714 | 1.0007519042832604 | 0.9781249807030264 | 0.9929789624148877 |
| K0 | membrane_spatial | -0.103416 | 0.005164 | -0.124597 | 1.0007519042832604 | 0.9781249807030264 | 0.9929789624148877 |

## Distance-controlled AUC delta

| Condition | Phase | Gate | Carrier | H-wave | Membrane | Spike |
|---|---:|---:|---:|---:|---:|---:|
| gate_perm_s0 | 0.000000 | -0.375544 | 0.000000 | -0.223031 | -0.213625 | -0.337514 |
| carrier_perm_s0 | 0.000000 | 0.000000 | -0.380815 | -0.082187 | -0.080955 | -0.009273 |
| gate_mean | 0.000000 | -0.391347 | 0.000000 | 0.002204 | -0.003760 | -0.372264 |
| carrier_mean | 0.000000 | 0.000000 | -0.386366 | -0.007758 | -0.007699 | 0.019083 |
| K0 | -0.076230 | -0.179480 | -0.269359 | -0.228041 | -0.227170 | -0.163979 |

Numerical deltas only; this summary does not infer whether an effect is explained by activity-scale collapse.

````

</details>

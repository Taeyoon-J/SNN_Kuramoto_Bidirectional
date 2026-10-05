# SW0054 - frozen-core causal mechanism ablation

This validation-only diagnostic uses the aligned seed0 low-LR epoch25 checkpoint from SW0053:
`trained_models/SW_0050_sample_diversity_s0_w0_lr0p0003_epoch25/checkpoints/epoch_25.pt`.
Every condition is rerun from a reset recurrent state on the same gamma batch and uses the same fixed readouts.

Conditions: normal; three fixed region permutations of both the dendritic `gamma_wave` and membrane/spike `g_wave` gates, preserving each gate value's time trajectory; per-image/per-region temporal means applied to both gate sites; and Kuramoto inter-region stiffness `K=0` with gamma, frequency parameters, and theta initialization untouched. The evaluator requires pulse feedback and graph feedback to be off, checks graph outputs bitwise and theta allclose for every gate condition, and asserts K is zero during its rollout.

For phase, gate, h-wave, membrane, and gated spike, results include distance-controlled foreground-pair AUC by squared patch distance and macro AUC. Fixed readouts report spike-synchrony CC at threshold .35 and membrane×Gaussian spectral (`sigma=1.5`, `k=10`) metrics, predicted object count, and foreground fraction. Targets are read only for post-prediction AUC/scoring.

Before full scoring, a four-image preflight runs the same causal matrix at T256/settle64. Its marker binds checkpoint, aligned gamma, manifest, HDF5 file metadata, and relevant code hashes; the evaluator refuses occupied GPUs and existing result/log outputs.

```bash
bash collaborative_test/SW_0054_causal_mechanism_ablation/preflight.sh GPU_ID
bash collaborative_test/SW_0054_causal_mechanism_ablation/evaluate.sh GPU_ID short
bash collaborative_test/SW_0054_causal_mechanism_ablation/evaluate.sh GPU_ID long
```

Static checks, four tensor unit tests, and a one-image CPU smoke test against the real checkpoint, gamma, manifest, and HDF5 assets passed. The smoke test exercised all six conditions and both readouts; it is a code-validity check rather than evidence about validation performance. The four-image GPU preflight and full evaluation remain pending.

# SW0059 - centered delayed-gate validation pilot

SW0059 tests an opt-in transduction change on the existing SW0053 seed0
low-LR epoch25 checkpoint. It does not train or alter checkpoint tensors.

The legacy `raw` path is unchanged: dendritic drive is
`sin(theta_current) * mask_delayed`, while the membrane receives the raw
delayed mask. New `centered_raw` mode uses
`2*mask_delayed - 1 = sin(mean(theta_delayed))` as the dendritic drive, repeated
over the oscillator-component axis, and still sends the raw, uncentered
`mask_delayed` to the membrane. This specifically removes the nonnegative
gate's DC offset from the drive while preserving the delayed gate identity and
the existing membrane gate semantics. It leaves state-dict names, shapes,
initialization, and strict loading unchanged.

The fixed pilot is HDF5-aligned IDs 1320-1351, T256/settle64, shared
dendritic projection, factorized dynamics, graph spatial decay .35, geodesic
steps3/radius1.5/contrast2/temperature.5/cap16, vth .06, and spike-CC threshold
.35. The raw reference already exists in the validated SW0054 pilot summary:
normal spike-CC `.6685/.6920/.4899`. The centered evaluation records the same
three metrics. This is a 32-image validation pilot used to decide whether a
training experiment is warranted; it is not a full-split claim.

GPU0/1 are accepted only. The script refuses existing/partial result, log, or
validated-report files, checks GPU occupancy, and evaluates only the existing
checkpoint. Static checks, five focused unit tests, strict state-dict loading,
and a four-image real checkpoint/gamma/HDF5 smoke test passed before launch.

```bash
# Optional same-evaluator raw reference (the SW0054 summary may be used instead)
bash collaborative_test/SW_0059_centered_delayed_gate/evaluate.sh 0 raw
# Centered delayed-gate validation pilot
bash collaborative_test/SW_0059_centered_delayed_gate/evaluate.sh 0 centered_raw
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0059_centered_delayed_gate/compare.py \
  --centered-report /Data0/kevinswk/patch_v2_sw/trained_models/SW_0059_centered_delayed_gate/seed0_centered_raw_short_n32_T256_settle64_validated.json \
  --sw0054-summary collaborative_test/SW_0054_causal_mechanism_ablation/results/seed0_epoch25_short_n32_pilot_summary.json \
  --output collaborative_test/SW_0059_centered_delayed_gate/results/seed0_comparison.json
```

The raw command is optional because SW0054's raw normal result uses the same
checkpoint, subset, short window, and spike-CC threshold.

The 32-image centered result is `.677768/.674981/.511379`, versus the raw
reference `.668520/.692026/.489879` for FG-ARI/foreground IoU/matched-object
IoU. The deltas are `+.009248/-.017044/+.021500`: object separation and matched
object overlap improved, but foreground coverage declined. Because it did not
improve all three registered metrics, the exact centered mode is not promoted
directly to a training run. Full reports and the validated comparison are in
`results/`.

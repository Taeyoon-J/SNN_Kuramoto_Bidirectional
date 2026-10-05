# SW0060 - signed delayed-mask carrier pilot

SW0060 follows the mixed SW0059 result. SW0059 removed the carrier as well as
the mask DC component; FG-ARI and matched-object IoU improved, while foreground
IoU declined. This pilot isolates that ambiguity by preserving the current
per-component carrier and centering only its delayed modulation:
`sin(theta_current) * (2*mask_delayed-1)`. The membrane keeps the original raw
delayed mask.

The checkpoint, gamma, HDF5 data, IDs1320-1351, T256/settle64, dynamics,
spike-CC readout, and threshold .35 exactly match the SW0054 raw reference and
SW0059 pilot. This is validation-only; it changes no training data or weights.
The candidate is promoted only if all three fixed metrics improve over raw.

```bash
bash collaborative_test/SW_0060_signed_mask_gate/evaluate.sh 0
```

The pilot completed at `.667959/.688642/.486288` for
FG-ARI/foreground IoU/matched-object IoU. Relative to raw, the deltas were
`-.000561/-.003384/-.003590`; all three were lower. Centering only the delayed
mask while preserving the carrier is therefore rejected under this pilot.
Validated reports are in `results/`.

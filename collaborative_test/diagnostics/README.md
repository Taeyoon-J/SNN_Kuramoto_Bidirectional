# Actual spike / gate activity diagnostic

`spike_gate_activity.summarize` accepts actual `[B,C,N,T]` emitted spikes and
their actual gates. It needs NumPy only and does not call a model, read data,
change weights, change masks, or allocate a GPU. It is not yet attached to the
running immutable 235/236 source and makes no new native64 measurement claim.

SW0155/157 found saturated binary events and weak membrane credit in an older
native32 lineage. SW0158 membrane-amplitude credit and SW0159 event-regime
changes also did not establish the desired performance improvement. Do not
generalize that observation to the current native64 model or repeat those
recipes without checking actual current traces first.

For the native emission rule `S=event*gate`, the helper reports event occupancy
and transitions only where the gates expose those events. It identifies exact
spike/gate equality and component-shared gates; it does not claim that equality
proves membrane removal would preserve performance. At gate zero, the binary
event cannot be reconstructed. Missing denominators produce `None`, not a
fabricated zero rate. Tiny positive gates cannot turn a suppressed event into
an inferred positive event through an absolute tolerance.

Future use should record checkpoint/source identity, TRAIN/DEV scope, actual
settle window, and per-image outputs before aggregation. A counterfactual
gate/membrane ablation remains necessary for a causal performance claim.
No synthetic fixture below is a model result.

```powershell
python -B -m unittest collaborative_test.diagnostics.test_spike_gate_activity
```

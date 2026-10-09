# SW0124 temporal prototype readout

This is a read-only stage-1 evaluation of a deterministic temporal-prototype
readout over frozen SW0097 source spike traces. It does not train or modify a
checkpoint. The protocol and exact operation order are in `protocol.json` and
`design.md`; the implementation is in `prototype_readout.py` and `evaluate.py`.

The readout centers and applies the registered SW0123 legacy four-channel
TRAIN RMS to actual settled component-spike histories. Detached production
positive-product affinity selects up to eleven distinct trace anchors. Three
soft assignment/prototype updates use negative mean squared temporal distance,
followed by a final assignment. Existing SW0123 largest-slot-background and
minimum-two-patch labeling stays fixed. Production QCC is reported as the
comparison readout.

The evaluator writes all three seeds' GT-free predictions and verifies their
hashes before opening validation masks. It then requires exact per-image
SW0097 source-QCC reproduction within `1e-10` before calculating any temporal
prototype scores. It uses the fixed 320-image validation contract and shared
image indices for the registered three-seed bootstrap. Existing outputs are
never overwritten; provide a new `--output-dir` for a new attempt.

The create-once real-data preflight is one B8/T1024 batch per seed and writes
only a small provenance record. Evaluation requires all three matching
preflights. `coordinator.py` defines those three preflight tasks followed by one
three-seed evaluation; `dispatcher.py` holds the shared per-GPU flock across
each child, admits only a GPU with no compute owner and at most 512 MiB used,
monitors owned process descendants, and never retries a failed task.

Focused CPU checks:

```powershell
python -m unittest collaborative_test.SW_0124_temporal_prototype_readout.test_prototype_readout collaborative_test.SW_0124_temporal_prototype_readout.test_evaluate collaborative_test.SW_0124_temporal_prototype_readout.test_dispatcher -v
```

The suite validates variable temporal lengths, deterministic anchors,
identical-trace K=1 behavior, live spike gradients, source-QCC baseline
rejection, and the shared-image bootstrap contract. It does not substitute for
the registered source/GPU evaluation.

## Actual execution (2026-10-09)

Registered commit3d4b462 was deployed as10 SHA-verified files. Local and Frontier14 CPU tests passed; all3 immutable source/evaluation/calibration bindings passed beforelaunch. Supervisor1491982 is live; GPU0 source seed0 actual B8/T1024 preflight passed and seed1 was observed running. All3 preflights precede full320 three-seed comparison. No optimizer updates or checkpoint edits. Result pending; no gating/scaling/promotion claim. Peer branch697eeb6 unchanged thiscycle. See server_launch_evidence_20261009.json.

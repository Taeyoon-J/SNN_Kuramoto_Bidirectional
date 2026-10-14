# SW0157 TRAIN gate and event-credit diagnostic

SW0157 is a read-only, eight-image TRAIN diagnostic. It compares source97, the same source with the SW0143-T graph operator installed for inference, and the completed SW0155 `encoder_graph` and `joint` checkpoints. The same ordered source TRAIN rows are used for all four models: positions 0–3 and 2048–2051 of the immutable 4096-image TRAIN order. Each four-image batch uses the real aligned HDF5 TRAIN masks and production modal32 labels.

Each model runs the native32 source at T=1024, settle=512, with a 64-step live tail. No optimizer is created or stepped. For each component, the report separately summarizes full-window and settled-window (steps 512–1023) membrane margins relative to the existing threshold `0.06*exp(b)`, inferred binary event occupancy, gate-positive event occupancy, transitions, emitted-spike identity `S=E*g`, and the shared-gate invariant. It measures gradients from the unchanged SW0155 balanced relation objective by parameter family. A second backward pass substitutes component membrane for component spikes; this is labeled a diagnostic proxy and is not a proposed loss or a performance score.

The report is written only after all provenance checks, finite checks, and before/after model-state checks pass. It records source/checkpoint/manifest/assets/HDF5 identities, selected IDs and positions, RNG seeds, code fingerprints, and zero optimizer updates. A 300-second GPU-side watchdog bounds the diagnostic. Incomplete work is preserved in the queue log/state and does not produce a valid completion report.

Run focused tests from the repository's `patch_v2_sw` directory:

```powershell
$env:PYTHONPATH='.'
python -m unittest collaborative_test.SW_0157_train_gate_credit.test_diagnostic collaborative_test.SW_0157_train_gate_credit.test_queue
```

Inspect the planned reservation without acquiring a GPU:

```powershell
python -m collaborative_test.SW_0157_train_gate_credit.diagnostic_queue --dry-run
```

The queue waits for both SW0155 training artifacts and the SW0143-T reference, then reserves one exclusive idle GPU for the diagnostic. It does not launch training or read validation/test masks.

After revalidating upstream files while holding the shared GPU lease, the queue checks the GPU UUID, active compute owners, and memory use once more immediately before launching the diagnostic. A failed or changed check releases the lease and returns to waiting.

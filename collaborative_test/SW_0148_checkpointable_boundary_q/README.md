# SW0148 checkpointable SW0146 continuation

SW0148 reruns the unchanged SW0146 seed-1 `encoder_graph` and `joint` arms with a durable checkpoint after each completed logical update. The model, source data, objective, optimizer, update budget, native32 prediction path, and GT-last score gate remain bound to SW0146. This is an operational continuation after foreign-owner interruptions; it is not a new scientific result until both arms and the fixed evaluation finish.

Each checkpoint is one atomically replaced, self-verifying `boundary.ptx` container. It records model and encoder state, complete Adam state, history, source/admission/preflight/data hashes, RNG state, and runtime signature. The commit occurs only after four B4 microbatches, gradient clipping, Adam update, joint scalar projection, and history append. Interrupted partial updates are recomputed from the preceding boundary. Only a queue-observed foreign owner permits retry: training resumes a valid checkpoint or starts fresh before the first commit when only recognized uncommitted temporary files exist; prediction restarts only if its final output is absent. Other failures stop the task. The queue uses shared GPU leases, requires a stable idle window, never kills foreign processes, and hides CUDA from the final CPU scoring stage.

The pipeline is:

1. Train or resume both fixed arms from the already validated SW0146 admission and preflights.
2. Predict each arm on the same ordered 320 validation images without reading targets.
3. Validate and freeze both candidate predictions and the existing source97 and Slot32 references.
4. Open the fixed target masks and compute the registered metrics and paired bootstrap gate.

Run CPU contract tests from the repository root with CUDA hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest collaborative_test.SW_0148_checkpointable_boundary_q.test_checkpoint -v
python -m unittest collaborative_test.SW_0148_checkpointable_boundary_q.test_queue -v
python -m collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue --dry-run
```

The resume test proves exact continuation for its deterministic CPU fixture. It does not claim bitwise equivalence across GPU models or software versions. SW0148 does not resume a damaged or mismatched checkpoint and does not overwrite earlier SW0146 artifacts.

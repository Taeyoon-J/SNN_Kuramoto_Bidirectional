# SW_0148_checkpointable_boundary_q

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## queue_v2_review.md

<details>
<summary>queue_v2_review.md</summary>

Source SHA256: `e857637d60e3d2aa8e39557c922250f428498d6d3299d728debf7effce0e4be1`

````markdown
# Checkpoint GPU admission correction

The original checkpoint runtime signature includes the physical `CUDA_VISIBLE_DEVICES` string. The original supervisor could select another idle GPU after foreign-owner interruption, leading to a terminal runtime mismatch even when both GPUs have the same model. This correction preserves that strict signature and chooses only the original physical GPU when a valid committed checkpoint exists. Before the first commit, any of the four exclusive idle GPUs remains eligible. Predictions can use any exclusive idle GPU.

`stage_queue_v2.py` is a separate operational supervisor with its own runtime directory. The original eight sealed SW0148 files, scientific recipe, checkpoints, and SW0146 lineage remain unchanged. Corrupt checkpoints, invalid CUDA metadata, and invalid committed update counts fail before GPU reservation. A busy checkpoint-bound GPU causes a wait rather than migration.

All 19 Linux CPU tests passed without skips: the original ten checkpoint/pipeline tests and nine v2 admission tests. The new tests exercise the actual GPU reservation loop with simulated occupancy, pinning, cancellation, affinity forwarding, malformed checkpoint rejection, and completed-result reuse. These checks do not prove production CUDA resume or segmentation improvement; those remain pending actual training.

The first supervisor was frozen and identity-checked before replacement. It had no training children. Its original state and switch evidence are archived under `results_archive/server_start_v2_20261012/`. The replacement supervisor is PID543798. Both training tasks were queued at launch; no new segmentation score is claimed.

Use the v2 entry point for this run:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v2 --dry-run
```

Do not launch another supervisor while the recorded process is live. Observe the existing `results_archive/queue_v2_20261012/state.json` and actual process first.

````

</details>

## queue_v3_review.md

<details>
<summary>queue_v3_review.md</summary>

Source SHA256: `7b135564b0a9142af5bde705de943f0317fb338480f6e2c4e2fb4b9fa40b3bba`

````markdown
# Faster exclusive GPU admission

The live v2 queue observed GPU2 idle at 4 MiB, but a foreign workload entered before the 60-second admission window completed. That window was an operational conservative policy, not a scientific requirement or protection against users who do not honor our leases.

V3 changes only the queue docstring, runtime namespace, queue version, and idle window from 60 to 5 seconds. It still requires two idle observations and a final recheck, keeps the exclusive inherited lease, checks foreign owners while running, and pins checkpoint resumes to the same physical GPU. Model, encoder, graph, loss, training budget, source bindings, runtime signature, and evaluation are unchanged. A shorter window increases launch opportunities and may also increase foreign collisions; only our child is stopped on collision and a valid committed boundary is preserved.

The eleven Linux CPU tests passed without skips. They cover the existing v2 admission cases plus the actual stable-idle timing loop and rejection of a newly observed owner. V2's waiting supervisor was identity-checked and replaced before any training child started. V3 supervisor PID551842 was verified live. Evidence is under `results_archive/server_start_v3_20261012/`. Neither a native CUDA checkpoint-resume result nor a new segmentation score exists at launch.

The current entry point is `python -m collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v3`; observe its existing process/state before any launch. Earlier supervisors and their artifacts remain preserved.

````

</details>

## queue_v4_review.md

<details>
<summary>queue_v4_review.md</summary>

Source SHA256: `6258122671673e0c8bb4e974f9c74bb530d5b4468b0788906fc17a4a63c25514`

````markdown
SW0148 v4 recovery queue

The reviewed queue preserves the original five-stage training/prediction/scoring DAG and shared inherited GPU leases. Train commands use the sealed parity adapter; prediction and score commands remain original. Only the failed joint arm may start immediately. The original encoder_graph supervisor, child identity and process groups must finish before v4 inspects/adopts/resumes that arm. Permission failures do not mean missing processes; live predecessor helpers are waited on. No old process is signaled.

The adapter's six regression tests and the queue's seventeen focused tests passed on Frontier CPU with zero skips. Both original model/loss/data and original eight source files remain unchanged. Launch receipt verifies supervisor 779598 and records the old encoder still active. At first observation joint child 779656 was live; actual intermediate-update CUDA resume and scientific evaluation remained unverified.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `07eb1697fc6242746807b1c3f52e44c67de94894f1a586eba60b5da3a6a23264`

````markdown
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

````

</details>

## recovery_v2_review.md

<details>
<summary>recovery_v2_review.md</summary>

Source SHA256: `195af8d40e0944d925eaaeb04b7fb21d8dd3e33603d8c6f7fa26936455bac4eb`

````markdown
SW0148 parity resume correction

The original resume comparison removed initial_source_Q_parity from saved bindings but retained its None placeholder in fresh bindings. recovery_v2 installs a process-local wrapper that removes only that unset fresh marker, then invokes the unchanged trainer. The original trainer reattaches saved exact parity and enforces full checkpoint bindings, runtime, optimizer and RNG checks. All eight original files retain their registered hashes. No model, loss, data, update budget or decoder changes.

Create-only, file-and-directory-fsynced attempt receipts bind the adapter, helper, original source hashes, incoming checkpoint, completed update, source lineage and runtime. Six Linux CPU regression tests passed with zero skips, including the original train_arm resume branch with expensive setup/load mocked. This proves the comparison correction; actual intermediate-update CUDA resume remains unverified. The live encoder_graph child is preserved. Recovery dispatch and final scientific scores remain pending.

````

</details>

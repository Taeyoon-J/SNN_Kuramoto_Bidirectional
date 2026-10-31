# SW_0134_native_spike_binding

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## followup_evidence_requirements.md

<details>
<summary>followup_evidence_requirements.md</summary>

Source SHA256: `78f7aa0ea5b48d4acd73d3a75c45e5eea5e45a59e15fdf7bc45444c811efdba5`

````markdown
# Evidence required after the SW0134 pilot

This document records prospective follow-up requirements. It does not change
the running seed1 pilot, its 4096-update endpoint, its three arms, or its
registered FG-ARI expansion test. These follow-ups are not running or queued.
The pilot can establish downstream spike utility and joint-learning utility;
it cannot establish every individual gate's contribution or data scaling.

## Decision after the fixed pilot endpoint

Freeze all four complete 320-image predictions before opening masks. Validate
the checkpoint, optimizer steps, actual training orders and prediction hashes.
Apply the original paired FG-ARI expansion test against source97, gate_joint
and actual_frozen. Report all three metrics even when expansion fails.

If expansion passes, repeat the unchanged recipe on seeds0/2 before promoting
it. A seed1 gain alone cannot replace the three-seed incumbent. If it fails,
use the already registered TRAIN-only assignment/RGB diagnostic to identify
collapse or representation failure; do not rescue the result with the
secondary QCC readout or a new evaluation contract.

## Individual contributions need controlled retraining

Inference-only removal is a fragility diagnostic, not sufficient proof that
each mechanism improves a model trained with an alternative. Train matched
controls from identical seed-specific starting states, with the same image
orders, updates, binder, decoder and evaluation endpoint. Register each exact
intervention and its initialization/gradient checks before launching it.

| Mechanism | Controlled question | Interpretation limit |
|---|---|---|
| Kuramoto coupling | Compare the full model with coupling removed while preserving the uncoupled phase evolution and downstream modules. | Tests coupling, not the existence of any rhythmic input. |
| Sinusoidal modulation | Replace only the selected delayed modulation by a TRAIN-derived constant; retain the phase carrier and the remaining dynamics. | Name every altered carrier/hold/emission path; a blanket constant replacement confounds them. |
| Dendritic temporal integration | Remove the previous dendritic state contribution while retaining the same learned projection and branch reduction. | Tests temporal state, not the entire dendritic computation. |
| Membrane retention | Remove previous membrane retention while keeping the threshold, event function and emission modulation. | Preserve and report any native hold predicate; do not silently remove two mechanisms. |
| Thresholded event | Compare native threshold events with a separately specified continuous-output control under the same downstream budget. | This changes signal amplitude/distribution; report activity and a TRAIN-only amplitude-matched diagnostic. |

Each control needs a code-level equation, not just a name such as "gate off".
Log activation activity, gradient connectivity and assignment occupancy on
fixed TRAIN images. A nonzero gradient alone is not usefulness evidence.
Evaluate complete models on the fixed patch contract, reporting per-seed
scores and paired image differences. Three-seed replication is required for
the paper's main contribution claims. A control that matches or improves the
full model leaves that mechanism's claimed benefit unproved and directs the
next architecture/loss change.

## Data scaling must start before exposure to the large pool

Do not reuse source97 as a clean small-data initialization: it has already
seen70k. Reconstruct one reproducible clean initialization per seed and clone
its exact weights, buffers and optimizer state across nested2500/10k/70k
conditions. Disclose any encoder pretraining. Jointly train encoder, graph,
core, integration and binder/decoder; keep the selected recipe unchanged.
Calibrate any TRAIN-derived constants on the common small prefix, never on
the large pool before the small-data arm is trained.

Use two distinct comparisons:

- Equal compute:4375 batch16 updates,70000 image presentations per condition.
  This measures diversity at fixed compute, with28/7/1 passes respectively.
- Equal repetition:ten complete passes, retaining the final partial batch.
  This gives1570/6250/43750 updates and25000/100000/700000 presentations.
  The per-image loss must be normalized for the partial batch. Ten passes
  are a registered budget, not a claim of convergence.

Use seeds0/1/2, the same fixed320 evaluation IDs and the same mask/readout
contract. Persist actual image IDs, initialization hashes and optimizer
counts. Report differences for2500->10k and10k->70k with their uncertainty;
do not infer a scaling trend from a single favorable seed. If the large-data
TRAIN loss is still falling, a longer-training comparison needs its own
prospective budget and no validation-based endpoint selection.

The final70k model must strictly exceed the comparable own70k Slot Attention
three-seed mean on FG-ARI, foreground IoU and matched-object IoU. Keep the
reserved independent set unread until the final recipe is frozen. Scores
from a different renderer, oracle allowed-label evaluation, or a different
patch grid cannot satisfy this objective.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `86b11384391d4f135174ff10b61d569c8f0f9641157416870ed1d0c971d886fb`

````markdown
# SW0134 native-spike competitive binding

Latest recovery observation: three actual training children verified in /proc and nvidia-smi on GPUs0/1/2. No scores yet. Evidence: SW_0134_native_spike_binding/recovery_gpu_start_20261010.json.

Latest: all three pilot training children were stopped when foreign PID2517185
entered their reserved GPUs. No checkpoint was saved; partial-update count is
unknown. Original state, empty logs and empty output folders were preserved.
An unchanged-code recovery queue (supervisor2566458) is verified alive and
waiting for exclusive GPUs; it restarts from the same registered warm32
state, not an interrupted checkpoint. No new score or scientific rejection.
See [recovery evidence](owner_interruption_recovery_20261009.json).

Pilot queue status: `running`. 3 child processes were verified live in /proc at the recorded observation.
The initial launch failed before its first update because direct script execution
could not import the repository package. Failure logs/state were preserved;
all child stages now use Python module execution. Seven server CPU checks,
including the actual child entrypoints with PYTHONPATH removed, passed before recovery.
No performance result or model promotion is available. See [pilot evidence](pilot_launch_review_20261009.json).

The scientific code bundle has been deployed with exact file/dependency hashes.
Independent local and server CPU suites both passed 13 tests, including real
throwaway optimizer updates with source-shaped synthetic assets for all three
arms. Actual source checkpoint provenance was validated for seeds 0/1/2.
All three actual-source GPU preflights passed canonical validation, including
native full1024 parity, warm32 provenance and all three real disposable arms.
Seed1 passed after a technical recovery preserving the original GPU-owner
interruption evidence and scientific recipe. Its supervisor/child are terminal.
The earlier preflight-only observation preceded pilot launch. See [passed preflight evidence](results_archive/passed_preflights_20261009/summary.json)
and [exact first-pass evidence](results_archive/first_pass_preflights_20261009/summary.json).
The recovery preserved the original state/log hashes and scientific recipe;
see [recovery process evidence](seed1_recovery_launch_20261009.json).
The unchanged SW0133 pilot is complete; its registered expansion gate failed.

See [deployment evidence](scientific_bundle_deployment_review_20261009.json).

The separate preflight queue was subsequently deployed and launched after the
expanded local and server suites both passed 15 tests. Supervisor PID 2427541
was confirmed live in `/proc`. At launch observation all four GPUs had a foreign
compute owner, so seed0 was waiting for an exclusive GPU and seeds1/2 were queued.
The queue reserves an available GPU automatically; seeds1/2 depend on a valid
seed0 preflight. This is a queued GPU diagnostic, not completed main training.
See [launch and process evidence](preflight_queue_launch_20261009.json).

The queue subsequently reserved GPU0 and started seed0 child PID2483147.
Both that child and supervisor PID2427541 were confirmed live in `/proc`.
See [actual GPU start evidence](gpu_preflight_start_20261009.json). Seeds1/2
remain dependent on a canonically valid seed0 result; no performance result
or model promotion is implied by this technical diagnostic.

This candidate replaces QCC-seeded assignment with an explicitly adopted
Slot Attention-style binder over complete, raw actual spike histories. The
native dynamics remain intact. Three matched arms will distinguish downstream
signal use, joint representation learning, and learned-readout effects. Both
the new spike-derived masks and original QCC readout will be reported.

See [registered protocol](protocol.json) and
[design rationale](../SW_0133_soft_partition_rgb/design_review_after_soft_pilot_20261009.md).
Synthetic helper tests establish implementation properties only. They do not
prove gating usefulness, scaling improvement, or superiority to Slot Attention.

````

</details>

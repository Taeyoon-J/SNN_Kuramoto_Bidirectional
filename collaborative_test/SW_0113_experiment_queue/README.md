# SW0113: one owner-aware experiment dispatcher

This queue retains the existing SW108 seven tasks and SW109 twenty-four tasks and adds the independently preregistered SW110, SW111 and SW112 branches. It is scheduling infrastructure, not a new model hypothesis. A single central dispatcher owns scheduling; independent coordinators with process-local GPU reservations must not launch concurrently.

## Safe handoff

Before superseding the existing waiting SW108/SW109 coordinators, verify their owned worker sets are empty and record their PIDs, states, logs and completed artifacts. Preserve private state/logs and completed outputs. Stop only the verified owned waiting coordinators. Never kill foreign GPU processes or restart already completed tasks. Import task identities/statuses, rather than silently enqueue duplicate jobs. Root/Luna own the implementation and live handoff; this document does not perform it.

## Scheduling and dependencies

- SW108 source baseline must validate before that source's causal interventions. Its completion is not a dependency of SW109/110/111/112.
- Each SW109 seed source/data preflight precedes that seed's training arms; each saved endpoint precedes its evaluation. Preserve all twenty-four existing task contracts. Remove the unrelated whole-SW108 completion barrier.
- Each SW110/111/112 source/cache/implementation/real preflight precedes its own paired256 arms and full320 evaluations. SW111 additionally requires its fresh per-seed shared decoder warmup and the single seed0 lambda artifact before its training arms.
- Prioritize ready short model pilots and their necessary preflights/evaluations. SW109 preflights remain independently eligible; long4375-update jobs have lower scheduling priority and must eventually receive service. Do not interrupt already running owned workers to enforce priority.
- All three seeds and candidate/control endpoints are preregistered. Reuse matched controls only after the exact core-update and provenance audit in each protocol; otherwise run fresh controls. Decoder/core state or new adapter state cannot be silently dropped.
- Promotion gates open only explicitly declared downstream work. One candidate's endpoint/preflight failure blocks its dependent branch, not unrelated valid jobs. A global source/hash mismatch can stop every affected source-dependent branch. No reserve access is added by this queue.

## Global GPU ownership

Use one live dispatcher and per-GPU global OS leases (for example fcntl locks on the server) shared by every owned launcher. Before launching CUDA, acquire the GPU lease, recheck actual foreign ownership/process and memory eligibility, and persist the task reservation. Launch at most one worker per genuinely free GPU0..3, at most four workers total. Verify ownership again after CUDA registration. A local dictionary is insufficient during CUDA startup. A lease is held for the worker's lifetime and released only after verified terminal status; stale lease recovery must check PID and recorded worker identity. A foreign owner appearing on a GPU stops or defers only the affected owned task safely and preserves its partial state. It never authorizes killing that owner's process.

Later SW0111/SW0112/SW0114 runners may be appended through `append_registry.json` while the dispatcher is live. The registry is append-only and versioned: each update increments `version` by one and preserves the prior experiment-name prefix. Only reviewed SW0111/SW0112/SW0114 adapters with the required task-plan, artifact, command, and validator API are accepted. Existing task definitions and outputs are never rewritten; newly appended tasks must use unique IDs and valid dependencies.

## Durability and failure

Persist task ID, source/config/script SHAs, dependency artifacts, GPU lease, worker PID/start identity, stage, exit status and timestamps with atomic state replacement. Record failures and partial checkpoints without automatic scientific retries or coefficient changes. Resume only through the runner's explicitly validated state contract. Poll queues without monopolizing GPUs; record waiting reasons and remain responsive to user pause/cancel. Every scientific runner must pass local review before eligibility. Existing SW107/SW105 or other live owned work is preserved and reconciled before filling remaining capacity.

# Active all-GPU research queues

2026-10-17 user requested all four GPUs be eligible for queued tests. Every ready
supervisor scans0,1,2,3 with the same exclusive lease and foreign-owner guard.
This does not imply four GPUs are idle or that another owner's job is displaced.

| Job | Actual parent PID | Archive queue path | Dependency |
| --- | ---: | --- | --- |
| SW0194 native64 TRAIN64 seed0 | 1849671 | SW_0194_native64_train64/results_archive/queue | Already running onGPU2; worker1849711 |
| SW0196 numerical/throughput benchmark | 1867954 | SW_0196_geodesic_matmul/results_archive/queue_all_gpus | Ready, all4 eligible |
| SW0197 native64 baseline seed1 | 1867955 | SW_0197_native64_baseline_replication/results_archive/seed1/queue | Ready, all4 eligible |
| SW0197 native64 baseline seed2 | 1867956 | SW_0197_native64_baseline_replication/results_archive/seed2/queue | Ready, all4 eligible |
| SW0195 paired TRAIN/development diagnostic | 1853195 | SW_0195_native64_generalization_gap/results_archive/queue | Requires completed194 final model |

The old196 waiter1857076 was stopped before CPU admission or GPU work, with its
queue state preserved. Its optional195 dependency was removed in the replacement
ready queue.196 scientific operator/model/cohort/gates remain unchanged.194 and195
sources/checkpoints were not modified or restarted.

28 local and Linux preflight tests,4 CLI entrypoints passed. All three new ready
queues additionally completed actual CPU model admission before waiting for GPUs.
Registration receipts and model hashes are under197/results_archive/all_gpu_queue_v1.
196's scheduling delta/tests are under196/results_archive/source_tests_v2.

197 is read-only32-image baseline replication, not a new learning recipe. Its
seed0 is reused from193, and all three development seeds must not be presented as
the full320-image result. The final native64 FG-ARI>=.90 goal remains unverified.

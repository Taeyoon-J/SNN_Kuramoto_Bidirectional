# Active all-GPU research queues

2026-10-17 user requested all four GPUs be eligible for queued tests. Every ready
supervisor scans0,1,2,3 with the same exclusive lease and foreign-owner guard.
This does not imply four GPUs are idle or that another owner's job is displaced.

| Job | Actual parent PID | Archive queue path | Dependency |
| --- | ---: | --- | --- |
| SW0194 native64 TRAIN64 seed0 | 1849671 (completed) | SW_0194_native64_train64/results_archive/queue | Completed; audited .9002/.8308/.7317 on32 development images |
| SW0196 numerical/throughput benchmark | 1867954 (completed) | SW_0196_geodesic_matmul/results_archive/queue_all_gpus | Speedup1.6558 but Q/mask parity failed; not adopted |
| SW0197 native64 baseline seed1 | 1867955 (completed) | SW_0197_native64_baseline_replication/results_archive/seed1/queue | Audited32 images: .8175/.4478/.6665; seed2 pending |
| SW0197 native64 baseline seed2 | 1867956 (completed) | SW_0197_native64_baseline_replication/results_archive/seed2/queue | All3 baseline seeds audited; .8514/.5781/.6940 on32 images; .90/full320 not achieved |
| SW0195 paired TRAIN/development diagnostic | 1853195 (completed) | SW_0195_native64_generalization_gap/results_archive/queue | Audited TRAIN16/dev32; no demonstrated overfit signal |
| SW0198 native64 readout normalization | 1943908 (completed) | SW_0198_native64_readout_normalization/results_archive/queue | Actual32 masks audited; gate failed, no adoption |
| SW0199 matched QCE control | 1973294 (completed) | SW_0199_native64_absolute_calibration/results_archive/control/queue | No new accepted steps; actual model and16 masks equal warm; unscored pending pair |
| SW0199 absolute-Q calibration | 1973323 | SW_0199_native64_absolute_calibration/results_archive/calibrated/queue | StartedGPU2; yielded to foreign owner, ready to resume original boundary |
| SW0200 full settled-window temporal credit | 2000695 | SW_0200_native64_long_credit/results_archive/queue | Actual GPU1 full512 gradient admission passed; worker2003084 training |

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

199 adds two matched short training jobs, unlike the read-only195/196/197/198
diagnostics.20 Linux tests and2 CLI entrypoints passed, including real CPU
checkpoint/Adam/RNG admission for both arms before scheduling. Its registered
16-image mini compares calibrated versus control AND frozen TRAIN8 warm reference.
No fresh70k training or performance improvement is claimed at registration.

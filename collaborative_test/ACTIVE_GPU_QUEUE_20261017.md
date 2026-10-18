# Active all-GPU research queues

2026-10-17 user requested all four GPUs be eligible for queued tests. Every ready
supervisor scans0,1,2,3 with the same exclusive lease and foreign-owner guard.
This does not imply four GPUs are idle or that another owner's job is displaced.

| Job | Actual parent PID | Archive queue path | Dependency |
| --- | ---: | --- | --- |
| SW0194 native64 TRAIN64 seed0 | 1849671 (completed) | SW_0194_native64_train64/results_archive/queue | Completed; audited .9002/.8308/.7317 on32 development images |
| SW0196 numerical/throughput benchmark | 1867954 (completed) | SW_0196_geodesic_matmul/results_archive/queue_all_gpus | Speedup1.6558 but Q/mask parity failed; not adopted |
| SW0197 native64 baseline seed1 | 1867955 (completed) | SW_0197_native64_baseline_replication/results_archive/seed1/queue | Audited32 images: .8175/.4478/.6665; all3 baseline seeds now complete |
| SW0197 native64 baseline seed2 | 1867956 (completed) | SW_0197_native64_baseline_replication/results_archive/seed2/queue | All3 baseline seeds audited; .8514/.5781/.6940 on32 images; .90/full320 not achieved |
| SW0195 paired TRAIN/development diagnostic | 1853195 (completed) | SW_0195_native64_generalization_gap/results_archive/queue | Audited TRAIN16/dev32; no demonstrated overfit signal |
| SW0198 native64 readout normalization | 1943908 (completed) | SW_0198_native64_readout_normalization/results_archive/queue | Actual32 masks audited; gate failed, no adoption |
| SW0199 matched QCE control | 1973294 (completed) | SW_0199_native64_absolute_calibration/results_archive/control/queue | No new accepted steps; actual model and16 masks equal warm; matched pair audited |
| SW0199 absolute-Q calibration | 1973323 (completed) | SW_0199_native64_absolute_calibration/results_archive/calibrated/queue | Audited16 images: .9005/.8612/.7356 vscontrol .9244/.8599/.7276; FG gate failed, recipe closed |
| SW0200 full settled-window temporal credit | 2000695 (completed) | SW_0200_native64_long_credit/results_archive/queue | Audited16 images: .8997/.8577/.7234 vscontrol .9244/.8599/.7276; failed extension gate, recipe closed |
| SW0201 normalized spike group-count loss | 2014332 (completed) | SW_0201_native64_normalized_count/results_archive/queue | Audited16 images; no committed new steps, model and scores unchanged; recipe closed |
| SW0202 finite Adam, original QCE | 2032092 (completed) | SW_0202_native64_finite_Adam/results_archive/queue | Audited16 images .9001/.8596/.7281; FG gate failed, finite steps applied2; recipe closed |

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

199 calibrated is the oldest unfinished candidate after foreign-owner yielding.
It obtainedGPU1 after201 completed.202 was registered only after that actual
resumption, and199 was not interrupted. No GPU workload is restarted or displaced
to change queue order.

## SW0203 registered 2026-10-18

Three additional native64 paired geometric-consensus readout jobs passed four local
and Linux tests, CLI validation and actual CPU admission of all three independent
frozen models before scheduling. Parents2042904/2042905/2042906 correspond to
seeds0/1/2. Each scans physical GPUs0,1,2,3 under the existing exclusive shared
lease and foreign-owner guard. All three were observed waiting;202 remains
running on GPU1. New203 changes grouping affinity units only, with unchanged
foreground support and no optimizer updates. Eight fixed development images per
seed; this is not a full320 score or new training result.

202 completed and203 seed2 worker2043968 obtainedGPU1. Other two203 seeds remain
registered exclusive waiters; all four GPUs remain eligible. Actual worker PID
and NVIDIA allocation were observed, not inferred from a state file alone.

## SW0204 ready learning test

Parent2049352 registered after18 local/Linux checks, two CLI checks and actual
CPU model/Adam/RNG admission. QCE(Q)+mean signed-component QCE is compared to
completed202 using the same finite-step policy and source TRAIN8 checkpoint.
Same two new B4 proposals, fixed16 development images, original inference.
All physical GPUs0..3 eligible; currently queued, not claimed GPU-started.

203 seed2 completed eight paired images and was independently audited: old
.9009074798/.3435644304/.7456563147, new .9012796105/.3435644304/.7444602992.
This negligible FG gain and lower object IoU is not an adoption signal.
Seed0/1 remain pending; no full320 or three-seed result is inferred.

203 all3 seeds completed and independently audited. On the identical eight
images, means old .8946/.5523/.7392, new .8952/.5523/.7390; tiny FG gain does
not meet the registered extension criterion and object IoU decreases. Recipe
closed. This is not full320 or a new .90 three-seed score.
204 worker2052928 subsequently obtainedGPU1; NVIDIA PID and inherited model/
Adam/RNG were verified. No204 result yet. All four GPUs remain eligible.

## SW0205 weak-seed native64 adaptation

Two ready jobs:seed1 parent2054790 andseed2 parent2054791. Six local/Linux tests,
queue CLI/AST checks and actual independent CPU model admission passed. Initial
weights must equal the corresponding frozen197 baseline, never seed0 weights.
Two guarded B4 QCE updates from fresh native64 Adam, following192 policy; fixed16
paired development masks per seed. This isolates native64 adaptation in the two
weaker seed lineages. No new loss or validation-data change. Both queues target
all four physical GPUs; waiting at registration while204 trains onGPU1.

204 completed:actual16-image seed0 score .9005/.8606/.7276, versus202
.9001/.8596/.7281 and warm .9244/.8599/.7276. FG difference +.0004 is not
a demonstrated gain; object IoU slightly lower. Independent48-mask/Adam audit
passed, extension gate failed, recipe closed.205 seed2 worker2066038 obtained
GPU1 afterward; seed1 remains queued. No205 evaluation result yet.

205 seed2 completed, audited .7884/.4455/.6743 vs baseline .8277/.4449/.6892;
one accepted Adam step, CNN changed. No adoption from this mini. Frozen-mask
CPU attribution shows increased cross-object merging despite reduced splitting.
Seed1 worker2095873 now trains onGPU1; actual initial independent model and
empty Adam checked. Three-seed result and full320/.90 remain unverified.

# SW0141 native32 source-preservation pilot — local candidate only

This folder contains a local candidate driver based on
`SW_0139_native32_representation/source_preservation_design.md`. It does not
change SW0139, register or run an experiment, start a server job, or claim a
pilot ran. Root review and explicit admission are still required.

The fixed comparison has two seed-1 arms, each starting from the same strict
SW0097/SW0135 native32 source and fresh Adam state:

* `c_only`: `lambda_C * C`
* `c_plus_anchor`: `lambda_C * C + lambda_A * A`

Both use the unchanged SW0139 symmetric temporal InfoNCE C, its fixed
canonical/paired brightness-contrast augmentation and nonce, 4096 source-
ordered TRAIN images, logical B16 accumulated as four B4 native rollouts,
T=1024/settle=512/live-tail=64, and the same graph/encoder learning rates and
single clip. The source Q anchor is float32 per-image off-diagonal MSE between
the student's native actual-spike product affinity and a detached same-source
teacher. No target masks are read during cache generation, calibration,
preflight, or training.

## Prepared command sequence (not executed here)

Set `PATCH_V2_REPO` to the actual `patch_v2_sw` checkout when running from the
private deployment folder. The seed-1 cache payload is 16 GiB; the source-0
first64 calibration cache is 256 MiB. The output directories are create-only.

```powershell
$env:PATCH_V2_REPO = 'C:\path\to\patch_v2_sw'
python pilot.py cache --seed 0 --device cuda:0 --output teacher_cache/source0_n64
python pilot.py cache --seed 1 --device cuda:0 --output teacher_cache/source1_n4096
python pilot.py calibrate --source0-cache teacher_cache/source0_n64 --device cuda:0
python pilot.py preflight --arm c_only --device cuda:0
python pilot.py preflight --arm c_plus_anchor --teacher-cache teacher_cache/source1_n4096 --device cuda:0
python pilot.py train --arm c_only --device cuda:0
python pilot.py train --arm c_plus_anchor --teacher-cache teacher_cache/source1_n4096 --device cuda:0
python pilot.py predict --arm c_only --device cuda:0
python pilot.py predict --arm c_plus_anchor --device cuda:0
python pilot.py score
```

The lambda calibration starts from fresh source0 and fresh Adam, performs one
disposable lambda-C C-only update, then measures four B16 accumulated C/anchor
gradient norms after that update. Zero/nonfinite anchor norms fail closed;
there is no epsilon fallback or re-tuning. The disposable state is never used
to initialize either seed-1 arm. Every source-Q B4 chunk is SHA-checked before
consumption; the cache manifest binds exact ordered IDs/rows, dependencies,
T/settle, dtype, grouping, and chunk hashes. Interrupted cache/output folders
are preserved and cannot be silently resumed or overwritten.

The endpoint freezes both new-arm QCC prediction arrays, the already validated
same-seed SW0137 source97 array, completed SW0139 old+C predictions, and the
registered same-seed Slot32 baseline before opening target masks. It reports
all three native32 production metrics and paired image bootstrap comparisons.
This remains a one-seed objective pilot: it cannot establish all-three-seed
Slot superiority, gate necessity, or a data-scaling result.

## CPU checks

With a NumPy-first environment and CUDA hidden for tests:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
python -m unittest discover -s . -p test_pilot.py -v
```

The checks cover the actual frozen helper's float32 Q loss/cache contracts,
ordered B4 chunk integrity, corruption detection, finite-difference-safe
provenance gates, and create-only artifact behavior. They do not substitute
for a real-source GPU parity/preflight or demonstrate training/evaluation.

## Private queue draft (not admitted or run)

`source_preservation_queue.py --dry-run` prints the ten-stage dependency plan
without creating state or querying GPUs. Its non-dry-run mode is guarded by
the shared SW0123 exclusive GPU lease/owner helpers. It caps GPU work at two
children, requires no compute owner and at most 512 MiB used, passes the lease
descriptor to each child, uses a per-child Triton cache, and stops only its own
process group if a foreign GPU owner appears. The final score child runs with
`CUDA_VISIBLE_DEVICES=''`. A failed stage is terminal; existing outputs, logs,
queue state, or partial artifacts prevent restart/overwrite.

`source_preservation_bundle_manifest.json` and
`install_source_preservation_bundle.py` are local packaging drafts. The
installer requires explicit `PATCH_V2_REPO`, verifies dependency and payload
hashes, uses create-only writes, and does not start the queue. Neither draft
constitutes deployment or experiment registration.

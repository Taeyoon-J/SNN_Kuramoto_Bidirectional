# SW_0141_native32_source_preservation

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `beb80958f80a346ae422c6be86a3ee0759ce4d29fec1dda51e72390288519449`

````markdown
# SW0141 native32 source-preservation objective pilot

Compare C-only and C+source-Q from fresh identical source1/Adam states. Both remove the old phase/Q statistical objectives and train encoder plus graph on the same ordered4096 images/256 logicalB16 updates. Native core/scalars stay frozen. The canonical source-Q teacher uses no masks.

Local and server CPU20/20 checks passed, including the actual checkpoint→protocol→prediction validator file roundtrip. Actual source RGB/gamma is bit-exact. Supervisor2908798 and cache workers2908822/2908823 were verified live on leased GPU0/1. Calibration, disposable GPU preflights, both training arms, frozen native32 predictions and GT-last fixed320 scoring are queued in the ten-task DAG. Both caches, source0 calibration and both disposable GPU preflights subsequently passed. Both training arms started on leased GPU0/1 (PIDs2926522/2926523) and were verified live; see [training_start.json](results_archive/training_start.json). Both training arms and native32 predictions completed. The original CPU score failed before GT access on a missing legacy Slot metadata key; a CPU-only in-memory hash-handoff repair completed scoring without retraining.

This is a one-seed inherited70k-lineage continuation pilot. It cannot establish three-seed Slot superiority, increasing-data gains or all-gating necessity. Only a candidate that beats source and C-only with positive paired95% FG lower bounds may justify seed/70k expansion. See protocol.json and results_archive receipts.

## Completed seed1 result (2026-10-11)

| Model | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| Source97 | .754135 | .606900 | .582201 |
| C-only | .752123 | .566258 | .580526 |
| C+source-Q | .752108 | .607767 | .581191 |
| Old+C (SW0139) | .738505 | .592047 | .566686 |
| Slot32 seed1 | .841819 | .203417 | .222115 |

Both FG expansion gates failed. Close both fixed recipes with no lambda-grid, seed or70k expansion. Removing old statistical priors recovered the old+C FG decline; source-Q recovered foreground IoU without object-grouping improvement. SW0137 remains the native32 incumbent. No three-seed win, data-scaling or all-gate claim.

[Raw evaluation](results_archive/score_recovery_20261011/evaluation.json), [recovery receipt](results_archive/score_recovery_20261011/score_recovery_receipt.json), [independent replay](results_archive/score_recovery_20261011/independent_endpoint_analysis.json), and [recovery code](score_recovery.py). Source and Slot per-image references exactly reproduce SW0137. The protocol retains its original registration-time status; this README and best-model tracker record the endpoint.

````

</details>

## source_preservation_pilot/README.md

<details>
<summary>README.md</summary>

Source SHA256: `b432e878c77526642da839587b70de5a54107ba23378353c03cf83e7ef0448a2`

````markdown
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

````

</details>

## successor_design_review.md

<details>
<summary>successor_design_review.md</summary>

Source SHA256: `fbc4eca883c124e26e681a6601e613ef130f6de3d736530c3fe2bba39bf05819`

````markdown
# SW141 이후: native32 spike binding의 masked completion과 core plasticity

설계 검토만 기록한다. 새 실험 번호/코드/queue를 만들지 않는다. SW141은 종료하며 coefficient 변경이나 다른 seed 확장을 하지 않는다.

## 결정과 근거

다음 한 가지 전환은 **실제 spike-only imagewise binder를 native32에서 직접 학습하면서, 입력에서 지운 영역을 object-slot RGB decoder로 복원하는 것**이다. 같은 objective에서 core 고정/학습을 비교한다. 같은 global contrastive objective에 threshold만 추가하는 실험을 우선하지 않는다.

SW141 seed1 C-only FG .752122679, C+A .752107669는 source .754135095와 Slot .841819292를 넘지 못했다. Source 대비 두 FG CI는 0을 포함한다. Old+C .738505보다 회복했으나 신규 object assignment 개선은 아니다. C는 다른 이미지의 patch를 negatives로 사용하고 A는 기존 Q를 보존하므로, 같은 이미지 안에서 instance를 생성할 직접적인 경쟁 학습이 약하다. 두 arm 모두 encoder/graph만 학습했고 native core와 12개 integration/threshold scalar는 고정됐다. History의 first/last minibatch 평균 및 clipping 비율만으로 이 원인을 입증하지 않는다.

SW134는 16-grid에서 학습한 뒤 SW136으로 32-grid 전이한 결과가 좋지 않았다. SW135 README에는 native32 resource probe만 있고 native32 binder 본학습 결과가 없다. 따라서 **native32 직접 학습 + masked completion**은 그 전이 실험의 재시도가 아니다. 기존 full-RGB reconstruction과 달리 가려진 patch는 자신의 RGB를 encoder에 전달할 수 없다. 이 문제를 해결하려면 visible 영역으로부터 core의 공간 coupling과 imagewise slots를 통해 정보를 모아야 한다. 이것도 object segmentation을 보장하지는 않는다.

## 고정 입력·architecture·objective

- Source: native32 SW137의 SW97 seed1 전체 source와 동일 encoder/statistics/spatial parameter mapping. 새 encoder와 graph를 random initialization하지 않는다. Source는 이미 70k에 노출된 16-grid lineage이므로 continuation pilot이다.
- RGB 128x128, 실제 1024 nodes, graph chunk/checkpoint, T1024/settle512/live-tail64를 유지한다. Static encoder/graph 계산은 burn-in 이전 live graph로 준비한다. State를 960에서 detach하며 모든 settled 512 spike frames를 readout에 사용한다.
- Mask는 128 이미지의 8x8 tile 중 정확히 16개를 TRAIN RNG로 선택한다. 한 tile은 16x16 pixels=4x4 native patches다. 해당 RGB를 uint8 128로 대체한다. Target은 변경 전 RGB다. TRAIN mask RNG는 ID/update에 hash-bound하며 두 arm에서 동일하다. Val inference는 unmasked RGB이다.
- Head는 SW135의 raw actual emitted spikes `[B,4,1024,512]`만 받는다. 2048→64 projection/LayerNorm, exchangeable 11-slot initialization, 3 GRU competition iterations를 그대로 사용한다. Temporal mean, history centering, raw theta/gamma/RGB/XY head input을 추가하지 않는다. 11은 capacity이며 occupancy나 GT object count를 강제하지 않는다.
- Shared relative-coordinate RGB decoder는 SW135의 66→64→64→3 및 checkpointed pixel chunks를 재사용한다. Slot latent는 spike-derived 값뿐이고 P가 유일한 pixel mixture weight이다. 독립 alpha head와 RGB skip을 금지한다. Decoder geometry는 P의 centroid/scale로 계산하며 P gradient를 유지한다.
- `R = [MSE(masked pixels) + 0.25*MSE(visible pixels)] / max(mean_c Var_pixels(X_c), 1e-6)`; RGB는 0..1, variance denominator는 detached unmasked TRAIN target에서 계산한다. 각 MSE는 해당 pixel/channel 수로 정규화한다. Logical batch는 imagewise R 평균이다. Old PLV/Q density prior, cross-view C, source-Q teacher 및 RGB target의 pseudo-instance labels를 추가하지 않는다.

Masking은 데이터셋 변경이 아니라 사전 등록한 TRAIN corruption이다. Full target RGB는 loss에만 사용한다. 학습/예측 mask는 실제 spike를 통과하며 평가 GT를 보지 않는다.

## 최소 paired pilot

1. **공유 head 준비:** source1 core/encoder/graph를 고정하고 matched 4096 TRAIN IDs의 masked native32 actual spikes를 한 번 cache한다. Float16 cache는 4096×4×1024×512로 payload 약16 GiB다. 실제 dtype 오차를 source float32와 기록하고 두 arm은 동일 cache를 사용한다. Binder/decoder Adam 1e-3, clip1로 256 logical B16 updates(4096 exposures), 동일 순서로 학습한다. Cache로 graph rollout을 매 head update마다 반복하지 않는다. Warm weights와 Adam moments를 둘 다 저장한다. 기존 32-update decoder row-scramble 양성 여부를 admission guard로 재사용하지 않는다: 충분한 joint learning 이전의 utility 실패만으로 이 신규 recipe를 폐쇄하지 않는다.
2. **Joint 두 arm:** 동일 warm head/moments와 fresh source backbone에서 시작한다. A는 encoder/graph만 trainable, core/12 scalars frozen. B는 encoder/graph와 native nongraph core, a_d/a_m/b 12 scalars도 trainable이다. 두 arm 모두 동일 masked R와 actual-spike head를 사용한다. Encoder LR3e-6, graph/native core LR3e-5, 새 a_d/a_m/b LR1e-3; a_d/a_m만 기존 [0,1] projection을 유지한다. Threshold는 `.06*exp(b)`이며 임의 sparsity/event-rate target은 없다. R coefficient는 1로 고정한다. Backbone과 head에 별도 Adam/clip1을 사용하며 각각 R gradient를 한 번만 계산한다. 두 arm 256 logical B16 updates, microB4×4, 동일 4096 IDs/order/masks이다. Warm exposure 4096 + joint exposure4096을 명시한다.
3. **Preflight:** 먼저 source unmasked zero-init native32 rollout parity와 실제 masked B4 memory/optimizer update를 확인한다. 실제 R가 encoder/graph에 전달되는지, B에서는 dendrite/membrane/threshold/integration에 finite credit 및 disposable update가 존재하는지 기록한다. Projection boundary에서 일부 parameter가 그대로일 수 있으므로 scalar gradient와 실제 a/b 이동을 별도 기록한다. 모든 event-rate 변화를 로그로 남기되 특정 occupancy를 맞추기 위해 calibration하지 않는다. 두 arm의 frozen parameter는 hash로 확인한다. Native defaults/arithmetic를 변형하여 activity를 만드는 SW126 recoding을 재사용하지 않는다.

GPU time은 새 masked/head setup에서 실제 측정한다. 기존 B4 resource 33.86s는 상한 보증이 아니다. Cache + head warm + 두 arm256만 먼저 실행하며 4096 joint updates를 미리 예약하지 않는다. Random fresh binder의 256-update 실패는 이 budget에서의 실패이고 architecture 전반 불가능의 증거가 아니다. 그렇다고 결과를 보고 추가 epochs를 선택하지 않는다.

## 평가·기전 확인·폐쇄

최종 unmasked fixed320 IDs1320..1639에서 **P argmax native32 labels**가 신규 primary readout이다. Hard-slot 최대 면적을 BG로 지정하고 동률은 첫 patch 등장 순서로 canonicalize한다. Source QCC/Slot의 기존 BG 규칙은 그대로 유지하고 차이를 명시한다. 동일 core actual spike의 production positive-clamped product QCC(.50/min8/largest-component BG)를 secondary로 별도 보고한다. Primary와 secondary를 유리한 쪽으로 선택하지 않는다. GT 4px modal/min-ID ties 및 세 metrics는 변경하지 않는다. 모든 predictions/hash를 먼저 고정한 뒤 GT를 읽는다.

Warm 및 최종 first64 TRAIN에서 (a) P row permutation, (b) detached target patch-mean RGB를 채널별 4 bins로 나눈 동일 color bin 안의 P permutation을 진단한다. Slot latents는 그대로 두고 P geometry/mixture만 다시 계산한다. 첫 검사는 assignment 사용, 두 번째는 coarse color histogram 외 공간 assignment의 사용에 대한 제한적 증거다. 둘 다 학습 label/threshold 선택에 쓰지 않는다. Decoder loss 개선만으로 instance 의미를 주장하지 않는다.

Stage1 성공은 B primary FG가 source1 및 A보다 **각각 +.01 이상**, 두 paired image-bootstrap 95% CI lower bound>0, 두 IoU는 native32 Slot mean보다 각각 +.05 이상이다. 이 규칙은 시행 전에 고정한다. 실패하면 이 recipe를 닫고 mask fraction/LR/slot-temperature/window를 sweep하지 않는다. FG 개선 없이 단지 nonzero gradient나 nonsaturation이 생겼다고 확장하지 않는다. 성공하면 같은 recipe로 source seeds0/2를 모두 실행하고 all3/Slot 비교를 한다. Single-seed result는 최종 목표 완료가 아니다.

Gating 기여는 B가 성공한 경우에만 SW140의 exact true-dynamics interventions를 새 checkpoint에서 paired endpoint로 반복하고, 유용성이 보이는 event/retention intervention은 matched retrained control로 확인한다. Source140의 K/delayed gate/D-history 효과와 membrane/event 약함을 그대로 보존해 보고한다. Core-plasticity A/B 승리는 모든 gate의 개별 필요성을 입증하지 않는다.

그 뒤에만 동일 새 model 초기화에서 native32 70k 세 seed 학습과 fixed-update unique-data scaling을 등록한다. 70k pretrained source에서 pilot continuation gain을 fresh data-scaling 효과로 바꾸어 쓰지 않는다. Reserve90640..90959는 최종 recipe freeze 전 읽지 않는다.

````

</details>

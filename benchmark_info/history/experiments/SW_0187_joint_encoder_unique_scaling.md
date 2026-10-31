# SW_0187_joint_encoder_unique_scaling

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `f036768fe25fdd2357a2ca5569beea7a7caf7fb9d6eaf02201ea122be9d1ade4`

````markdown
# SW0187 — 공동 encoder 조건의 데이터 증가 검증

186번은 고정 native32/320장 세 seed 평균 FG를 .814836에서 .834681로 높였다.
원본 encoder까지 학습하는 이 조건에서 고유 학습 데이터 증가를 검증한다.
기존186과183 소스·결과는 변경하지 않는다.

각 seed0/1/2는 같은 Source097 초기 tensor, zero RF31 context, 빈 Adam으로
시작한다. 원본 RGB encoder source_base LR3e-5를 포함한 다섯 family와
QCE temp.1, B16/micro4, T1024/settle512/live-tail64, clip1, actual full-forward
accept/rollback은186과 같다. 원본 CNN 외 frozen preprocessing은 유지한다.

첫512 이미지를 사용하는32 proposals를 한 번만 계산하고 model/Adam/RNG,
accepted counter를 두 arm으로 정확히 복제한다. control512는 기존512를
두 번째 permutation으로 반복한다. candidate1024는 같은 index+512의 새512를
사용한다. 각 arm의 논리 예산은64 proposals/1024 exposures, 실제 연산은
공통32+두 branch32=96 proposals/seed다. Accepted 수를 맞추는 retry는 없다.
같은 기존 TRAIN1024 cache를 재사용하고 첫512 RGB/GT/ID byte parity와
나머지512의 canonical TRAIN 소속을 검증한다. VAL/reserve는 학습하지 않는다.

실제 세-seed CPU factory에서186과 초기 model/gamma parity, 다섯 family의
중복 없는 optimizer, empty Adam, CNN gradient와 변경, frozen preprocessing을
확인한다. 공통 prefix의 첫 두 GPU proposals에서 emitted-S 및 다섯 family의
positive gradient/proposed delta와 full-forward parity를 확인한다. 각 branch
최종 checkpoint의 실제 CNN SHA 변경과 모든 Adam step도 검증한다.

두 arm×세 seed의 conditional/standalone fixed320 예측과 protocol을 모두
확정한 뒤에만 GT를 채점한다. Primary는 standalone이다. 고유 데이터 이득은
candidate FG mean>동시대control이며 paired-image bootstrap95% CI 하한>0일
때만 인정한다(RandomState178,10000회, 같은 이미지의 세 seed 평균 차이).
Slot70k 대비 세 지표 엄격 초과 및 FG CI, 기존186/181 대비 차이와 FG.87도
별도로 보고한다. Foreground IoU 저하를 숨기지 않는다. 독립 검산은 실제
candidate3840 endpoint-image 행과 Slot960 행, modal GT, means/CI/gate 및
동일 prefix/분기 RNG/Adam, 실제 CNN 학습을 확인한다.

과거186 prefix 해시 비교는 진단 기록이며 실행 허용 조건이 아니다. 같은
현재 prefix에서 분기하는 비교가 인과 비교의 근거다. 실패 시 이 recipe의
작은 LR/온도 rescue sweep은 하지 않는다. 성공해도 supervised pilot이며
fresh70k, unsupervised 개선, gate 인과 기여나 ordering 최적성을 주장하지 않는다.
원래70k/.87/논문 기전 목표는 남는다.

유휴 GPU만 독점 사용하고 외부 작업이 시작되면 우리 worker만 양보한다.
관측 timeout으로 작업을 재시작하지 않는다. CPU/실제 CLI/cache/fork와
평가·검산 admission을 먼저 통과시킨 뒤 배포·큐 등록한다.

````

</details>

## FOUNDATION_PROVENANCE.md

<details>
<summary>FOUNDATION_PROVENANCE.md</summary>

Source SHA256: `6703bf583d3d4fa227136e07219937952d4ed16ec274470fb338e6aa80e99937`

````markdown
# Foundation training budget clarification

Source097 inherits a core that had already trained on the complete 70,000-image
RGB pool. Its final 4,096-image stage is additional training, not its complete
training history. The server receipt verifies all three actual Source097
checkpoint hashes, their Source095 parents and the Source090 epoch-1 parents.
Source095 manifests record 4,375 batches of 16 and 70,000 distinct TRAIN IDs;
Source097 manifests record another 256 batches of 16. Both stages record no
ground-truth masks used for training.

Consequently, SW0186's TRAIN512 twice and SW0187's repeated512 versus unique1024
refer only to the additional supervised native32 adaptation stage. SW0187 tests
the benefit of distinct adaptation images and labels at equal proposal budgets.
It does not test whether the inherited model benefits from seeing previously
unseen RGB images, nor whether total foundation data increased from 512 to 1,024.

The documented downstream lineage is Source090 epoch1 (70,000 exposures),
Source095 (another 70,000), then Source097 (4,096). Earlier encoder/graph
pretraining has a separate budget. Later Source090 epochs are not included in
this inherited checkpoint chain. A new native32 joint-encoder 70,000-image
training pass remains outstanding.

Evidence: [three-seed server receipt](results_archive/foundation_provenance_v1/foundation_provenance_receipt.json),
SHA256 `87b812f2d4256c9805640c2f5cdcbb0d7b28b32b2a2f2a5562d312c65d0d0e10`.
No validation masks were read for this provenance check. This clarification
changes no registered model, training, prediction or scoring source.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `f94dc64519de7e074db2a6dcc6c30573f539f08b0e34c73100adcb48c95fbc64`

````markdown
# SW0187 ? joint encoder, unique1024 versus repeated512

Registered: three training supervisors, three prediction supervisors, GT-last
scoring and an independent final auditor. All eight parents were verified live.
Seed1 has subsequently actually started on GPU1. Its first two real proposals
verified emitted-S/five-family credit and full-forward parity; a committed update
changed the original CNN and populated its Adam state. The scoped proof is in
`results_archive/start_v1/seed1_first_two_GPU_admission.json`. This is a working
learning path, not a new performance result. Other seeds remain registered.

The improved SW0186 recipe keeps its original RGB CNN trainable. Both arms share
one contemporaneous32-proposal prefix, then fork exact model/Adam/RNG state.
Control repeats512 images; candidate uses1024 distinct images at the same
logical64-proposal/1024-exposure budget. Shared computation costs96 actual
proposals per seed. This tests unique data in this additional training stage,
not whether images were unseen by the pretrained Source097 foundation.

Local41 tests and six actual CLI checks passed before deployment. Server45 tests,
all six actual CLIs, all three five-family factories and initial gamma parity,
the canonical TRAIN1024 cache and all three actual Slot archives passed before
registration. Initial tensors also matched the archived SW0186 initialization.
Sources/queues are frozen while live. No datasets, checkpoint weights or keys
were transferred. The original preregistration README is archived in cpu_v1.

The primary scaling gate compares candidate versus its contemporaneous control,
with a positive paired-image FG confidence bound. Slot70k, SW0186, SW0181,
conditional masks and both IoUs are reported separately. Fresh70k, FG.87,
unsupervised improvement and gate causality remain outstanding.

See [the registered design](DESIGN_KO.md), `results_archive/cpu_v1` and
`results_archive/start_v1`. Best supervised FG remains SW0186; unsupervised SW0165.

Seed2 actual GPU credit/parity, accepted Adam steps and changed original CNN
were independently inspected; see `results_archive/start_v1/seed2_first_two_GPU_admission.json`.
This admission does not establish improved masks or a completed scaling result.

Seed0 actual GPU credit/parity, accepted Adam steps and changed original CNN
were independently inspected; see `results_archive/start_v1/seed0_first_two_GPU_admission.json`.
This admission does not establish improved masks or a completed scaling result.

Seed0 has completed its shared prefix and both64-proposal branches. Actual
model/Adam artifacts and CNN learning passed validation (37/39 accepted updates
for control/candidate). The first branch model/Adam transaction was identical
and candidate indices were exactly control+512. Prefix RNG bindings agree and
the frozen runtime fork-RNG guard executed; the transaction hash itself excludes
RNG. Records are in `results_archive/completed_seed0_v1`. No GT score is available;
all six prediction freezes are still required before scoring.

Seed2 has also completed both trained branches and actual artifact validation
(control/candidate accepted47/45). Seed0/2 trained artifacts are preserved.
Seed1 was fresh-requeued once to release its GPU1 resume restriction. Its15
partial proposals and checkpoint are preserved on the server; those15 proposals
are reported as extra compute excluded from the delivered models. The same
model/data/loss/factory fingerprint is verified. The replacement parent is live
and its original prediction parent resumed. Current-run GPU admission for the
replacement seed1 must be checked again. See [the resource amendment](RESOURCE_REQUEUE_KO.md)
and `results_archive/resource_requeue_seed1_v1`. No validation score selected this
resubmission. All model source files remain frozen; no new performance is claimed.

The replacement seed1 has actually started on GPU2. Its fresh-run first two
proposals passed five-family/emitted-S credit and full-forward parity; committed
updates changed the original CNN. The new-run receipt is in start_v1 with the
resource_requeue filename, separately from the retained original admission.
Seed0 both-arm fixed320 predictions are also frozen and validated in
`results_archive/frozen_seed0_v1`. GT has not yet been scored.

All three delivered cohorts have now completed their shared prefix and both
64-proposal branches. Control/candidate accepted counts are37/39 (seed0),31/35
(seed1 replacement),47/45 (seed2). All six actual trained CNN/Adam artifacts and
shared model/Adam fork/data-ID checks passed. The resource-only excluded15
proposals remain preserved separately. Masks for the remaining seeds are queued;
no new FG-ARI or unique-data benefit is claimed before the full six-archive score.

Training-budget clarification: Source097 already inherits full70k RGB training; 4,096 is its final additional stage. The 512/1024 counts describe additional supervised adaptation only. See [verified foundation provenance](FOUNDATION_PROVENANCE.md).

Completed: three-seed candidate .826556/.577652/.679338 versus contemporaneous control .829479/.608872/.680583. Unique-data FG gate fails; SW0186 remains the supervised FG incumbent. Independent audit v2 passed; original audit identifier failure is preserved. See [completed results](RESULT_KO.md).

````

</details>

## RESOURCE_REQUEUE_KO.md

<details>
<summary>RESOURCE_REQUEUE_KO.md</summary>

Source SHA256: `404bed1832a21c95f3c315de1c7c3f89a7d1c82466e377ae7be4dd138debb5a3`

````markdown
# Seed1 ?? ??? ??

Seed0?2? ? branch ?? ?? ??? seed1? GPU1 ??? ?? ????.
GPU1 ?? ?? ??, ?? training/prediction parent? ?? ?? PID? ??
??, ??? prefix15, ??? CPU/factory/queue fingerprint? ?? ????.
??320? ? ?? ??? ?? ?? ???, ? ??? ??? ???? ???.

? ?? ?? ???? ????. ?? prefix15 checkpoint????? ??????
??? `SW0187_delivery_20261017/resource_requeue_seed1_v1`? ????.
?? seed1 prediction parent? ?? ??? ?, GPU ??? ?? ?? training
parent? ???? ?? ?????????? ???loss??? ??? fresh ???
?? ?? ????. Prediction parent? ? ???? ? ??? ?? ???
GPU? ? runtime? ?????. ?? ?? ?? ???? ???? ??? ??? ???.

?? ?? ????? ??? ?? ??? ????. Prefix?? branch? ??
??? ???64 proposals/arm,96 ?? proposals/seed?. ???15 proposals?
?? ??? ???? ?? ?? ?? ???? ?? ????. ??? ??
96?3 ?? ??15?? ?? ?? ??? ????. ?? ?? ??? ????,
?? seed1 ??? ??? ??? ???? ???. ? ??? ?? ?? prefix??
? ???? ???? ?? CNN ???model/Adam?RNG runtime ??? ?? ???? ??.

Dry-run ??? ?? journal, ?? ?? SHA? ? PID? ??? ? ?? ????.
?? ?? timeout??? ? ??? ?? ???? ?? ?? PID? ?????.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `044ba13cd76255602819a9221262b3769d7c5c1b73da554c708cdf99114dc6d8`

````markdown
# SW0187: additional unique images did not improve FG-ARI

All three seeds, both trained arms and all fixed320/native32 predictions are
complete. The independent v2 audit recomputed 3,840 candidate endpoint-image
rows and 960 Slot rows, all means, paired confidence intervals and gates.

| Model / additional adaptation data | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0187 control: repeated512 | 0.829479 | 0.608872 | 0.680583 |
| SW0187 candidate: unique1024 | 0.826556 | 0.577652 | 0.679338 |
| SW0186 incumbent | 0.834681 | 0.569040 | 0.689802 |
| Comparable Slot70k epoch10 | 0.781652 | 0.204360 | 0.203165 |

| Seed | control FG-ARI | candidate FG-ARI |
|---|---:|---:|
| 0 | 0.848430 | 0.854629 |
| 1 | 0.787443 | 0.782767 |
| 2 | 0.852564 | 0.842273 |

The primary candidate-minus-contemporaneous-control FG difference is -0.002923;
paired-image 95% CI [-0.007999, 0.002256]. This does not establish an FG decrease
or increase. Foreground IoU falls by 0.031219, CI [-0.039247, -0.023166].
Object IoU changes by -0.001246, CI [-0.006058, 0.003537]. The scaling gate fails.
Against historical SW0186, candidate FG changes by -0.008124,
CI [-0.014093, -0.002209]. No incumbent changes and no FG.87 claim are made.
Both SW0187 arms exceed the comparable Slot means in all three metrics.

The original RGB CNN genuinely learned in all six models; accepted control/
candidate updates are 37/39, 31/35 and 47/45. An encoder frozen by mistake is
therefore ruled out for this run. This does not establish that the loss, graph
or discrete assignment uses the additional data effectively. Close this exact
recipe; do not spend another cycle on small learning-rate/temperature changes.
Any next experiment should address the objective-to-assignment mechanism or
test continued learning from the actual incumbent with an explicit budget.

The 512/1024 counts are additional supervised adaptation data, not total
foundation training data. Source097 already inherits full70k RGB training.
See [verified foundation provenance](FOUNDATION_PROVENANCE.md).
This is not a fresh native32 joint-encoder70k run, unsupervised improvement or
gate-causality proof. The seed1 resource requeue excluded and preserved 15 old
proposals, which remain explicitly reported as extra compute.

The original audit v1 expected an obsolete experiment identifier and failed
before metric recomputation. Its source and failure receipt remain preserved.
Separate v2 corrects only that identifier and uses a separate output path;
the producer contract passes actual-entry-point tests locally and on server.
Training, masks, score code, metric definitions and thresholds were unchanged.
The v2 independent audit then passed. No GPU training or inference was repeated
to correct the audit. Maximum metric absolute error is recorded in the raw audit.

Score SHA256: `31bcb373a41fe533d0f2218c5861e6c7022996396536c79f9e8279d0a8716b11`.
Audit SHA256: `583b770aaa41b878b14e4de87092437d062263fb6823cc8118a59edc94e998ce`.

[Raw score](results_archive/score_v2/evaluation.json),
[independent audit](results_archive/score_v2/completion_audit.json),
[archive receipt](results_archive/score_v2/archive_receipt.json).

````

</details>

## results_archive/cpu_v1/registered_README.md

<details>
<summary>registered_README.md</summary>

Source SHA256: `cf8d982c2a67004a3da479b7ab3ef2c68fb8ace3c9735e60292bd123c0f32f60`

````markdown
# SW0187 — joint encoder, unique1024 versus repeated512

Preparing CPU and runtime admission. No GPU training has been started.
Both arms share a contemporaneous32-proposal prefix and then fork exact
model/Adam/RNG state. Only the second512 image identities differ.
The original RGB CNN remains trainable exactly as in SW0186.
See [the registered design](DESIGN_KO.md).

````

</details>

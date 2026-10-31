# SW_0232_native64_exact_teacher_resume

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `7aadaff27a755588f4d74719b024241d7ace7ac7b86a6975043431ce1b626f25`

````markdown
# SW0232: 수락된128회 checkpoint에서 직접 이어 학습

231번 재계산은 두 seed에서 가중치 약1e-7 수준의 차이로 bitwise128 prefix 기준에
실패했다. 실패 checkpoint/로그를 보존하고 이를 정확한 이어 학습으로 취급하지 않는다.
이번에는 원래223번의 실제 encoder/projection/Adam/history/RNG를 직접 복원한다.
동일성 검증은 완화하지 않으며 복원된128회 상태를 독립 child snapshot으로 보존한다.

loss·TRAIN64·teacher·batch16·LR/clip·graph/나머지 core/gates/readout은 유지한다.
원래128회에서384회 더 업데이트해 누적512회로 진행한다. 새 계산은384 updates/
6144 exposures이고 누적 학습은512 updates/8192 exposures다. 새128회 재계산은 없다.
원래 source로 만든 teacher를 그대로 유지하고, 실제 accepted128 모델의 CPU 복원과
gradient 검사도 GPU 큐 등록 전에 확인한다. 기존 child boundary가 있으면 child
checkpoint/Adam/RNG에서 재개하며 accepted source를 덮어쓰지 않는다.

고정16장·세 seed를 실제223번128회 모델과 비교한다. baseline FG>=+.01 및
paired CI 하한>0,128회 모델 FG 초과를 FG 후보 기준으로 유지하고 두 IoU 평균
비감소를 balanced 기준으로 유지한다. mask-supervised mini이며 전체320장 .90이나
70,000장 재학습 결과로 승격하지 않는다. 이 고정 예산 확인 실패 시 epoch sweep은 없다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `9acb5af0d0fcb354d5974ec8a97ef5933c419780b9701c8862e3eaaf4def682a`

````markdown
# 동료 최신 연구 재확인

이번 사이클도 origin/patch_v2를 fetch했고 최신 commit은
5c88a6faa3c7f3a63f2d2223303c2a74ae1d45db로 이전230번 검토와 같았다.
PV2_0103의 graph 고정 seed1 결과 .6777/.6895/.6357과 비고정16epoch 결과
.6867/.6008/.4887을 구분한다. FG 하나만으로 판정하지 않으며, 추정 평균과
다른 평가 계약을 우리 실측 최고 기록으로 사용하지 않는다.

우리는 고정 graph와 실제 세 seed/세 지표 비교를 유지한다. 이번231번은 동료의
16epoch를 그대로 도입하는 것이 아니라, 우리223번의 실제 학습 loss가 종료 시점에도
감소한 기록을 근거로 원래 objective의 고정 업데이트 예산을 한 번 검증하는 실험이다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `a6ad7ee9e948d9bdd846384c1c0c3d5fb42b99ae754ebaf9e8d657e6c7bd27bf`

````markdown
[Completed fixed16 result](RESULT_KO.md): .884622 / .677086 / .712312. Original FG gate remains false; full320 best unchanged.

# SW0232: exact accepted128 checkpoint continuation

Restore SW0223 encoder/projection/Adam/history/Torch RNG and verify bitwise equality,
then add384 updates to cumulative512 on the same TRAIN64/original teacher objective.
Read-only actual accepted-start CPU restoration/gradient admission precedes GPU queues.
Original checkpoint and failed SW0231 recomputations are preserved. No result yet.
Native64 fixed16/three-seed supervised mini; no new70000-image training.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `40c5e6dc8d7523206ffd1d17ff846901cb9244c4b2b9bd5c857ec68966269be5`

````markdown
# SW0232: 추가 학습 평균은 상승, 고정16장만으로 FG 개선 확정 불가

**장수와 업데이트 구분:** 기존 checkpoint에 추가 학습한 고유 이미지64장,
해당 추가 학습 단계의 누적 optimizer update512회, 평가 이미지16장,
seed0·1·2 평균 FG .884622다. 모델의 전체 학습 이력이64장뿐이라는 뜻도,
512장으로 학습했다는 뜻도 아니다. 기존 checkpoint의 이전 학습 이력은
그대로 상속한다. [평가 범위 구분](../EVALUATION_SCOPE_KO.md)을 함께 확인한다.

실제 accepted SW0223 128회 checkpoint의 encoder·projection·Adam·history·Torch RNG를 세 seed 모두 bitwise 복원했다. 같은 TRAIN64·teacher·LR·graph·나머지 gate/readout으로 384회 추가해 누적512회 학습했다. 새 노출은6,144회이며 상속된128회를 새 학습으로 세지 않는다.

고정16장·native64·세 seed 평균:

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 기준 | .842891 | .600692 | .702161 |
| 실제128회 SW0223 | .859634 | .645611 | .706340 |
| 누적512회 SW0232 | **.884622** | **.677086** | **.712312** |

128회 대비 FG 차이는 +.024988, paired image bootstrap 95% CI [-.006611,.062020]이다. Foreground IoU 차이는 +.031475, CI [.000588,.067281]이다. Object IoU 차이는 +.005971, CI [-.010016,.021859]이다. FG 평균 상승은 유망하지만 고정16장만으로 확정하지 않는다. 원래 FG/full320 candidate gate와 balanced gate는 모두 **false로 보존**한다.

원본128회 prefix·새384회 budget·세 actual512 checkpoint·teacher·raw mask144개를 독립 검증했다. 원시 점수·평균·CI 오차는1e-12 미만이며 로컬 재계산 오차는0이다. 실제 teacher loss와 DEV frequency의 물체 내부 변동은 세 seed 모두 줄었다. 이것만으로 최종 mask 개선이나 모든 gating의 기여를 단정하지 않는다.

다음은 예산이나 loss를 바꾸지 않고 이 고정 checkpoint를 기존 first64 validation에서 확인하는 SW0235다. 이전 SW0223도 동일한16장 불확실성을 secondary64에서 확인했으며, 기존 실패 gate를 덮어쓰지 않았다. 이번에도 같은 원칙을 따른다. 이미 사용한 validation이므로 untouched test로 취급하지 않는다. 실제128회 모델과 원래 모델을 모두 같은64장 조건에서 비교한다.

현재 전체320장 최고는 SW0227의 .858916/.605286/.674313이며 이 미니 결과로 교체하지 않는다. 신규70,000장 학습도, .90 전체 평가 달성도 아니다.

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `f0a4efa1fab69b215e253234512360d77fc94fd5e5968ae929082cb672dd4a28`

````markdown
# SW0232: 실제 accepted checkpoint에서 이어 학습

SW0223의 실제 128회 checkpoint에서 encoder·gamma projection·Adam·history·Torch RNG를 복원하고, 같은 TRAIN64와 teacher loss로 384회 추가해 누적 512회를 비교한다. 재계산으로 원래 128회를 재현하려던 SW0231과 구분한다. 동일성 기준은 완화하지 않는다.

새 학습 예산은 384 updates / 6,144 exposures이며, 상속된 128회를 포함하면 512 updates / 8,192 exposures다. graph·나머지 core/gating·spike/membrane readout은 그대로 유지한다. mask-supervised native64 미니 실험으로, 70,000장 신규 학습이 아니다.

세 seed를 고정 16장에서 baseline 및 실제 SW0223 128회 모델과 비교한다. FG 개선과 paired CI, 두 IoU를 함께 확인한다. 전체 320장이나 FG-ARI .90 달성으로 해석하지 않는다. 로컬 16개 검사는 통과했으며 서버 Linux 검사와 세 seed의 실제 accepted checkpoint CPU 복원·gradient 검사를 통과해야 GPU 큐가 등록된다. 실제 시작 단계는 START_EVIDENCE.json에 기록한다.

````

</details>

# SW_0228_native64_center_preservation

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `1e7a38f7c492854e81f1159137bc8068514a875638ef617def82528a6fe4dbc5`

````markdown
# SW0228: 원래 주파수 중심 보존 강화

223/226 진단에서 물체 내부 주파수 분산과 함께 인접 class 중심 거리도 줄었다.
특히 약한 seed1/2에서는 내부 분산과 중심 거리의 감소율이 비슷했다. 데이터512
확장 효과도 작고 CI가0을 포함했다. 이는 단일 원인 증명이 아닌 다음 loss 가설이다.

223번과 같은 원래 초기 checkpoint, TRAIN64,128 Adam updates, batch16,LR/clip,
encoder 전체 및 실제32-weight 입력 projection 공동학습을 유지한다. 다른 core,
graph/gates/statistics/readout은 고정한다. 기존 source teacher의 class별 주파수
중심을 그대로 사용하며 임의의 단위1 margin이나 class/channel 순서를 만들지 않는다.

기존 teacher MSE는 class-balanced within variance + source-center MSE로 분해된다.
이번 loss는 within variance +16×source-center MSE +원래 gamma anchor다. source
중심에서 벗어나는 벌점만15배 추가하고 within 항의 계수1은 유지한다. 계수16은
현재 validation 진단에서 선택한 하나의 사전 고정 비교이며 계수 sweep은 하지 않는다.
TRAIN class_labels는 teacher에만 추가하고 예측/GT 이전 readout에는 주지 않는다.
기존3개 teacher 배열의 SHA가 동결된223번 대조군과 같아야 admission을 통과한다.

기존223의 계수1 mask를 재사용한다. 세seed 최종16장 mask를 모두 동결한 후
IDs1320,1322,..1350의 원래 native64 modal GT를 읽는다.1024 steps/settle512와
RNG/readout을 유지한다. source 대비 FG≥.01/CI하한>0이고 계수1보다 높은 FG면
확대 후보이며, 두 IoU 비감소면 balanced 후보로 기록한다. paired bootstrap은
RNG221/10000회다. source-center drift와 within/adjacent-class 통계도 함께 보고하되
모든 gating의 인과 기여나 최종 FG≥.90를 mini로 주장하지 않는다.

227번 전체320장 확인을 우선한다. 각227 seed가 최소한 한 번 실제 GPU allocation을
받은 기록이 생긴 후에만228 큐가 GPU를 요청한다. 그 뒤 빈GPU0..3을 독점 lease로
사용한다. 이 allocation 조건은 완료/성능 증거가 아닌 예약 우선순위다. Linux 각
프로세스의 THP만 끄고 전역 설정은 변경하지 않는다.70,000장 재학습은 없다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `a8babcc2de393c3689131ba83ce1c29d3b7a6c69ea506795600f66229f1f3ed9`

````markdown
# 동료 브랜치 확인

이번 공유 전 `origin/patch_v2`를 fetch했고 최신 commit은
`7a21632b93805f4f1da69a2e77a9ddd95c624d0a`로 이전 확인 이후 변경이 없었다.
새 native64 개선법을 가져왔다고 주장하지 않는다. 동료의 세seed 재검증에서
단일조건 개선이 유지되지 않았다는 기록을 고려해, 이번 loss 비교도 같은 초기
모델·데이터·readout·예산의 세seed와 동결된223 대조군을 유지한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `3ffbbc56d981f765fa2132eca8dfa685226823365d3d254ebc74b6178acf59fe`

````markdown
# SW0228 COMPLETE: center16 loss rejected for FG improvement

Fixed16/three-seed mean .844707/.603466/.712916 vs matched center1 .859634/.645611/.706340.
FG candidate gates failed; stronger center anchoring limited within-object denoising.
All144 masks/checkpoints/teachers independently audited. No full320 promotion or70k expansion.
See RESULT_KO.md. Earlier launch/audit preparation notes follow.

# SW0228

Source-center loss weight16 vs frozen223 weight1; same TRAIN64/128updates/joint encoder+projection.
All3 actual CPU admissions passed,14 Linux tests and3 CLI checks passed.
Three queues wait until each SW0227 seed has received its first GPU allocation.
No GPU training/performance result at this snapshot. See START_KO.md and DESIGN_KO.md.

Post-score independent verification is prepared in `completion_audit.py`.
It checks the three committed encoder/projection/Adam checkpoints, immutable
source teachers and their center16 loss decomposition, matched center1 control,
and all 144 candidate/control/baseline masks before independently recomputing
metrics, paired confidence intervals and candidate gates. It performs no new
model forwards or training. Four audit tests passed locally and on Frontier,
including rejection of altered intervals/nonfinite scores and correct handling
of an IoU regression or failure to improve over center1. This preparation is
not a completed performance result; the full320 incumbent remains unchanged.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `5fc2c80c092d3b6d0b45cddbec06a8ad90ae186d36f69186743fa0e342ea1766`

````markdown
# SW0228 완료: 물체별 주파수 중심 유지 강화는 FG 개선에 실패

같은 초기 모델·TRAIN64·128 Adam updates·batch16·학습률·encoder/core 입력
projection 공동학습을 유지하고, source object-center MSE의 계수만 1에서 16으로
높였다. graph와 나머지 core/gates/readout은 고정했다. mask supervision을 사용한
소규모 비교이며, 평가 조건은 native64 고정 16장(IDs1320,1322,..1350)·세 seed다.

| 조건 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 모델 | .842891 | .600692 | .702161 |
| 중심 계수1 공동학습(223번) | **.859634** | **.645611** | .706340 |
| 중심 계수16 공동학습(228번) | .844707 | .603466 | **.712916** |

계수16−계수1의 FG 차이는 −.014926, paired 95% CI는 [−.037458,.002309]다.
Foreground IoU 차이는 −.042144, CI[−.089370,−.000895]로 감소했고,
Object IoU 차이 +.006576은 CI[−.015462,.031895]가 0을 포함한다.
원래 모델 대비 FG 차이 +.001817도 CI[−.001534,.005158]로 개선이 확실하지 않다.
사전 등록 FG/balanced 후보 기준 모두 실패했다. 이 계수16 설정을 전체320장이나
70,000장 학습으로 확대하지 않고 FG 개선 경로에서 종료한다.

| seed | 계수1 FG-ARI | 계수16 FG-ARI | 계수16 FIoU | 계수16 Object IoU |
|---|---:|---:|---:|---:|
| 0 | .927862 | .927293 | .867387 | .757920 |
| 1 | .820461 | .817463 | .498243 | .704661 |
| 2 | .830578 | .789365 | .444769 | .676168 |

실제 주파수 진단에서는 source 물체별 중심의 이동 MSE가 세 seed에서
.00000545/.00000790/.00000356으로 작았다. 그러나 물체 내부 주파수 분산은
원래 모델 대비 약 10.3%/3.8%/2.8%만 감소했다. 계수1에서는 같은 분산이
.003360/.013505/.005320이었지만 계수16에서는 .004543/.016779/.006822로
더 컸다. 중심 유지 제약이 내부 정렬까지 제한한 것으로 해석할 수 있다.
중심 간 거리를 유지하는 것만으로 mask 성능이 개선된다는 가설은 지지되지 않았다.
이 진단은 개별 gating의 인과적 기여나 보편적인 최적 loss 계수를 증명하지 않는다.

세 checkpoint의 encoder·실제 32-weight projection·Adam 128회, teacher의
원래 중심/가중치/gamma 및 TRAIN label, 계수16 loss 분해, control과 예측 protocol을
확인했다. 모든 protocol을 확인한 후 DEV GT를 읽고 총144개 mask를 독립 재채점했다.
metric/평균/CI 최대 오차는 1e-12 미만이며 로컬 재계산 오차는 0이었다.
검증 때문에 추가 학습이나 GPU forward를 실행하지 않았다.

반복 사용한 validation 이미지에 고정된 모델들의 paired bootstrap이며 seed 모집단의
불확실성을 추정한 것은 아니다. 전체320장 최고 기록 .848615/.592533/.672047은
유지한다. 223/225번 후보의 전체320장 확인은 별도 227번에서 진행한다.

근거: `results_archive/completed_v1/summary.json`, `completion_audit.json`,
`archive_receipt.json`. 원본 데이터·checkpoint·이미지별 벡터는 공개하지 않는다.

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `ac99a755e62276bf7fb7602473084797b412eddd554bc6622ca04f315e83d82b`

````markdown
# SW0228 예약: 원래 주파수 중심 보존 loss 비교

로컬14개·Linux14개 테스트와3개 CLI 검사를 통과했다. uniform center drift의
loss/gradient만16배로 강해지고, 중심을 움직이지 않는 내부 편차에 대한 계수는
유지되는 것을 확인했다. class ID를 바꿔도 loss가 같으며 임의의 class/channel
순서를 만들지 않는다. 기존 source teacher3개 배열이 동결된223번 대조군과 같고,
세 실제 source 모델의 projection gradient가 유한·양수이며 초기 자연 분산을
1.05e-9 이내로 재현함을 CPU에서 검증했다.

세 seed supervisor는 실제 살아 있고 waiting_for_full320_first_attempts 상태다.
**아직 GPU 학습이나 성능 결과는 없다.** 227번의 각seed가 첫 GPU 배정을 받은
기록이 생긴 뒤에만228 큐가 빈 GPU0..3을 요청한다. 기존 전체320장 확인을 먼저
배정하면서 이후 빈 장치를 자동으로 사용할 예약이다. 이 조건은227 완료나 성능을
의미하지 않는다. 외부 GPU 작업이 나타나면 본인 worker만 양보한다.

223번과 같은 초기 모델·TRAIN64·128 Adam updates·batch16·LR/clip·encoder 및
32-weight core projection을 유지한다. 바꾸는 것은 source-center MSE의 계수1→16뿐이다.
class_labels는 TRAIN teacher에만 추가하며 GT 없이 예측하는 readout에는 주지 않는다.
다른 core/graph/gates/statistics, native64/RNG/readout/DEV16은 그대로다.

mask를 모두 동결한 후 세 지표와 source-center drift/within/인접class 통계를 함께
확인한다. 계수 sweep이나70,000장 재학습은 이번에 하지 않는다. 목표는 전체320장에서
세seed 평균이 같은 Slot을 세 지표 모두 넘고 FG≥.90인지 검증하는 것이며, 이번
mini/CPU gradient 검사로 그 목표나 모든 gating의 인과 기여를 달성했다고 보지 않는다.

````

</details>

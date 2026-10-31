# SW_0227_native64_joint_projection_full320

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `bd7ef8f599ccc1d4669810f573aa55d7b5c8d06418da5c9da5c7e4905ccc0b93`

````markdown
# SW0227: 고정 공동학습 모델의 전체320장 확인

225번의 사전 정의된 balanced 확대 기준을 통과한 223번 seed0/1/2 checkpoint를
고정한다. 225번 64장 평균은 .877164/.597239/.670309이고 FG 개선 CI는 양수다.
원래16장 gate 실패 및64장 선택 이력을 유지한다. 추가 학습은0회다.

canonical128/no crop, native64, IDs1320..1639, 원래2px modal GT, 1024 steps와
settle512, RNG193000+offset 및 원래 spike/membrane readout을 유지한다.
225번의 첫64개 mask와 frequency를 hash 검증 후 재사용하며 나머지256장을
새로 예측한다. offset0을 추가로 실행해 mask와 frequency 모두 bitwise 재현해야
진행한다. checkpoint/전체 RGB/64장 prefix/원래16장/control baseline을 CPU에서
검증한다. Linux 큐와 worker에만 THP를 끄고 전역 설정은 변경하지 않는다.

세320장 protocol과 원래 comparable Slot 세 seed의 prediction/protocol/hash/IDs를
모두 검증한 후 GT를 읽는다. Slot은 기존154/210의 우리데이터 학습 reference이며
공개 checkpoint의 다른 계약 .8946 수치와 혼동하지 않는다. source 및 Slot 재계산
평균은 기존210의 전체320장 집계와1e-12 이내로 같아야 한다. paired image
bootstrap10000회 RNG227로 candidate−source 및 candidate−Slot을 함께 기록한다.
같은 전체320장 계약에서 세seed 평균이 Slot을 세 지표 모두 엄격히 넘는지와
FG≥.90 여부를 별도 판정한다. seed 모집단의 CI나 untouched test라고 주장하지 않는다.

70,000장 재학습은 없다. mask-supervised lineage와 다른 core/graph/gate 고정 상태를
유지하며 모든 gating의 인과 기여나 data scaling을 이 확인만으로 주장하지 않는다.
226번 TRAIN64→512 비교는 별도로 계속한다. GPU0..3을 독점 shared lease로 예약하고
foreign compute가 나타나면 본인 worker만 yield한다. 기존 작업을 중복 등록하지 않는다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `8c6c2d6610b2719ab105dcb03dd0738653ae31937415f30541bc2905ed7bea36`

````markdown
# 동료 브랜치 확인

이번 공유 전 `origin/patch_v2`를 fetch했다. 최신 commit은
`7a21632b93805f4f1da69a2e77a9ddd95c624d0a`이며 이전 확인 이후 새 변경은 없었다.
해당 기록은32×32의8-epoch 개선이 세seed에서 유지되지 않아 주장을 철회한 내용이다.
새 native64 개선법을 적용했다고 주장하지 않는다. 이번 전체320장 확인도 최고 seed만
고르지 않고 고정된 세 checkpoint와 같은 Slot 대조군으로 판정한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `32b43df8386d0e81e94979b34c133025ba659d5cae5787c5f4801beb0196f892`

````markdown
[Completed full320 result](RESULT_KO.md): .858916 / .605286 / .674313, independently audited. FG090 pending.

# SW0227

Fixed SW0223 checkpoints selected by audited SW0225; full320, no training.
All3 actual CPU admissions and Slot controls passed; three exclusive GPU queues registered.
Reuses64 masks/seed, evaluates256 new images plus exact replay.
No new performance result. See START_KO.md and DESIGN_KO.md.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `a3d288adfd41baa6c39a162fdc088f26557a9b2ce36d03adc0f332e97ad7902b`

````markdown
# SW0227: native64 전체320장 최고 FG 기준점 갱신

고정 320장(IDs1320–1639), native64 modal patch GT, seed0·1·2 평균이다. SW0223의 고정 checkpoint를 사용했고 추가 학습은 없다. 기존 64장 mask를 재사용하고 seed별 256장을 추가 생성했다. 실제 source checkpoint와 mask/frequency 재현 검증을 통과했다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 이전 기준 SW0210 | .848615 | .592533 | .672047 |
| SW0227: encoder + gamma projection 학습 모델 | **.858916** | **.605286** | **.674313** |
| 같은 계약의 Slot Attention | .786902 | .198201 | .195203 |

이전 기준 대비 FG 차이는 +.010301, paired image bootstrap 95% CI [.005659,.014751]이다. Foreground IoU 차이는 +.012753, CI [.001770,.023733]이다. Object IoU 차이는 +.002266, CI [-.002547,.006817]이므로 Object IoU 개선은 아직 확정하지 않는다. CI는 고정 세 seed에 조건부인 image bootstrap이며 seed 모집단 불확실성은 아니다.

| seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 0 | .882964 | .804963 | .693819 |
| 1 | .839643 | .529764 | .667478 |
| 2 | .854141 | .481131 | .661643 |

원시 mask 2,880개(candidate·이전 기준·Slot)를 독립 재계산했다. 집계·CI 오차는 5.56e-16 이하이고, 세 지표 평균 모두 Slot을 엄격히 초과한다. 각 seed가 개별적으로 FG에서 Slot을 이긴다는 뜻은 아니다. mini16에서 실패했던 기존 gate 기록은 보존하며 full320 근거와 구분한다.

이 결과를 고정 전체 평가의 새 FG 기준점으로 등록한다. FG .90 목표는 미달성이다. mask-supervised lineage이며, 새 native64 70,000장 학습 결과나 동일 supervision/cost의 unsupervised 비교가 아니다. 반복 사용한 validation으로, untouched test 성능이라고 부르지 않는다. 모든 gating의 기여나 데이터 증가 효과는 이 결과만으로 입증되지 않는다.

232번은 이 실제 checkpoint에서 추가 학습이 효과가 있는지 별도로 검증 중이다. 원본 최고 모델과 이번 checkpoint는 모두 보존한다.

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `28b51d816c47bc00d580af0e21e7a487b61e724b9169a347c08dee146a1eb093`

````markdown
# SW0227 시작: 같은 checkpoint의 전체320장 확인

로컬9개·Linux9개 테스트, 3개 CLI 검사를 통과했다. 실제 comparable Slot 세 seed의
예측·protocol·hash·IDs 및 기존210 reference를 GT 없이 검증했다. 세223 checkpoint를
실제 CPU에서 복원해 전체320장 RGB와 원래16장 및225의64장 prefix를 검증했다.
CUDA 초기화와 optimizer update는 CPU admission에서 모두0이다.

세 seed 큐를 등록했고 관측 당시 seed0/2 worker는 GPU1/3에서 실행 중이었다.
각64장 캐시가 검증·재사용됐고 mask와 frequency의 추가 bitwise replay를 통과한 뒤
나머지256장씩 예측한다. seed1은 독점 GPU를 기다린다. 다른 사용자의 GPU 작업은
공유하거나 종료하지 않는다. 226번 데이터 증가 비교는 별도로 유지한다.

이번에는 새로운 학습이 없다. 223/225에서 선택한 고정 모델을 전체320장·세seed에서
기존 모델과 같은 Slot reference와 비교한다. 재계산된 source 및 Slot 평균은 원래
210 reference와1e-12 안에 같아야 한다. 세 지표 모두 Slot 평균 엄격 초과와 FG≥.90을
별도 판정한다. 이번 시작 기록에는 성능 결과가 없고 전체320장 incumbent는 유지한다.

225의64장 평균 .8772를 전체320장 결과라고 취급하지 않는다. 기존16장 gate 실패,
반복 validation 선택 및 mask-supervised lineage도 유지한다. 추가70,000장 학습이나
모든 gating의 인과 기여를 이번 테스트로 주장하지 않는다.

````

</details>

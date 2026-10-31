# SW_0183_shared_prefix_unique_scaling

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `0c91eef908bc7e4129a6b0f068fb8b0cf9013c16a91da0490532619f128e5b8c`

````markdown
# SW0183 — 같은 현재 학습 상태에서 반복 데이터와 새 데이터 비교

## 근거와 범위

SW0182는 과거 SW0181의 proposal2와 제안된 모델/Adam 해시가 달라 기술 종료했다.
실패 기록과 등록된 검사 코드는 보존한다. 동일한 저장 상태에서 실행한 seed1의
네 replay(control512 두 번, candidate1024 두 번)는 Q/Z, clip 전후 gradient,
제안된 모델/Adam tensor가 모두 정확히 같았다. 양쪽 입력의 실제 B16 RGB/GT도
같았다. 따라서 현재 새 입력 경로가 차이를 만들었다는 증거는 없다.
과거와 현재의 runtime 기록과 Q/Z도 같았지만 일부 gradient norm과 parameter
변화 norm에 미세한 차이가 있었다. 과거 proposal2의 전체 tensor snapshot은
없으므로 그 차이의 원인을 확정하지 않는다. 추가 과거 hash 추적을 선행조건으로
삼지 않고, 현재 같은 prefix에서 출발하는 새 비교를 등록한다.

이 실험은 pretrained SourceT의 mask-supervised 추가 학습이다. Fresh native32
70k 학습, unsupervised 개선, 모든 gate의 인과적 기여를 입증하는 실험은 아니다.
SW0181/SW0165와 Slot 기준점은 보존한다.

## 데이터와 두 arm

SW0182의 고정 TRAIN1024 cache와 기존 SW0171/SW0181의 TRAIN512 cache를 그대로
사용한다. 첫512 ID/RGB/modal GT의 byte-exact 일치와 새512의 canonical TRAIN
정렬을 기존 검증과 실제 cache hash로 확인한다. VAL1320..1639 및 reserve는
학습에 사용하지 않는다. 데이터 생성·foreground·loss·patch 계약은 바꾸지 않는다.

각 source seed 0/1/2는 같은 fresh SW0097 SourceT+zero-context와 빈 Adam에서
시작한다. 기존512의 첫 permutation으로 32 proposals를 한 번 실행한다.
그 prefix32 경계의 모델·Adam·RNG·accepted counter를 보존하고 두 arm으로 복제한다.

- `control512`: 기존 두 번째 permutation의 기존512 이미지를 사용해 32 proposals.
- `candidate1024`: 같은 permutation index에 +512를 적용해 새512로 32 proposals.

각 arm은 공통 prefix를 포함해 논리적으로 64 proposals, B16×64=1,024 image
exposures를 가진다. 실제 실행은 공통32+control32+candidate32=96 proposals/seed다.
Fork 때 Adam을 새로 만들지 않으며, 두 arm의 시작 모델·Adam·RNG digest와
논리 cursor32가 정확히 같아야 한다. 과거 SW0181의 proposal hash는 참고 기록이며
이번 실험의 실행 허용 조건으로 사용하지 않는다.

## 학습·재개·비용

32×32 patch, T1024/settle512/live-tail64, B16/micro4, product-Q LOO class-balanced
CE temp0.1을 유지한다. Context encoder 3e-4, graph/native-core 각3e-5,
integration1e-3, global clip1과 projection, full-forward actual-descent acceptance
및 거절 시 model/Adam/RNG 복원은 SW0181/182와 같다.
각 논리 proposal의 RNG는 171000+source_seed+zero_based_proposal_index다.
Prefix와 각 branch checkpoint를 별도 경로에 보존하며 원래 physical GPU/runtime
binding으로 재개한다. 유휴 GPU만 독점 사용하고 외부 작업이 나타나면 자기 worker만
양보한다. Accepted 수를 맞추기 위한 retry나 추가 proposal은 없다.
실제 gradient/post/pre forward, image exposures, committed Adam step, 공통 prefix
연산 공유량을 따로 보고한다. Prefix의 첫 두 proposals에서 실제 emitted-S gradient와
네 family의 gradient/변화, no-grad parity를 확인한다.

## 평가·판정·중단

두 arm×세 seed의 실제 S→native Q→production QCC→SW0165 standalone 예측과
protocol을 모두 확정한 뒤 고정320 native32 GT를 읽는다. Conditional endpoint는
별도 진단이며 standalone 결과와 혼합하지 않는다.

고유 데이터 이득은 candidate의 seed 평균 FG가 동시대 control보다 높고,
동일 이미지별 세-seed 평균 차이의 paired bootstrap 95% CI 하한이 0보다 클 때만
인정한다(320 images, RandomState178, 10,000 resamples). Slot 대비 FG와 두 IoU의
엄격 초과 및 FG paired CI, SW0181/SW0165 대비 차이, FG0.87 여부도 별도로 보고한다.
둘 중 성능이 좋은 모델은 보존하되, paired gate가 실패하면 scaling 이득으로
주장하지 않고 이 recipe를 종료한다. LR/온도 rescue sweep은 하지 않는다.

배포 전에 실제 CLI/CPU source/cache/fork/optimizer/checkpoint 검사를 수행한다.
연결 관측 실패만으로 작업을 재시작하지 않는다. 모든 결과와 실패를 이 폴더에
기록하고 연구 브랜치에 공유한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `581687d66cbe0997debcc8913b0f1dfee387ab6275d5215a9b7e5370a41aadb7`

````markdown
# SW0183 — completed shared-prefix unique-data comparison

All training, six arm predictions, GT-last scoring and independent audit are complete.
The 1024-image candidate scores **.810934 / .608932 / .661713**; the matched
512-repeat control scores **.813732 / .577707 / .665824**. Both exceed Slot,
but the registered unique-data gate fails. SW181 remains the supervised
three-seed FG incumbent; no native70k or gate-causality claim is made.

See [full results and limits](RESULT_KO.md) and [preregistered design](DESIGN_KO.md).
Actual server validation passed 29 tests, actual executable CLI checks,
first-two GPU spike/gradient/Adam admission, and independent final metrics.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `ecaf062c454daff33ec69a0238ce47b005a2e5c1c45d76e2670e6193df09151c`

````markdown
# SW0183 완료: unique-data 이득은 확인되지 않음

고정 native32×32, 동일 320장, seed 0/1/2 평균입니다. TRAIN modal mask를
사용한 supervised 추가 학습이며, 70,000장 신규 학습 결과가 아닙니다.

| 모델/조건 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 512장 반복 control | 0.813732 | 0.577707 | 0.665824 |
| 1,024장 candidate | 0.810934 | 0.608932 | 0.661713 |
| 기존 supervised SW181 | 0.814836 | 0.589302 | 0.671681 |
| 기존 unsupervised SW165 | 0.804299 | 0.609975 | 0.651149 |
| Slot Attention, 우리70k 학습 | 0.781652 | 0.204360 | 0.203165 |

512/1,024는 동일 32-proposal prefix의 실제 model/Adam/RNG를 분기했고,
각 조건은 B16×64=1,024 proposal-image 노출로 맞췄습니다. 실제 accepted
updates는 control 35/39/40, candidate 31/37/34로 다릅니다. 따라서 동일
proposal/노출 budget 비교이며 동일 accepted-update 수 비교는 아닙니다.

1,024−512 FG-ARI는 −0.002798, paired-image bootstrap 95% CI
[−0.006642, +0.001133]입니다. 증가 이득이 입증되지 않았고, 통계적으로
확정된 성능 하락이라고도 단정하지 않습니다. Slot 대비는 +0.029282,
CI [+0.010226,+0.047724]이며 세 지표 모두 초과합니다. .87은 미달입니다.

SourceT foreground를 고정한 conditional FG는 0.802937→0.803142로 거의
변하지 않았습니다. standalone foreground IoU 증가는 주로 seed0에서
배경 오검출이 줄어든 것과 동반됩니다. 이것이 FG-ARI 개선으로 직결되지는
않았습니다. seed0 control FG 0.864375는 이번 실험의 단일 seed 최고이나,
세 seed 평균 최고 SW181을 교체하지 않습니다.

독립 검산은 여섯 실제 prediction archive와 각 conditional/standalone의
3,840 image endpoint를 사용했습니다. NumPy contingency ARI/IoU와 SciPy
Hungarian 결과의 원 점수 최대 차이는 3.34e−16 미만입니다. GT modal
변환, 모든 평균·paired bootstrap, 공통 prefix history/model/Adam/RNG와
첫 분기 transaction 동일성도 검증했습니다.

다음 방향: 같은 QCE를 더 반복하는 rescue sweep 대신, QCE가 학습하는
상대적인 class-affinity와 실제 QCC/foreground readout이 필요로 하는
absolute 연결 강도의 불일치를 TRAIN-only 고정 probe에서 먼저 측정합니다.
관측 근거를 얻은 뒤 readout에 맞는 loss를 등록합니다. validation에서
threshold를 검색하거나 기존 183번 소스를 수정하지 않습니다.

원본: [점수](results_archive/score_v1/evaluation.json),
[독립 검산](results_archive/score_v1/completion_audit.json),
[검산 소스](completion_audit_source.py).

````

</details>

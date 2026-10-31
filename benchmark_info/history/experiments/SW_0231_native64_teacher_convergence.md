# SW_0231_native64_teacher_convergence

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `6c122f4b3badc861b2a789ccf0d9551e42be56d9ecb8062037f193ea3d5b7c91`

````markdown
# SW0231: 검증된 원래 loss의 학습 예산 확인

228/230번 간격 보존 loss는 종료한다. 223번 원래 teacher denoising/gamma anchor,
초기 모델·TRAIN64·순서·LR/clip·batch16·encoder+실제 core projection 학습을
유지하고 고정 업데이트 수만128에서512로 늘린다. graph/나머지 core/gates/readout은
고정하며 TRAIN mask supervision을 사용한다. 전체70,000장 재학습은 아니다.

223번 실제128-update history에서 마지막8 epoch의 평균 TRAIN loss가
각각 약4.10%/1.92%/1.10% 감소했다. 이는 아직 objective가 줄어든다는 근거이며
최종 mask 성능이나 데이터 scaling 개선의 증명은 아니다. 추가 예산이 실제
generalization에 도움이 되는지 한 번만 확인한다. DEV로 최적 step을 선택하거나
여러 epoch/LR를 sweep하지 않는다.

첫128 updates를 실제 재현하고 encoder/projection/Adam/history/Torch RNG가
수락된223번 checkpoint와 bitwise 같은지 확인한 후384 updates를 더 진행한다.
동일성 실패 시 추가 업데이트와 native mask 평가를 진행하지 않는다. 실제128회
snapshot/proof를 별도 보존한다. resume 시 검증된 checkpoint/Adam/RNG를 복원한다.
수치상 같은128회 prefix를 재현하는 계산은 포함되므로 추가 계산 비용을 숨기지 않는다.

평가는 기존 고정16장·세 seed·원래 spike mask/RNG로 진행하며 실제223번128회 모델을
대조한다. baseline FG>=+.01/paired CI 하한>0 및128회 모델 초과를 FG 후보 기준,
FIoU/OIoU 평균 비감소를 balanced 기준으로 유지한다. 모든 결과는 mini로 기록하고
전체320장·.90 달성으로 승격하지 않는다. 실패 시 이 예산 확인 분기는 종료한다.

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

Source SHA256: `d06ec26ef6b05c4b6d5673b61c2118c21a528030531bd082bedfc92c4b2bc293`

````markdown
# SW0231: original teacher objective convergence check

One fixed512-update TRAIN64 encoder+projection pilot vs accepted SW0223 at128.
Same objective/source/order/LRs; explicit bitwise model/Adam/history/Torch RNG
prefix proof before update129 and before final prediction. All masks use original
native64 spike readout on fixed16/three seeds. No performance result yet.
See DESIGN_KO.md. Supervised TRAIN targets; no new70000-image training.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `cb236ed80432cf7c20d2ef43b99ea3c15a1741c52accebc3faecfbd9d6d6e092`

````markdown
# SW0231: 정확한 128회 prefix 검증 실패

seed0·2의 재계산은 128회 경계에서 accepted SW0223와 bitwise 동일하지 않아 중단됐다. Projection 최대 절대 차이는 각각 1.1920929e-7, 2.9802322e-8, encoder는 8.9406967e-8, 4.4703484e-8이다. 최초 history 차이는 두 seed 모두 update2의 gradient norm에서 나타났다. 수치 차이의 근본 원인은 확정하지 않았다.

검증 허용 오차를 완화하지 않았다. seed1도 같은 재계산을 반복하지 않도록 해당 실험의 프로세스만 종료했고, 세 supervisor가 모두 종료됐음을 서버 /proc에서 확인했다. 실패 checkpoint와 로그는 보존했다. 512회 학습이나 mask 평가 결과는 없으며 모델 성능 실패로 해석하지 않는다.

후속 SW0232는 accepted SW0223의 실제 가중치·Adam·history·Torch RNG를 직접 복원한다. 복원한 128회 상태를 동일한 bitwise 검사로 확인한 뒤 384회만 추가 학습한다. 기존 SW0227 전체 평가와 원본 checkpoint는 유지한다.

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `282704c048ec9a725246491e9d05d808ea63c3f4f37664927ea71793eb211fd4`

````markdown
# SW0231 서버 검증 시작

128회 종료 당시에도 원래 공동학습 loss가 계속 줄어든 기록을 근거로, 같은 loss의
고정512회 업데이트를 한 번 비교한다. loss/데이터/초기 모델/학습률/graph/gates/
readout을 바꾸지 않는다. 같은 TRAIN64·batch16·실제 encoder+projection 학습이다.
TRAIN mask supervision을 사용하며 70,000장 추가 학습이나 무감독 결과는 아니다.

첫128 updates가 실제223번 모델의 encoder/projection/Adam/history/Torch RNG와
같은지 검증하고 snapshot/proof를 보존한다. 다르면 이후 업데이트·native 평가를
진행하지 않는다. 통과하면 같은 Adam 상태에서384 updates를 더 수행한다.
prefix를 다시 계산하는128회 비용도 포함한다. 고정16장·세 seed 평가에서 실제
223번128회 모델과 비교하며, 전체320장 기록은 별도227번이 확인한다.

로컬13개 테스트를 통과한 코드를 배포했다. 서버에서도13개 테스트와
train/evaluate/queue CLI 검사3개를 통과했고 실제 세 seed CPU 검사를 시작했다.
모든 CPU 검사에 통과해야 독점 idle GPU 큐를 등록한다. 이 문서는 서버 검증 시작
시점의 기록이며 GPU 학습 완료나 성능 개선을 주장하지 않는다.

서버 launcher PID2799508, CPU 검사 PID2799562/2799563/2799564.
배포 SHA는 START_EVIDENCE.json, 수락된223번 학습 기록 분석은
TRAIN_LOSS_DIAGNOSIS.json에 기록한다. 동료 브랜치는 이번에도5c88a6f로 변동이
없음을 fetch로 확인했다. 기존227번은 중단·재시작하지 않는다.

````

</details>

# SW_0230_native64_relative_separation

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `5c2db33489c6778bf228dccb44e65807a04f8deaaaa7a1e086f0a32bc790bbe4`

````markdown
# SW0230: source-relative 물체 간 주파수 간격 유지

223/225번은 내부 정렬을 개선했지만 229번에서 일부 false merge 증가가 관찰됐다.
228번의 절대 중심 고정 강화는 내부 정렬까지 약하게 만들어 FG 개선에 실패했다.
이번에는 원래 계수1 teacher denoising/gamma anchor를 유지하고, TRAIN 물체별
주파수 중심 사이 거리가 source보다 줄어드는 경우에만 부드러운 손실을 추가한다.
절대 중심 이동·공통 평행이동·거리 확대에는 추가 패널티가 없고 임의 unit margin도 없다.

각 이미지에서 존재하는 모든 class(BG 포함)의 unordered pair를 같은 비중으로
평균한다. d는 4개 frequency component의 평균 제곱 거리이며 추가 항은
ReLU(d_source-d_current)^2 / max(d_source,float32 epsilon)다. 계수는1로 고정한다.
원래 teacher/gamma 항도 유지하므로 전체 loss가 공통 이동에 불변인 것은 아니다.

같은 초기 모델·TRAIN64·128 Adam updates·batch16·LR/clip·encoder와 실제
4x8 projection 공동학습을 유지한다. graph/나머지 core/gates/readout은 고정한다.
TRAIN supervision만 사용하며 DEV는 고정16장·세 seed·원래 spike mask/RNG다.
대조군은 실제223번 계수1 모델이다. FG>=+.01/paired CI 하한>0 및223번 초과를
FG 후보 기준으로, FIoU/OIoU 평균 비감소를 balanced 기준으로 유지한다.
실패하면 이 설정을 종료하며 계수/LR/epoch sweep은 하지 않는다. 성공해도
전체320장 확인 전에는 최고 기록이나 .90 달성으로 승격하지 않는다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `e67c16714b3088bf5ca774a82e57603f4abc1a296e9b27616ca651453b7d095c`

````markdown
# 동료 연구 확인

origin/patch_v2 commit 5c88a6faa3c7f3a63f2d2223303c2a74ae1d45db,
PV2_0103_seed_rescue의 README/commands/metrics 변경을 확인했다.
동료의 contract-v2 native32 seed1에서 graph 고정 8epoch는
.6777/.6895/.6357, 비고정16epoch는 .6867/.6008/.4887이었다.
graph 고정은 두 IoU와 비용에서 유리하지만 FG는 .0090 낮다.
따라서 세 지표 모두 우월하다고 해석하지 않는다. seed2 고정 측정 전의
예상 평균도 실측으로 취급하지 않는다. 이 계약에는 Slot 대조군이 없다.

우리 쪽은 이미 graph를 고정하며 실제 세 seed와 세 지표를 함께 확인한다.
공유 graph/seed 안정성 및 단일 지표 선택 오류의 교훈을 유지한다.
동료의 다른 데이터·해상도 점수를 우리 최고 기록으로 섞지 않으며,
이번 상대 주파수 손실은 우리223/225/228/229 진단에서 도출한 별도 가설이다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `e18cdad2810e5298f201d41be8373dd129cfd97af72be7b31ae0b03e7687a428`

````markdown
# SW0230 COMPLETE: relative separation rejected

Fixed16/three-seed .852402/.624603/.711310 vs matched SW0223 .859634/.645611/.706340.
FG paired95% CI is strictly negative; candidate gates failed. All144 masks independently audited.
No full320 or70k expansion; full320 incumbent unchanged. See RESULT_KO.md.

Earlier preparation notes follow.

# SW0230: source-relative class-center separation

Prepared matched TRAIN64/128-update encoder+projection comparison against SW0223.
Original center weight1 retained; add a squared one-sided source-distance deficit,
normalized by source distance. No new absolute margin or center16 anchoring.
No model performance result yet. Frozen16/three-seed admission and candidate gates
are specified in DESIGN_KO.md. Source targets use TRAIN masks; not unsupervised.

Independent post-score audit is prepared in `completion_audit.py`: committed
encoder/projection/Adam state, immutable teacher/control, original-plus-relative
loss decomposition and all144 masks are checked before independently scoring
the fixed16 cohort and reproducing paired intervals/candidate gates. No extra
model forward or training is required. Four audit tests passed locally and on
Frontier, including invalid CI/NaN rejection and separate FG/balanced gates.
This preparation does not establish improved model performance.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `35ce2eac145bf40b3c984a4105f468468ecb80b5f0be770fccb5234e5971830d`

````markdown
# SW0230 완료: 상대 주파수 간격 유지도 FG 개선에 실패

228번의 절대 중심 유지 대신, 223번의 원래 teacher denoising/gamma anchor를
유지하면서 TRAIN 물체별 중심 간 거리가 source보다 줄어들 때만 상대 패널티를
추가했다. 같은 초기 모델·TRAIN64·128 Adam updates·batch16·LR/clip·실제
encoder+4x8 projection 공동학습을 사용했다. graph/나머지 core/gates/readout은
고정했다. mask supervision을 사용하는 native64 고정16장·세 seed 비교다.

| 조건 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 모델 | .842891 | .600692 | .702161 |
| 원래 공동학습(223번) | **.859634** | **.645611** | .706340 |
| 상대 간격 패널티 추가(230번) | .852402 | .624603 | **.711310** |

230−223의 FG 차이는 −.007232, paired95% CI[−.014287,−.001225]로 감소했다.
FIoU 차이는 −.021008, CI[−.064246,.011209], Object IoU 차이는 +.004969,
CI[−.007752,.018735]다. IoU 변화의 불확실성을 유지하며 FG 감소를 분명히 기록한다.
원래 모델 대비 FG +.009511도 CI[−.007517,.034865]가 0을 포함한다.
사전 등록 FG/balanced 후보 기준 모두 실패했다. 이 설정의 전체320장/70,000장
확대 및 계수·학습률·epoch sweep은 진행하지 않는다.

주파수 진단상 인접 class 중심 간 평균 제곱 거리는 세 seed에서
.087482/.156513/.050628로 원래 source의 .087986/.162614/.053982에 가깝게
유지됐다. 그러나 물체 내부 분산은 .003571/.015140/.006245로, 223번의
.003360/.013505/.005320보다 모두 컸다. 중심 거리 수축을 덜어도 내부 정렬과
최종 FG 개선을 동시에 얻지는 못했다. 228/230번 결과는 class 평균의 이동이나
수축만을 억제하는 접근을 우선순위에서 내릴 근거이며, source 주파수 간격이
실제 spike assignment에 최적이라는 가설을 지지하지 않는다. 개별 gating의
인과성이나 다른 구조에서의 가능성을 부정하는 결과는 아니다.

세 실제 encoder/projection/Adam checkpoint의128회 업데이트, 나머지 core의
불변성, source teacher/대조군/모든 mask protocol과 loss 분해를 확인했다.
모든 protocol 확인 후 DEV GT를 읽고144개 mask를 독립 재채점했다.
독립 metric/평균/CI 오차는 1e-12 미만, 로컬 재계산 오차는0이었다.
추가 학습이나 GPU forward는 없었다. 고정 모델에 조건부인 이미지 bootstrap이며
seed 모집단 CI나 untouched test 결과가 아니다.

전체320장 최고 기록 .848615/.592533/.672047은 유지한다. 223/225번 고정 후보의
전체320장 확인은 별도227번에서 계속 진행한다.

근거: results_archive/completed_v1/{summary.json,completion_audit.json,archive_receipt.json}.
원본 데이터·checkpoint·이미지별 벡터는 공개하지 않는다.

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `b814ef90ea2b6520bbf831432089ff2366972353bd15cdfc9cb6ebb8f0797713`

````markdown
# SW0230 서버 검증 시작

228번 계수16 중심 유지 설정은 종료했다. 이번에는 223번의 원래 계수1 teacher
denoising에 source-relative 물체 간 중심 거리 감소 패널티를 추가한다.
중심을 제자리에 고정하는 방식과 달리 중심 이동은 허용하고, 원래보다 가까워진
class pair에만 추가 패널티를 적용한다. 임의 단위 거리 margin은 사용하지 않는다.

같은 초기 checkpoint·TRAIN64·128 Adam updates·batch16·학습률/clip과
encoder+실제4x8 core 입력 projection 학습을 유지한다. graph와 나머지
core/gates/readout은 고정한다. mask supervision을 사용한다. 고정16장·세 seed
대조는 실제223번 모델과 진행하고, 세 지표 및 사전 등록 후보 기준으로 판단한다.

로컬 테스트15개를 통과한 버전만 서버에 보냈다. 서버에서도 테스트15개와
train/evaluate/queue CLI 검사3개를 통과했고, 세 seed의 실제 데이터·모델·teacher와
projection gradient CPU 검사를 시작했다. 이 검사가 모두 통과해야 GPU 큐가
등록된다. 이 문서는 서버 검증 시작 시점의 기록이며 GPU 학습 완료·성능 결과는 아니다.

서버 launcher PID2759128, CPU 검사 PID2759175/2759176/2759177.
배포 파일별 SHA는 START_EVIDENCE.json에 기록한다. 외부 GPU 사용자가 있으면
기다리고, 학습 중 외부 사용자가 나타나면 자기 작업만 양보하는 기존 큐를 유지한다.
기존227번 전체320장 평가는 중단하거나 재시작하지 않는다.

동료의 최신5c88a6f 결과는 PEER_REVIEW_KO.md에 검토했다.
이 실험 결과가 확인되기 전까지 전체320장 최고 기록은 .848615/.592533/.672047이다.

````

</details>

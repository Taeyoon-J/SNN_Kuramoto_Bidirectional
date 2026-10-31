# SW_0176_local_phase_credit

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `e8888b283313dabb92b0814d3ad58ed2fad007cb690aa15b42c2a3477f7f0a0f`

````markdown
# SW176: 전체 관측 구간에 local phase gradient 전달

## 근거와 가설

SW175 critical-component loss는 고정320의 conditional/standalone FG와 IoU 개선에 실패했다. SW173에서 full1024 gradient는 tail64 대비 매우 크고 방향도 달랐으며, SW174에서 event surrogate 경로만 제거해도 이 현상이 남았다. 이번에는 loss나 forward 구조가 아니라 긴 Kuramoto 재귀를 거친 gradient 전달 정책 하나를 바꾼다.

각 step의 Kuramoto `theta_prev`만 detach하고, 같은 drive/A/coupling과 원래 연산으로 theta output을 계산한다. history에 저장한 각 output은 live로 유지해 delayed gate에서 해당 step parameter로의 gradient를 남긴다. D/M/event/reset recurrence는 유지한다. 전체1024 frames를 grad-enabled로 계산해 마지막512 settled frames의 objective가 각 local phase step에 직접 gradient를 줄 수 있도록 한다. 이는 exact full gradient가 아닌 명시적인 approximate gradient다. 모든 긴 recurrence를 제거한다거나 효과가 입증됐다고 주장하지 않는다.

## 변경과 대조

forward, gate 구조, threshold, 실제 component spike, production Q, 실제 A와 learned coupling gain, 추론 readout을 바꾸지 않는다. supervised critical-component objective, negative Q 항, 기존 source checkpoint+seeded zero-final context, frozen base encoder, TRAIN512×동일 두 순서, B16/micro4, optimizer/LR/clip/projection,64 updates를 SW175 critical-component arm과 맞춘다. 후보는 fresh source에서 시작한다.

완료된 SW175 critical-component arm의 초기 state, source/preprocessing/frozen 구성, TRAIN 순서, optimizer 설정, update cursor/경계, RNG 조건을 대조군으로 재사용할 수 있는지 먼저 검증한다. 일치하지 않으면 fresh matched tail64 대조군이 필요하다. 학습 정책을 제외한 차이가 남으면 직접 인과 비교를 주장하지 않는다.

## 실행 전 검증

CPU graph 검사는 동일 forward, 이전 theta로의 gradient 차단, local parameter 및 delayed output으로의 gradient 보존, context manager 정상/예외 복원을 확인한다. 실제 TRAIN 캐시와 완료된 control provenance로 CPU admission을 통과해야 한다.

짧은 GPU preflight는 TRAIN의 첫 고정 이미지 한 장에서 parameter update 없이 원본과 후보의 six traces, actual Q/A, conditional/standalone masks 동일성을 검증한다. 전 과정에서 VAL/reserve 정답은 읽지 않는다. fullwindow gradient의 유한성, 실제 S와 context/graph/Kuramoto/integration 각 family의 credit, 시간과 peak memory를 기록한다. forward/provenance 불일치나 비유한 값은 구현/가설 문제를 해결하기 전 학습하지 않는다. 작은 진단 loss의 하락 자체를 학습 확대 조건으로 사용하지 않는다.

64-update 학습을 구현할 때 첫 두 committed updates에서 실제 finite gradient·parameter 변화와 메모리를 확인하며, 이를 별도의 disposable update로 중복 실행하지 않는다. 그래프의 한계로 B4가 실행되지 않으면 조건 변경을 숨기지 않고 새 설계를 등록한다. 현재 문서는 trainer 또는 GPU preflight가 실행됐다는 증거가 아니다.

## 평가와 종료

같은 native32 고정320, seed1의 conditional Source-T foreground와 standalone native QCC foreground를 사용한다. 두 모델의 모든 예측을 확정한 뒤에만 정답을 읽는다. FG-ARI / foreground IoU / matched-object IoU를 모두 기록하고 candidate FG가 SW164 seed1과 matched tail64 control보다 각각 높으며 paired bootstrap10000회(seed176)의 95% lower CI가 0보다 높아야 한다. 두 IoU 평균도 각 비교군 이상이어야 한다. 두 endpoint가 모두 조건을 만족할 때에만 seed0/2 및 더 큰 고유 TRAIN 데이터로 확장한다.

실패하면 이 정책을 종료하고 epsilon/LR/threshold/time-window 미세조정으로 구제하지 않는다. 성공해도 gating의 인과적 필요성, 세 seed 평균의 Slot Attention 초과, FG-ARI .87, native32 70k fresh training과 positive data scaling은 각각 별도 검증이 필요하다. 기존 최고 모델은 결과가 검증될 때까지 유지한다.

````

</details>

## EVALUATION_QUEUE_KO.md

<details>
<summary>EVALUATION_QUEUE_KO.md</summary>

Source SHA256: `057aea5228c20fae47e22c1ac7b118dd461aa778e74d5c2e69c8928b4870ef27`

````markdown
# SW176 자동 평가 등록

64회 업데이트 학습이 끝나면 동일한 고정 32×32 patch / 320장 / seed1 평가를 실행한다. 예측 부모 PID 228242, CPU 채점 부모 PID 228264의 실제 명령행과 대기 상태를 확인했다. 서버 CPU 검사 10개와 실제 대조군 예측 파일 일치 검사를 통과했다.

학습 종료 전이므로 최종 모델 CPU 로드는 아직 수행하지 않았다. 예측 부모가 완료된 학습 manifest와 checkpoint를 검증하고 실제 모델을 CPU에서 로드한 후에만 비어 있는 GPU를 확보한다. 기존 학습은 재시작하지 않는다.

후보의 conditional 및 standalone 예측을 모두 확정한 뒤 정답 mask를 읽고 세 지표를 계산한다. SW164 원본과 SW175 critical-component 대조군은 보존된 같은 320장의 예측을 사용하며 다시 학습하지 않는다. 각 지표와 bootstrap 비교를 재계산해 검증한다.

이 실험은 TRAIN512 두 번 노출, seed1의 짧은 supervised 비교다. 70,000장 새 학습 또는 세 seed 최종 성과로 해석하지 않는다. 최종 성능은 아직 미정이며 최고 모델은 변경하지 않았다.

## 실제 평가 시작 확인

64-update 학습이 완료됐고, 최종 checkpoint를 CPU에서 실제 로드하여 manifest와 state SHA가 일치함을 확인했다. 서버 CPU 검사 10개도 통과했다. GPU1의 실제 prediction worker PID 237914가 부모 228242 아래 실행 중이며 scorer 228264는 예측 확정을 기다린다. 최종 점수는 아직 없고 최고 모델은 유지한다. 증거는 `results_archive/training_complete_evaluation_started_20261016/`에 보존했다.

## 채점 foreground 검사 수정

Root 재검토에서 v1 채점 core가 conditional뿐 아니라 standalone도 Source-T foreground와 동일해야 한다고 잘못 검사한 것을 발견했다. 등록한 standalone 출력은 native QCC support를 쓰므로 이 조건은 적용할 수 없다. 기존 예측의 `_read_row`는 standalone mask가 저장된 실제 QCC support와 일치하는지 이미 검증한다.

정답을 읽기 전 대기 중인 CPU 채점기 228264만 의도적으로 중지했다. 이로 인한 v1 queue의 failed 상태와 중지 전후 명령행·상태를 보존했다. 원본 학습·예측·채점 코드 파일을 수정하지 않고 `score_v2.py`, `score_queue_v2.py`, `test_score_v2.py`를 새로 배포했다. 조건부 Source-T foreground 검사는 유지하고 standalone에는 등록한 native-QCC 계약을 적용한다. metric 계산, 대조군 재계산, bootstrap 및 확대 조건은 동일하다.

서로 다른 conditional/standalone foreground를 가진 320장 synthetic HDF5 검사, 잘못된 conditional의 GT 접근 전 거부, 기존 v1 prediction 큐 인수인계를 포함해 로컬·서버 각각 13개 검사가 통과했다. 수정 scorer PID 245039가 실제 대기 중이다. 새 score 경로는 `results_archive/evaluation_seed1_v2_20261016.json`이다. 학습과 GPU 예측은 재시작하지 않았으며 최종 성능은 아직 미정이다.

````

</details>

## PEER_REVIEW_20261016_KO.md

<details>
<summary>PEER_REVIEW_20261016_KO.md</summary>

Source SHA256: `2d511875f5c826eee8da527b4b2f6bcadd35f034ca4f05b368978e7a683105ad`

````markdown
# SW176 대기 중 동료 연구 확인

확인한 동료 브랜치 `origin/patch_v2`의 최신 commit은 `cab6c676e24a149a45ea960586c04e579f78d384`다. 새 변경은 없었다. `PV2_0092_signed_coupling/README.md`와 `ROUTE_MAP_0.9.md`를 직접 읽었다.

동료는 양수 결합만 있는 그래프에서 물체들이 일관되게 합쳐지는 오류를 지적하며, 일부 이웃 간 반발 결합을 도입한 seed0 실험을 진행 중이다. balance 0/0.25/0.5/1의 최종 결과는 문서상 pending이다. 이는 후보 가설이며 성공 결과가 아니다. 우리 모델의 실제 A와 오류 유형을 확인하기 전 그대로 적용하지 않는다.

동료의 중요한 교훈은 surrogate loss 감소나 연결 분리 통계 개선만으로 최종 mask 개선을 주장할 수 없다는 것이다. foreground를 크게 늘려 FG-ARI만 높이는 경우도 있어 세 지표와 실제 foreground를 함께 확인해야 한다. 우리의 고정 native32 계약은 유지한다. 동료의 native16 계약·학습 supervision 제한과 우리의 authorized supervised pilot은 다르므로 절대 점수를 직접 비교하지 않는다.

현재 SW176은 기존 matched loss와 데이터를 유지한 gradient 전달 정책 비교다. 그 결과가 확정되기 전 새 구조 학습을 중복 실행하지 않는다. SW176 실패 시 같은 정책의 LR·threshold 미세 조정 대신 오류 진단으로 다음 구조 가설을 선택한다. 이 기록은 새로운 성능 결과 또는 gating 인과 검증이 아니다.

````

</details>

## README_KO.md

<details>
<summary>README_KO.md</summary>

Source SHA256: `c9ead310de444667f6a854365bcd4e81417bd8108095df232657e34602ee42aa`

````markdown
# SW176 local phase gradient: 시작 기록

긴 Kuramoto 재귀를 통한 학습 신호 전달만 바꾸는 실험이다. 자세한 변경·대조·확대 조건은 DESIGN_KO.md에 사전 기록했다. 모델의 forward 연산, native component spike, Q/A, threshold와 mask readout은 바꾸지 않는다. D/M/event/reset의 장기 gradient는 여전히 남아 있다.

최종 source 검증 실행 경로는 `python -m collaborative_test.SW_0176_local_phase_credit.resource_queue_v4`이다. 이 단계는 TRAIN 위치0의 이미지 한 장에서 no-update source 검증 및 full1024 gradient를 측정하는 실험이다. 64-update 학습이나 검증 성능 결과가 아니다.

로컬·서버 각 15개 CPU 검사와 실제 TRAIN 캐시 준비, fresh source+context 모델 생성을 통과했다. 생성한 초기 state SHA `f372b3a450ebeaf0d49e20d019d3c0676d4998cc1c49586e4c4e4c89e969d7f5`가 완료 SW175 critical-component 대조군의 초기 state SHA와 정확히 일치했다. 이후 GPU1 worker의 실제 PID를 확인했다. 결과 대기 중이며 기존 최고 모델은 유지한다.

최초 v1의 실제 CPU 모델 생성 검증은 context 초기화 SHA를 잘못된 메타데이터 계층에서 읽는 오류를 발견했다. 실제 SW175 producer는 `admission_report.context_zero_init_sha256`에 기록하며 `initialization` 아래가 아니다. GPU 시작 전에 발견했으며 v1의 GPU·optimizer update는 0이었다. 원본 일곱 파일을 그대로 보존하고, 네 v2 파일에 올바른 계층 검증과 producer-shaped 회귀 검사를 추가했다. 검증을 생략하거나 모델 초기화를 변경하지 않았다. v1은 실패 증거 보존용이며 실행에는 위 v2 경로를 사용한다.

파일 SHA, 원본 CPU 거절 증거, 실제 모델 생성 CPU 결과, 시작 PID 관측은 `results_archive/started_source_preflight_20261016/`에 있다. 동료 브랜치 `patch_v2`도 이번 사이클에 fetch했고 `cab6c676e24a149a45ea960586c04e579f78d384` 이후 변경은 없었다. 데이터셋·checkpoint는 Git에 포함하지 않는다.

## v2 시작 후 확인된 초기화 오류

v2 부모/worker가 종료됐고 실제 source forward 이전 `reset_peak_memory_stats`에서 `Invalid device argument`가 발생했다. CPU 준비와 fresh 초기 state 검증은 통과했지만, 메모리 계측 API 호출 전에 CUDA device를 명시적으로 초기화해야 했다. optimizer update와 성능 결과는 없다. 원본 v2 코드·로그는 `results_archive/terminal_gpu_initialization_failure_20261016/`에 보존했다. CPU admission guard 뒤에서 device를 초기화하도록 새 버전을 검증하며, 이 기록을 모델 가설의 실패로 해석하지 않는다.

## CUDA 초기화 v3 복구 시작

CPU 준비와 uninitialized-CUDA guard를 통과한 뒤 `torch.cuda.set_device`를 호출하고, 그 다음에 peak memory를 초기화하도록 수정했다. public run_preflight 경로의 호출 순서를 검증하는 회귀 검사를 추가했다. 로컬·서버 각 16개 검사와 실제 fresh 모델·초기 state 동일성을 다시 통과했다. 새 v3 source와 별도 runtime을 사용하며 기존 v2 오류 로그를 보존했다. GPU1의 실제 부모202412·worker203150 실행을 확인했다. 결과는 아직 대기 중이며 초기화 수정이 모델 성능 개선의 증거는 아니다. 증거는 `results_archive/started_cuda_recovery_v3_20261016/`에 있다.

## v3 검증기 실패: 과학적 음성 판정 아님

v3는 forward 동일성·Q와 emitted spike gradient 검사를 지나 family 검증에서 종료됐다. 그러나 실제 `_grad_stats` 생산자는 `numpy.bool_(True)`를 반환하고 validator는 Python `True`와의 identity를 요구해 정상 값도 거절한다. 실제 유한한 parameter gradient를 producer에 넣어 validator가 잘못 거절하는 것을 CPU에서 재현했다. 따라서 현재 로그로 모델 gradient가 비유한하다거나 학습 정책이 실패했다고 판단할 수 없다. parameter gradient의 실제 수치는 보고서가 쓰이기 전 종료돼 미확정이다. 원소 발산과 float32 norm overflow도 구분되지 않았다.

가설 변경 SW177은 보류한다. 검증기 타입을 수정하고 실제 producer→validator 회귀를 추가한 뒤, 동일176 정책의 bounded 실험으로 수치를 확보한다. 원본 실패 로그와 CPU 증거는 `results_archive/invalid_family_guard_v3_20261016/`에 보존했다. optimizer update는 0이며 최고 모델은 유지한다.

## v4 실제 GPU source 검증 통과

동일 정책을 유지하고 검증기 bool 타입만 바로잡은 v4에서 원본과 후보의 six traces, gamma, Q, A, graph projection 및 conditional/standalone partition이 모두 정확히 일치했다. 원본 source state와 fresh 초기 state도 보존됐다. 실제 gradient 원소와 FP32/FP64 norm은 유한하며, 보고서의 각 parameter norm에서 family norm을 독립 재계산했다. 로컬·서버 CPU 각18개 검사와 실제 GPU 보고서 SHA를 검증했다. 이전 v3 거절은 모델의 과학적 실패로 분류하지 않는다.

| family | 실제 gradient L2 |
|---|---:|
| context_encoder | 0.01849636434 |
| dendrite | 4.486319238e-05 |
| graph | 0.002445028587 |
| integration | 3.918622109e-06 |
| kuramoto | 0.001890892634 |
| live_context_encoder | 0.01849636434 |
| membrane | 2.556273069e-06 |
| native_core | 0.01169911045 |

고정 TRAIN 한 장, 추가 optimizer update 0, 소요 11.86초, peak allocated 1.023 GiB이다. 이 검증은 학습 신호가 전달됨을 확인한 것이며 FG-ARI 향상이나 각 gate의 인과적 유용성을 입증한 결과가 아니다. 다음 단계는 DESIGN의 fresh64-update pilot이다. 기존 최고 모델은 유지한다. 전체 증거는 `results_archive/completed_source_preflight_v4_20261016/`에 있다.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `9ba2358f81e046b88eafd32a697dfa1598a29f4f2e894f6e033eefd7b6a52b02`

````markdown
# SW176 결과: local phase credit 정책 종료

동일한 native32 고정320, seed1에서 학습·예측·채점을 완료하고 확정 예측으로 세 지표와 paired bootstrap을 독립 재계산했다. TRAIN512 두 번 노출의 supervised pilot이며 세 seed 평균 또는 70,000장 새 학습 결과가 아니다.

| 모델/출력 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0164 | 0.770932 | 0.614993 | 0.630523 |
| SW0175_critical_component_conditional | 0.725044 | 0.614993 | 0.579214 |
| local_phase_conditional | 0.716674 | 0.614993 | 0.597555 |
| SW0175_critical_component_standalone | 0.674915 | 0.399856 | 0.509791 |
| local_phase_standalone | 0.709090 | 0.424322 | 0.590484 |

독립 출력은 SW175 대조군보다 FG-ARI가 0.03417 높아졌고 paired95% CI는 [0.01774, 0.05054]다. 그러나 기존 SW164 seed1보다 0.06184 낮으며 CI [-0.07551, -0.04838]다. 조건부 출력도 기존 모델보다 0.05426 낮고 CI [-0.06474, -0.04398]다. 따라서 등록한 두 endpoint의 확대 조건은 실패했다.

seed0/2 또는 70,000장으로 확대하지 않고 이 정책을 종료한다. 기존 최고 SW165는 유지한다. 단순히 finite gradient가 흐른다는 것만으로 분할 성능이 개선되지는 않았다. 이 결과는 모든 gating이 불필요하다는 증거는 아니다. 작은 LR/threshold/time-window 조정으로 같은 정책을 구제하지 않는다.

v1 채점의 standalone foreground 검사 오류는 GT를 열기 전에 발견했다. 대기 중인 CPU scorer만 중지하고, 원본 학습·예측·코드 해시를 보존한 채 v2 scorer를 배포했다. 로컬·서버 각각 13개 검사 후 같은 확정 예측을 채점했다. 자세한 이력은 EVALUATION_QUEUE_KO.md에 있으며 metric·bootstrap·확대 조건은 바꾸지 않았다.

다음 후보는 이미지별로 학습되는 edge polarity와 동일 크기의 nonnegative 대조군 비교다. 기존의 균일 반발 개입 실패나 동료의 pending 결과를 성공 근거로 취급하지 않는다. 실제 negative interaction 사용과 최종 mask 개선을 함께 확인해야 한다. 아직 이 후보의 실제 GPU 학습 결과는 없다.

검증된 score SHA: `7a4c4a3fcc0b00dd210703133bea6f687aef72b82fc3937cf5d4603eaaba4ab6`. 원본 score와 독립 검증·최종 모델 lineage는 `results_archive/completed_score_v2_20261016/`에 보존했다.

````

</details>

## TRAINING_START_KO.md

<details>
<summary>TRAINING_START_KO.md</summary>

Source SHA256: `c07ebc5590708e1b02a546d2196f1b5f116a0e4e4832b35ee57e192bd888f91f`

````markdown
# SW176: fresh source에서 64-update 학습 시작

GPU1에서 실제 부모213909·worker214126과 CUDA 메모리 사용을 확인했다. 검증을 통과한 local-phase 정책을 학습에 적용한다. Kuramoto의 각 전이 입력 theta만 detach하며 full1024 구간을 계산하고 마지막512 구간의 실제 component spike로 기존 critical-component 손실을 계산한다. D/M/event/reset 연산과 forward는 유지한다.

seed1, 고유 TRAIN512장×2회, B16/micro4, fresh Adam64 updates다. SW175 critical-component 완료 모델은 재사용 대조군으로만 보관한다. 후보는 동일한 원본 source+zero-final context에서 새로 시작한다. 학습 목표에 TRAIN feasible target을 사용하는 supervised pilot이며, 70,000장 추가 학습이나 세 seed 평가가 아니다.

실제 control의 checkpoint·optimizer·history·boundary SHA를 별도 receipt에 묶었다. fresh 초기 state, base encoder, 전체512 ID 순서, optimizer의 parameter 이름·순서·LR·betas·기본 옵션이 일치한다. tuple/list 표현은 실제 Adam save/load 계약 검사로 정규화했다. 이미 학습한 control의 가중치나 Adam 상태를 후보에 로드하지 않는다. 같은 per-update seed, clip1, integration projection을 사용한다.

로컬·서버 각8개 CPU 검사를 통과했다. 서버에서 실제 full 모델과 새 Adam으로 public CPU factory를 실행한 뒤 boundary0을 저장하고 parameter를 변경한 후 복원해 초기 state와 빈 Adam 상태가 정확히 복원되는지 확인했다. 첫 두 committed updates에서는 실제 component spike와 context output/hidden layer에 학습 신호·parameter 변화가 생기는지 확인하며, 별도 disposable update를 추가하지 않는다.

64 updates 완료 후 같은 native32 고정320 이미지에서 conditional/standalone 예측을 확정하고 GT-last 채점한다. source SW164 seed1과 같은 loss의 SW175 tail64 대조군 대비 FG의 paired CI 및 두 IoU 조건을 모두 통과할 때만 seed0/2·더 큰 데이터로 확대한다. 최고 모델은 최종 검증 전까지 유지한다. 시작 및 실제 CPU 복원 증거는 `results_archive/training_started_20261016/`에 보존했다.

실제 boundary payload SHA와 등록된 update cursor를 검증하고, 첫 두 committed updates가 실제 component-spike gradient 및 context output/hidden layer의 credit 조건을 통과했음을 확인했다. 추가 disposable update는 없다. 원본 gradient·parameter 변화는 `results_archive/first_two_committed_updates_20261016/`에 보존했다. 이 검증은 segmentation 성능 향상을 입증한 결과가 아니다.

````

</details>

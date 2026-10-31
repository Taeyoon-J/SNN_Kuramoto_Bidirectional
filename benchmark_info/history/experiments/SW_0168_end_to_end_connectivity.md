# SW_0168_end_to_end_connectivity

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## ADMISSION_RESULT.md

<details>
<summary>ADMISSION_RESULT.md</summary>

Source SHA256: `7591314a15fec59d64bd4d0df91c4be67b5326c48dfc4779415285d356133607`

````markdown
# SW0168 actual-spike 연결 학습: 선행 검증 통과

2026-10-15. TRAIN8, 각 arm 두 disposable B8 update(microbatch4), 실제 native32 T1024/settle512/live64 경로를 사용했다. **v4 큐의 terminal 상태는 complete_admitted**이며, root가 별도로 현재 로컬 구현 fingerprint에 대해 공개 report validator를 실행해 통과했다. 보고서 SHA256: `20814ce5dbe4cdb6680cfa69a0c82631c62de0d2a78e923367245f96fb49890a`.

| 첫 update의 raw gradient L2 | Maximin 연결 loss | Local-edge 비교 loss |
|---|---:|---:|
| RGB encoder | .19404750 | .09900457 |
| Native graph | .05740032 | .03371599 |
| Native core | .02205521 | .02518801 |
| Integration a_d/a_m/b 전체 | .000003698 | .000001714 |
| 실제 live-tail component spike gradient 평균 | .000648819 | .000381971 |

두 arm의 두 update 모두 encoder·graph·native core의 gradient와 실제 parameter delta가 양수였다. Native Q와 실제 component spike까지 gradient가 전달됐다. Integration credit은 작으며, 이것만으로 모든 scalar나 각 physical gating의 인과적 기여를 입증하지 않는다. 전체 gradient clipping은 두 arm/두 update 모두 작동할 필요가 없었다(scale1).

Maximin TRAIN loss는 .384801603→.383569217, local loss는 .259767124→.259165991이었다. 서로 다른 loss 크기를 성능 점수처럼 비교하지 않는다. 재사용한8장의 두 update 결과이므로 데이터 scaling이나 held-out FG-ARI 향상 증거도 아니다.

별도 frozen source B4 reference와 각 fresh arm의 초기 gamma·six-trace·Q·actual-A가 정확히 일치했다. Cache의 B4 component-spike/A도 일치했고, production native-B1 QCC와 SW164 zero-residual partition/merge-log의 live source replay8장도 각 arm 초기값과 정확히 일치했다. B4 cached-Q와 B1 Q의 bitwise 동일성을 요구하거나 주장하지 않는다.

GPU1에서 peak allocated memory는2,773,300,736 bytes, 각 B8 update는 약10.0–10.3초였다. 본 학습은 별도로 B16/micro4/64 update/512장 두 pass이므로, 이 자원 측정과 학습량을 구분한다. 검증 전후 source/context cache 파일 receipt는 같았고 VAL data는 읽지 않았다.

최초 세 attempt의 실행 문제는 모두 보존했다. v1은 QCC NumPy 반환값 처리, v2는 정수 QCC-label/boolean-F 입력 경계, v3는 GPU 완료 후 같은 process에서 CUDA 전용 검사를 재호출한 문제였다. v3는 양쪽 update 함수를 정상 종료한 뒤 보고서 저장 전에 실패했다. v4는 같은 immutable preflight_v3 학습 함수를 사용하면서 GPU 증거를 먼저 저장하고, 별도 CUDA-hidden CPU interpreter에서 source를 재검증해 이 문제를 해결했다. 이들은 FG-ARI 실패 결과가 아니다.

다음은 새 학습에서 conditional fixed-source support와 standalone learned-support를 모두 fixed320에서 평가하는 것이다. 최고 모델은 계속 SW165다. Gate 기여·.87 평균·native32 fresh70k scaling은 미달이다.

증거: `results_archive/admission_completed_20261015/server_records.json`, `independent_validation.json`. 실제 생성 파일은 `collaborative_testing/generated/` 아래에 보관한다.

````

</details>

## ADMISSION_START.md

<details>
<summary>ADMISSION_START.md</summary>

Source SHA256: `9ff5af41c3254a286d411dc14b2711a266172fd7c78f7eb98953e868650c90f2`

````markdown
# SW0168 실제 spike 연결 학습: GPU 선행 검증 시작

2026-10-15. SW167의 frozen-source head 학습이 실패했으므로, 이번에는 별도 head 없이 실제 RGB encoder·graph·native core를 함께 학습한다. 정답 TRAIN mask를 사용하는 **supervised 실험**이며 SW165 최고 결과를 대체하거나 unsupervised 성과로 합치지 않는다.

Candidate는 물체 내부 연결의 bottleneck과 서로 다른 물체를 잇는 bridge에 pair-count 가중 credit을 준다. 비교군은 같은 4-neighbor edge에 균등한 local loss를 준다. 두 모델의 source·trainability·학습률·데이터·순서는 같고 loss만 다르다. Native positive-clamped spike Q를 그대로 사용한다.

서버에 연구 소스22개를 배포하기 전에 Python 구문과 CPU13개 테스트를 통과했다. 서버에서도 동일13개 테스트와 실제 TRAIN8장의 RGB/HDF mask/cache/source 일치 검증을 통과했다. 별도 독립 코드 검토에서 발견한 큐 timeout 분기 오류는 GPU 실행 전에 수정했고, 시간 초과 전에는 child를 중지하지 않는 동작을 검사했다.

최초 GPU 선행 검증 supervisor3667425/worker3667478은 B4 source/cache 검사를 통과한 뒤 B1 QCC 반환값을 Tensor로 취급한 API 오류로 종료됐다. 실제 production 함수는 NumPy 배열을 반환한다. 오류는 본 학습 전 발생했고 실패 상태·로그는 보존했다. 같은 시도를 재시작하거나 배포된 파일을 덮어쓰지 않았다.

`preflight_v2.py`는 NumPy 반환값을 직접 받도록 수정했다. 반환 API 회귀 검사를 포함한 CPU14개 테스트와 실제 TRAIN8 CPU admission은 통과했지만, 실제 첫 loss에서 QCC의 정수 component-ID grid를 boolean F로 변환하지 않은 오류로 종료됐다. B4/B1 source parity 뒤, optimizer update 전에 발생했다. worker 로그의 원인은 `support must be boolean`이며 supervisor의 뒤이은 identity 오류가 근본 원인은 아니다. v2 기록도 보존했다.

`preflight_v3.py`는 loss 호출 경계에서 `QCC labels > 0`을 사용한다. 정수 source labels 자체는 readout용으로 그대로 보존한다. 서버17개 CPU 검사에 더해 실제 TRAIN8의 **양쪽 loss16건을 실제 입력과 backward로 검사**했고 모든 Q gradient는 finite였다. 이는 cached Q에 대한 CPU 검사이며 physical rollout gradient 증거와 구분한다. 별도 v3 supervisor3668808/worker3668988가 GPU1에서 실제 실행됨을 확인했다. v3 fingerprint는 `bb75b00cde8eed970d8cc95dedfdd97dd430a7d7828c0578a5bd72643bf23a01`다. GPU0의 타 사용자 작업은 사용하지 않았다. **이 기록은 v3 실행 시작 증거이며 admission 통과나 성능 결과가 아니다.**

먼저 source B4 실제 six-trace/Q/A와 cache S/A를 검증하고, 별도로 새로 로드한 source의 production B1 QCC/SW164 partition·merge log를 각 arm 초기값과 비교한다. 이는 TRAIN8의 live source replay이며, 존재하지 않는 TRAIN B1 archive와 비교했다는 주장이 아니다. 각 arm 두 disposable B8 update(microbatch4)에서 실제 Q→component spike와 encoder/graph/native-core gradient 및 parameter 변화량을 검사한다. 본 학습 B16/64update와는 범위를 구분한다.

이 검증을 통과한 경우에만 고정512장·두 pass의 본 학습을 허용한다. Conditional frozen-support와 standalone learned-support 두 출력을 모두 평가한다. 목표는 고정 평가에서 성능이 실제로 개선되는지 확인하는 것이며, gradient 존재만으로 각 gating의 인과적 기여를 주장하지 않는다.

실행 증거: `results_archive/admission_started_20261015/server_records.json`, 최초 실패 `results_archive/admission_failed_20261015/server_records.json`, v2 시작/실패 `results_archive/admission_v2_started_20261015/`, `results_archive/admission_v2_failed_20261015/`, v3 시작 `results_archive/admission_v3_started_20261015/server_records.json`. 생성 파일의 실제 저장 위치는 `collaborative_testing/generated/`이며 canonical 결과 경로는 junction으로 유지한다.

````

</details>

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `f88581b6d5ed32b2d0e9180c8b99bb4e8dc2dfdf5b52f79f008e563a109ceae9`

````markdown
# SW0168 prospective: actual-spike maximin 연결학습

2026-10-15. **설계만; 구현/배포/실행 없음. TRAIN-mask supervised track.** SW165는 incumbent로 유지하고 sealed163/165/166/167은 변경하지 않는다. 목표 .87 및 native32 fresh70k/data-scaling/all-gate contribution은 여전히 미달이다. 이 pilot을 unsupervised 성과로 합치지 않는다.

## 근거와 단일 가설

SW167은 같은 frozen actual S/Q/A의 작은 residual energy head를 학습했다. Structured/local 두 loss가 감소했지만 final FG .770932717/.771162710은 SW164 seed1 .770932099와 사실상 같고 objectIoU는 낮았다. 따라서 fixed-source head recipe는 닫는다. 이 결과만으로 encoder 정보 부족이나 solver 최적성을 증명하지는 않는다.

이번에는 새 head 없이 **실제 Q를 생성하는 RGB encoder→graph→rhythm/core를 학습**한다. SW160 `train.py`는 SW155 `supervised_relation_loss`의 all-pair MSE+positive-rho target을 사용했고, SW167 `train.py`는 source cache 뒤 MLP만 최적화했다. 새 loss는 split을 만드는 연결 bottleneck과 merge를 만드는 강한 bridge에 객체쌍 수에 비례한 credit를 준다. Decoder/SlotAttention/RGB reconstruction/source-Q distillation/fixed-density prior는 없다. Split·merge 둘 다 남는다는 SW166 증거에 맞춘 **구조적 학습 가설**이며 성능 향상은 미검증이다.

## 모델·source·primary readout

Seed1 SW0097→native32 SourceT를 그대로 시작한다. `SW0135.foundation.load_native32_foundation(1)` + 기존 SW143-T 설치 경로를 사용한다. Actual base valid3x3 RGB encoder8채널을 unfreeze하고 native learned graph와 nongraph core 및 기존 a_d/a_m/b 세 tensor(12 scalars)를 함께 최적화한다. 새 encoder residual/MLP/threshold calibration은 추가하지 않는다. 등록된 statistics/clipping/adaptive_pool32/physical mapping은 고정이다.

Optimizer family 구성·중복 방지와 checkpoint/evaluator 골격은 `SW0160/model.py,train.py,predict_evaluate.py`에서 재사용하되, 새 residual encoder는 설치하지 않는다. Native base/graph/core LR3e-5, integration LR1e-3, fresh Adam, logical-gradient global clip1. 두 arm의 trainability·init·LR·노출·forward는 완전히 같다.

Inference의 그룹화는 SW165와 같은 actual-spike Q와 실제 current prepare_coupling tuple[0] A/gain의 signed energy, strict-positive all-cross merge, canonical ties 및 4-connectivity다. 새 Q threshold/K/min-size는 없다. 한 learned rollout에서 **두 사전등록 co-primary**를 모두 생성하고 어느 쪽도 사후 선택하지 않는다.

1. **Conditional-support 진단:** frozen original SourceT QCC의 F를 사용한다. 모든 F patch를 positive label로 유지하고 밖은0이다. 이 predictor는 frozen source-support 경로와 trained rhythm 경로로 구성되며 새로운 unseen image에서는 old+new 두 rollout이 필요하다. Fixed320에서는 검증된 SourceT labels를 재사용한다. ForegroundIoU는 per-image SW165와 정확히 같아야 한다. 이것만 좋아져서는 실용 모델의 성공으로 판정하지 않는다.
2. **Standalone 실용 경로:** 같은 learned rollout의 실제 component spikes를 native CPU QCC(min8/largest-background, 원래 modal/native32 설정)에 넣어 current F를 얻고, 같은 learned Q/A energy의 induced solver를 적용한다. 원본 source teacher/GT/외부 head 없이 하나의 learned rollout만 필요하다. source와 마찬가지로 QCC의 CPU Q와 energy의 GPU Q 계산 범위를 구분한다. 초기에는 두 경로 모두 SourceT+SW165와 일치해야 한다.

TRAIN의 feasible targets에는 캐시된 frozen-source F만 사용하므로 optimizer update당 teacher rollout은 추가되지 않는다. 이 제한이 foreground 적응을 직접 감독하지 않는다는 점을 standalone co-primary가 검증한다.

## 두 arm — loss만 다르다

학생의 실제 settled component spikes로 production `spike_synchrony_affinity` Q를 계산한다. 모든 loss 입력은 **native positive-clamped product Q**다. Raw signed product나 theta/PLV를 대체하지 않는다. Student Q와 A가 변할 수 있지만 loss는 Q에만 직접 걸며 A를 GT relation으로 학습하는 별도 shortcut은 없다. Graph는 native rollout의 Q 생성 경로로 credit를 받는다.

TRAIN의 frozen F 안에서 GT ID별 4-connected components를 target G*로 만든다(`SW0167.structured_energy.feasible_target`). F 안의 GT-background도 별도 target components로 남긴다. GT-background를 inference의0으로 이동시키거나 training patches에서 제거하지 않는다. F로 인해 같은 GT object가 끊기면 별도 target이다. 서로 다른 F-connected components 사이의 불가능한 연결은 학습 pair denominator에서 제외하고 수를 보고한다.

**Candidate: constrained maximin loss.** 4-neighbor edge score q_e=Q_ij이다. CPU union-find/sorting은 detached q를 받아 edge index와 pair-count weight만 반환한다. Loss는 선택된 index의 **live Q**에서 다시 계산한다.

* Positive pass: 서로 다른 G*를 잇는 edges를 제거하고, 각 G* component 안에서 maximum spanning forest를 만든다. Descending edge가 크기 u,v의 두 집합을 처음 잇는 순간 u*v same-component pairs의 maximin bottleneck을 대표한다. Weight u*v로 (1-q_e)^2를 합하고 전체 represented positive-pair 수로 나눈다. 이것은 내부의 약한 연결을 강화한다.
* Negative pass: 각 G* component를 먼저 union(동일 component 내부 score는 고정1, 학습 credit 없음)하고, 서로 다른 target을 잇는 실제 edges를 descending q로 처리한다. 두 현재 집합이 합쳐질 때 새로 연결되는 different-target node pairs 수를 weight로 q_e^2를 합한다. 전체 represented negative-pair 수로 나눈다. 이것은 강한 잘못된 bridge를 낮춘다. 연결 불가능한 F-components는 서로 union하지 않는다.
* L_conn은 존재하는 positive/negative normalized losses의 단순 평균이다. 두 class가 모두 없으면0. Ties는 canonical unordered patch-index 순서다. 정렬/union 결과는 detach한다. 이는 almost-everywhere bottleneck subgradient이며, SW165 all-cross greedy 목적의 exact gradient나 ARI gradient는 아니다.

**Matched local control:** 동일 4-neighbor edges에서 target same/different별로 (1-q)^2 / q^2의 uniform class means를 평균한다. 동일 class-empty 규칙을 사용한다. 새 capacity가 아니라 **global maximin pair-count routing 대 local-edge credit**만 비교한다. 기존 SW160 all-pair loss/positive-rho 보조항은 넣지 않는다. Lambda/온도/밀도/threshold sweep은 없다.

## 고정 데이터·시간·비용

SW163/SW167이 검증한 같은 seed1 ordered512 TRAIN IDs와 RGB/mask source mapping을 사용한다. 실제 학생 학습에는 RGB를 매번 native encode하여 gamma를 새로 만든다. Cached spikes는 학생 forward 대체가 아니라 frozen support/lineage 검사에만 쓴다. TRAIN original CPU-B4 QCC F와 cached GPU Q의 arithmetic 범위를 구분한다. Source cache를 B1으로 다시 만든다고 주장하지 않는다.

`SW0134_native_spike_binding.rollout.late_rollout` / SW135 foundation 경로의 full T1024, settled512, live-tail64를 사용한다. Live static drive/coupling은 burn-in 전에 준비하고960 시점 recurrent state만 detach한다. Q는 448 detached+64 live를 포함한 전체 settled512에서 계산한다. 두 arm 모두 B4 microbatch×4=logical16, imagewise mean loss 후 한 번 clip/Adam이다. 임의 short-window 대체는 없다.

**64 updates=512 TRAIN 두 complete passes**, shuffle168001, final64만 평가한다. 256-update fanout부터 시작하지 않는다. 기존 B4 resource33.9초/logical16은 다른 loss의 참고치일 뿐 runtime 약속이 아니다. 실제 두 disposable updates에서 시간/peak-memory/CPU forest 시간을 측정해64×2arm 예상비용을 기록한 다음 admission한다. Graph chunk/checkpoint 수학은 그대로 재사용한다. Logical boundary마다 weights/Adam/RNG/order를 저장해 owner preemption 시 partial-gradient 재사용 없이 재개한다.

## 최소 admission과 기여의 한계

CPU tiny graph에서 pair-count 총합, positive/negative constrained passes, disconnected support/GT0/ties, selected-live-Q gradient를 확인한다. 실제 첫 B4 source와 학생 init의 same-mode all-six-trace/Q parity, B1 SourceT/SW165 labels·merge-log exact replay, frozen-support hash 및 native actual-A capture를 확인한다. 새 backend identity나 cache/B1 bitwise identity를 요구하지 않는다.

두 disposable logical updates에서 finite losses와 actual changed optimizer families를 확인한다. Native Q에 대한 loss gradient, **actual Q→component spikes까지의 전달**, encoder/graph/Kuramoto-drive/dendrite/membrane 및 a_d/a_m/b의 raw gradient norms·postclip scale·actual parameter deltas를 따로 기록한다. Positive-product의 clamp 때문에 critical edges에서 dL/dQ가 있어도 dL/dS가0일 수 있다. Selected critical-edge의 Q/clamp-active 분포와 실제 dL/dS를 보고한다. 전체 preflight에서 student encoder/graph 또는 native rhythm으로 credit가 전혀 없으면 actionable하지 않으므로 중지한다. Lambda 추가, signed-product 대체, clamp removal/epsilon 변경으로 구조를 구제하지 않는다. 단일 scalar의 zero를 억지 update로 통과시키지 않는다. Membrane/binary E의 low credit는 FG 실험과 분리해 명시하며 all-gate proof로 해석하지 않는다.

첫/마지막 동일8 TRAIN no-update replay에서 local/critical-edge errors, represented-pair mass, event occupancy/transitions, margin 및 실제 Q/partition 변화를 기록한다. Loss 감소만으로 object binding을 주장하지 않는다. Binary E가 여전히 거의1이면 발화선택 유용성은 입증하지 못한다. 학습 gate gradient 자체도 인과적 기여 증거가 아니다.

## fixed320 및 중단·확장

양 arm final64의 **두 co-primary 모두** predictions/own provenance를 전부 고정한 뒤 동일 IDs1320..1639, native32 modal4px GT의 기존 세 지표를 계산한다. Seed0/2 복제는 아래를 모두 만족할 때만 등록한다.

* Conditional candidate FG가 seed1 SW164(.770932099) 및 conditional local-control보다 모두 높고 각각 paired95% CI lower>0; objectIoU>=SW164; foregroundIoU per-image exact.
* Standalone candidate FG도 SW164 및 standalone local-control보다 모두 높고 각각 paired95% CI lower>0; 두 IoU means 모두 SW164 이상. 새로운 +.01 같은 최소 FG margin은 두지 않는다. Conditional만 개선되고 standalone이 미달하면 조건부 진단 이득으로만 보고하고 확장하지 않는다.

Bootstrap10000/shared320 image draws/RNG168. 실패하면 recipe를 닫고 LR/passes/loss mix/threshold/support 선택 sweep을 하지 않는다. Tradeoff 결과는 기록하되 미통과 recipe를 통과로 바꾸지 않는다.

3seed 재현 전 .87 평균/Slot 승리를 새 모델에 주장하지 않는다. 성공 후에도 fresh native32 70k와 unique-data scaling은 별도 고정-update/노출 비교가 필요하고, 모든 gate의 기여는 trained checkpoint에서 동역학 개입과 필요시 matched retraining으로 따로 확인한다. Reserve90640..90959는 읽지 않는다. 생성 output은 기존 `collaborative_testing/prepare_output.ps1`의 중앙 output junction/archives 정책을 따른다.

````

</details>

## POST_TRAIN_DIAGNOSIS_KO.md

<details>
<summary>POST_TRAIN_DIAGNOSIS_KO.md</summary>

Source SHA256: `f334a882627c10aacb14aa05911f3b3e3a53c109821849cfa61a105c1d86883f`

````markdown
# SW0168 학습 후 physical gating 진단

2026-10-15. Sol6.1 설계 검토와 root의 실제 완료 artifact 수집에 근거한다. 고정320 평가 결과와 분리한 TRAIN-only 진단이다.

고정 TRAIN8에서 두 arm 모두 학습 전후 settled event-on fraction1, event-transition fraction0이었다. 이 기록에서는 `S=E*g`가 `S=g`로 줄어든다. Membrane/integration parameter의64회 update와 nonzero surrogate credit은 확인됐지만, **실제 발화 선택이 분할에 기여했다는 증거는 아니다.** 전체512장·70k·VAL에 대한 포괄적 진술이나 FG-ARI 성능 예측으로 확장하지 않는다. `dendric_layer` raw norm은 진단 이름 필터 누락으로 미측정이며 optimizer update 기록과 구분한다.

이는 SW168의 encoder·graph·native-core joint training이 실행되지 않았다는 뜻이 아니다. 두 최종 모델의64회 history/optimizer/checkpoint/boundary 검증은 통과했고 두 TRAIN loss도 감소했다. 다만 'gradient가 흐른다'와 '해당 physical forward가 물체 구분에 필요하다'는 별도 검증 대상이다.

평가 후 조건부로 검토할 단일 다음 가설은 **source TRAIN component RMS로 정규화한 입력, 느린 membrane 적분, binary hard reset**이다. 또 작은 threshold 조정을 반복하는 대신 current 단위와 실제 charge history/reset을 바꾸는 방향이다. 초기 시간상수는16 step, 정규화된 threshold 시작값은.5를 검토하며 rate target·밀도 맞추기·threshold sweep은 넣지 않는다. 이는 아직 prospective 아이디어이며 SW168과 source-output parity를 주장하거나 실행을 등록한 것이 아니다.

같은 forward/capacity에서 recurrent-history live/detached 학습을 비교하면 credit의 역할을 분리할 수 있다. 실제 forward 필요성을 주장하려면 학습된 모델에서 history-off/reset-off 개입을 별도로 확인해야 한다. 먼저 소규모 TRAIN에서 finite mixed events/transition과 실제 current/history credit이 생기는지 확인하고, untrained FG가 떨어진다는 이유만으로 곧바로 기각하거나 activity를 맞추는 sweep을 하지 않는다. 이 방향은 과거 DC-history 제거 또는 threshold-only 변경과 구분한다.

SW168의 conditional/standalone 평가를 먼저 마친 뒤 recipe 유지/종료와 다음 실험을 결정한다. 이 문서는 .87 평균, 모든 gating 기여, 70k scaling 또는 Slot Attention 승리를 보장하지 않는다.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `da8ab98a7fe410e89db828c9f5684734412312ca22a1f0b0d6b5c97565b55f9a`

````markdown
# SW0168 결과: maximin은 local보다 낫지만 기존 모델에는 패배

2026-10-15. 고정320장·native32·seed1, TRAIN512장·두 pass·64 Adam logical updates의 supervised joint-training 비교를 완료했다. Encoder·graph·native core·integration을 학습했다. 두 arm의 최종 checkpoint/optimizer/history/boundary와 최종 prediction/score를 독립 검증했고, root가 공개 `validate_score()`로 실제 GT에서 전체 지표와 paired bootstrap을 다시 계산해 통과했다. 평가 복구 동안 학습을 반복하지 않았다.

| 방식 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 SW164 / SW165의 동일 seed1 | **.770932099** | .614993430 | **.630523482** |
| Maximin, 기존 SourceT foreground 유지 | .757128250 | .614993430 | .624364865 |
| Maximin, learned rollout의 standalone mask | .751409368 | **.618290863** | .617541401 |
| Local control, 기존 SourceT foreground 유지 | .753386670 | .614993430 | .622111225 |
| Local control, standalone mask | .745430517 | .610882881 | .611957898 |
| 기존 SourceT QCC | .755273083 | .614993430 | .579513786 |

Conditional maximin은 local 대비 FG-ARI+.003741580, paired95%CI[+.000728168,+.006925432]였다. Standalone도 local 대비+.005978851, CI[+.001482236,+.010460219]였다. 따라서 이번 비교에서 structured connectivity credit은 matched local credit보다 좋았다.

그러나 기존 SW164 대비 conditional FG-ARI는−.013803849, CI[−.019752503,−.008323134]이고 standalone은−.019522730, CI[−.027443721,−.012086738]다. 두 Object IoU도 기존보다 낮았다. Conditional foreground는 모든 이미지에서 원본과 정확히 같았으므로 그 FG-ARI 감소를 foreground 범위 감소로 설명할 수 없다. Standalone extent IoU 상승만으로 instance 구분 성공을 주장하지 않는다.

Bootstrap은 등록한 공통320 이미지·10000회·RNG168이다. 이는 이미지별 paired 차이이며 세 seed 모집단 신뢰구간이 아니다. **두 co-primary를 모두 요구한 expansion gate 실패: 같은 recipe의 seed0/2·70k 확장, LR/passes/loss mix/threshold/support 선택 sweep을 하지 않는다.** Conditional만 골라 성과로 바꾸지 않는다.

고정 TRAIN8에서 initial/final·두 arm 모두 settled binary E-on1, transition0이라는 진단도 확인했다. 그8장에서는 S=g이고 membrane 발화 선택의 기여는 미증명이다. Nonzero surrogate gradients와64회 parameter update는 이 한계를 해소하지 않는다. 다만 이 포화 진단 자체가168의 FG-ARI 감소 원인을 확정하는 것은 아니다. `dendric_layer` raw norm의 이름 필터 누락도 별도로 기록했다.

최고 모델은 SW165 세 seed 평균 **.804298915/.609975448/.651148975**로 유지한다. 이번 결과는 seed1 pilot이며 그 평균과 섞지 않는다. FG-ARI 평균.87·functional gate contribution·fresh native32 70k/data scaling은 여전히 미달이다. 다음 모델 변경은 별도 설계로 등록하며168의 실패를 재학습이나 평가 계약 변경으로 지우지 않는다.

최종 score SHA256: `0a197c6292df1ccd9ebdeff6ea04563bcb7f87f903faa705f5aa75a77072fe70`.

증거: `results_archive/completed_score_20261015/server_records.json`, `independent_verification.json`; 학습·운영 이력은 `TRAINING_RESULT_KO.md`, gating 해석은 `POST_TRAIN_DIAGNOSIS_KO.md`. 실제 생성 파일은 `collaborative_testing/generated/` 아래에 보관한다.

````

</details>

## TRAINING_LAUNCH.md

<details>
<summary>TRAINING_LAUNCH.md</summary>

Source SHA256: `d5bd2691839438f89ecd5a24824b4098d17badf9d2d4a95f50021426ac32ca30`

````markdown
# SW0168 본 학습과 자동 평가 큐 실행

2026-10-15. GPU 선행 검증 통과 후 실제512장 TRAIN RGB/HDF mask/source-cache/target 입력 검증을 완료했다. 서버 본 학습 CPU14개 검사와 실제512 입력 검증, 실제 큐 admission 함수의3-tuple handoff가 모두 통과했다. 본 학습 소스는 독립 검토를 거쳤으며 배포 후 변경하지 않았다.

| Arm | Supervisor | Worker | 실제 GPU |
|---|---:|---:|---:|
| Constrained maximin | 3671729 | 3671833 | 2 |
| Matched local edge | 3671730 | 3671820 | 1 |

두 worker의 `/proc` 명령줄과 NVIDIA GPU 소유 PID를 확인했다. 각 worker가 실제 optimizer update를 수행한 로그도 확인했다. 이는 큐 state만으로 학습을 추정한 기록이 아니다. GPU0의 외부 PID2653200은 사용하거나 중지하지 않았다.

두 arm은 source seed1, 동일512장, 동일 초기값·shuffle168001·LR·Adam·64 logical updates를 사용한다. Logical B16은 microbatch4 네 번을 imagewise 평균한 gradient로 한 번 update한다.512장 두 pass/1024회 이미지 노출이며, fresh70k 학습이나 seed0/2 재현은 아니다. Encoder·native graph·core·기존12 integration scalar를 학습하고 새로운 MLP/residual encoder는 추가하지 않는다.

각 update의 durable checkpoint에 model/optimizer/RNG/order/history를 저장한다. 다른 GPU 사용자가 나타나면 자기 child만 중지하고 lease를 반환한 뒤 같은 durable boundary에서 재개한다. 사용하지 않은 parameter는 optimizer receipt에 명시하며 가짜 update를 만들지 않는다. Initial/final 고정 TRAIN8의 실제 spike·gate·margin·clamp·Q·conditional/standalone partition 진단도 별도로 저장한다.

학습 완료 후 두 co-primary mask를 자동 생성하도록 prediction 두 개와 CPU GT-last score 큐를 미리 등록했다.

| 큐 | 실제 supervisor PID | 등록 시 상태 |
|---|---:|---|
| Maximin fixed320 prediction | 3674744 | waiting_for_training |
| Local-control fixed320 prediction | 3674760 | waiting_for_training |
| 두 prediction 이후 GT-last score | 3674763 | waiting_for_both_predictions |

평가 코드의 서버9개 CPU 검사, 실제 SourceT/SW164 고정320 reference·RGB cache·baseline score SHA 검사, 실제 실행 중인 두 training supervisor 의존성 검사, prediction→score의 실제 process handoff가 모두 통과했다. 등록된7개 training/prediction/score process가 실제 live인 것을 확인했다. VAL mask는 아직 읽지 않았다.

Prediction은 양쪽 training artifact가 모두 완료돼야 시작한다. Conditional은 기존 SourceT foreground를 사용하고, standalone은 같은 learned rollout의 production native-CPU QCC에서 foreground를 다시 얻는다. 둘 다 같은 actual GPU-Q/A와 기존 signed induced merge를 사용한다. 모든 prediction을 고정한 뒤에만 기존 고정320/modal32 GT와 세 production 지표를 계산한다. Bootstrap은10000회, 공통320 이미지, RNG168이며 사전등록한 두 co-primary gate를 모두 적용한다.

**최종 FG-ARI 결과는 아직 없다.** SW165가 incumbent이며 .87 평균·gating 기여·fresh70k/data scaling은 이 실행 시작으로 달성됐다고 주장하지 않는다.

증거: `results_archive/training_started_20261015/server_records.json`. 생성 파일은 기존 junction을 통해 `collaborative_testing/generated/`에 저장한다.

````

</details>

## TRAINING_RESULT_KO.md

<details>
<summary>TRAINING_RESULT_KO.md</summary>

Source SHA256: `fd7a457a5803c674d34f37ff6a6166e891189f7001995ccc0394e8799d0e7ced`

````markdown
# SW0168 본 학습 완료: 성능 평가는 별도 진행

2026-10-15. 두 arm 모두512장·두 pass·64 logical B16 업데이트를 완료했다. Worker의 완료 로그에 더해 root가 각 최종 artifact의 공개 validator를 독립 실행해 checkpoint·optimizer·history·durable boundary 일치 검증을 통과했다. 재학습하지 않았다.

| Arm | Epoch1 TRAIN loss 평균 | Epoch2 TRAIN loss 평균 | 최종 manifest SHA256 |
|---|---:|---:|---|
| Maximin | .350470750 | .340214532 | `14fe4b3055155f15889ae293fd12c5b41cbd3abc95b39637c6662a07b4b57b71` |
| Local edge | .248665896 | .231140953 | `d075eab288cb6f2b393f81e845178b7980ff6a5800a695bf6da43001d38cf3a3` |

서로 다른 목적함수의 크기를 성능 점수로 비교하지 않는다. Maximin/local 양쪽에서 Kuramoto·membrane·integration parameter delta가64회 모두 양수였다. Native core의10개 parameter에는 실제 Adam step64가 기록됐으며 `dendric_layer.tau_n`, oscillator dense weight/bias도 포함한다. 다만 raw gradient 진단의 `dendrit` 문자열 필터가 실제 이름 `dendric_layer`를 놓쳤으므로, **dendritic raw gradient norm이 개별 측정됐다고 주장하지 않는다.**

더 중요한 제한은 고정 TRAIN8의 실제 동역학이다. Initial/final·양쪽 arm 모두 settled event-on fraction=1, event transition fraction=0이었다. 이 기록에서는 `S=E*g`의 E가 항상1이므로 settled S는 g와 같다. Membrane과 integration의 nonzero surrogate gradient 및 parameter update는 유효하지만, 이8장에 대한 실제 발화 선택의 기여를 입증하지 않는다. 다른 TRAIN 이미지나 전체70k까지 동일하다고 일반화하지 않는다.

Maximin supervisor는 worker가 `training_complete`를 출력하고 최종 모델까지 게시한 뒤, 정상 종료 시점의 `/proc`/Popen race를 identity 오류로 판정했다. 이 때문에 기존 prediction/score 큐가 의존성 실패로 종료됐다. 원본 상태와 로그를 수정하지 않고 보존했다. 완료 모델은 위 독립 validator를 통과했으므로 **학습을 다시 돌릴 이유가 없다.**

새 CPU recovery receipt는 두 모델·64회 이력·optimizer·checkpoint·boundary·원본 terminal state/완료 로그의 SHA를 결합한다. Receipt SHA는 `ceb3f2c06386c4b616cbbfd9c158656dfebe21268be29dd7429d8e99d0ad27e3`다. 첫 v2 prediction 재개는 GPU owner helper에 필수 callback을 빠뜨린 호출 오류로 GPU 실행 전에 종료돼 별도로 보존했다. v3는 실제 `_stable_idle` signature에 대한 autospec 검사를 추가하고 callback을 전달한다. 또한 active worker의 자기 GPU memory를 idle admission threshold와 혼동하지 않도록 검사하고, 정상 종료 시 실제 Popen을 기다려 재확인한다. 이는 모델·loss·평가 계약 변경이 아니다.

아직 held-out FG-ARI 결과가 없으므로 incumbent는 SW165이며 성능 개선을 주장하지 않는다. Source support를 쓰는 conditional과 같은 learned rollout에서 support를 구하는 standalone을 모두 고정320에서 평가해야 한다.

증거: `results_archive/completed_training_20261015/independent_verification.json`, `results_archive/evaluation_failed_20261015/server_records.json`. 실제 생성 파일은 `collaborative_testing/generated/`에 저장한다.

````

</details>

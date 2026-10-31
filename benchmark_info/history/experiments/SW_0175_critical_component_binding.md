# SW_0175_critical_component_binding

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `b3129d016bbd192861f1a43d952c3fbc8b007d1d5a4654a68d2f784cf43a2f11`

````markdown
# SW0175: topology-critical component binding

2026-10-15 prospective supervised paired pilot. 이 문서는 설계이며 구현·GPU 실행·학습 결과가 아니다. SW171–174의 sealed source와 실패 기록을 변경하지 않는다.

## 선택 근거와 한계

SW174는 raw event 역전파만 분리하면서 native six traces와 두 readout을 정확히 보존했다. 그래도 full1024 gradient norm은 context 755,906 / graph 505,522 / core 89,837, full/tail cosine은 -0.89887이었다. Primary transfer loss는 +0.00008569, conditional FG-ARI는 .60379075→.60171174로 악화했다. 따라서 event surrogate 하나를 원인으로 단정하거나 full-BPTT 학습을 확대하지 않는다. 원본 report SHA는 `7583c48c6909e2acdcf141a369c3e6b9f65c7c2681c146fb1e1ed226945e72c3`다.

실제 `kuramoto_layer.py`는 theta에 dt*(omega + coupling + kappa*sin(gamma-theta))를 누적한다. 긴 recurrent Jacobian이 현재 방향 불일치에 기여할 가능성은 남지만, SW174가 그 원인을 확정하지는 않는다. 이를 새 integrator/gradient clipping/시간 길이 sweep으로 바꾸지 않는다.

이번에는 SW173에서 실제 측정한 **여러 이미지의 positive critical-edge loss gradient 약 90%가 production product에서 차단되는 문제**에 직접 대응한다. Production Q는 각 signed Pearson rho를 0으로 clamp한 뒤 네 성분을 곱한다. 두 성분 이상이 nonpositive이면 positive Q loss는 그 edge의 모든 성분에 credit을 주지 못한다. 새 가설은 실제 물체 연결을 결정하는 forest edge의 가장 약한 signed component에 직접 credit을 주면 native spike binding이 개선된다는 것이다. 이는 FG 개선이나 all-gate 필요성의 증거가 아니다.

기존 SW155는 모든 same-FG/BG-BG pair에 평균 positive-rho 보조항을 이미 썼다. SW0175는 이를 새 방법으로 재명명하지 않는다. **GT-connected critical forest의 represented pair mass로 가중한 worst-component 목적이 기존 positive-Q 항을 대체**한다는 차이만 시험한다. Peer의 최신 로컬 검토(`peer_review_20261014d.json`, cab6c676)는 다른 계약의 readout/extent 결과이며 이 학습 가설의 검증으로 사용하지 않는다.

## 변경 하나: positive critical-route loss 대체

입력/forward/production classifier는 양 arm에서 완전히 같다. 원래 SourceT의 실제 emitted S[B,4,1024,1024]를 사용하고 settle512 뒤의 512 frames에 production과 같은 float32 centering, norm clamp1e-8, matmul, rho clamp[-1,1]을 적용한다. `Q_ij = product_d max(rho_dij,0)`다. 별도 phase/gamma/membrane teacher나 direct mask head는 없다.

매 학습 forward에서 기존 `SW_0168_end_to_end_connectivity/connectivity_objective.py::forest_routes(Q.detach(), feasible_target, source_F)`를 그대로 호출한다. Positive/negative edges, canonical ties, feasible GT target와 pair weights는 공통이다. Q를 rho로 바꿔 forest ordering을 다시 만들지 않는다.

- Control `native_Q`: 기존 pair-count weighted positive `(1-Q_ij)^2`, negative `Q_ij^2`.
- Candidate `critical_component`: positive만 `(1-min_d rho_dij)^2`로 교체한다. 구현은 `amax_d((1-rho_dij)^2)`이며 amax 동률에는 PyTorch의 균등 subgradient를 사용한다. Rho는 실제 spike trace에서 live로 계산하며 positive clamp를 거치지 않는다. Negative는 동일한 native `Q_ij^2`다.
- 각 class term은 기존처럼 해당 forest의 represented unordered-pair mass 합으로 나눈다. 유효 positive/negative class term의 산술평균을 image loss로 쓴다. Empty image는 `Q.sum()*0`이고 batch 평균에서 제외하지 않는다.
- 추가 lambda, positive-rho auxiliary, instance-prototype loss, density prior, RGB loss는 없다. Candidate positive term 범위는 [0,4], control은 [0,1]이며 이 scale 차이를 숨기거나 gradient로 재보정하지 않는다. Global clip/optimizer는 공통이다.

Worst rho→1은 native Q→1에 충분하지만 product의 정확한 값과 동일한 목적은 아니다. 음성항이 global synchronization을 견제할 수 있다는 것은 가설이며, GT/validation으로 coefficient를 고르지 않는다. 모든 rho가 똑같은 saturated-gate source에서도 amax tie credit을 특정 component 하나에 몰아주지 않는다.

## 공통 모델·데이터·실행

Fresh original SourceT seed1을 SW135 foundation으로 native32에 매핑하고 SW171의 zero-final RF31 context encoder를 붙인다. 원본 valid3x3 CNN은 frozen, context/learned graph/native core/3 integration tensors(a_d,a_m,b; 총12 scalar elements)는 양 arm 모두 trainable이다. Source geodesic contrast는 SourceT 값으로 frozen이다. SW168/171/172/174의 학습된 weight/Adam은 재사용하지 않는다.

재사용점은 SW171 `train_prefix.py::_native_forward`, `_named_families`, `_make_optimizer`, SW134 `rollout.py::late_rollout`, SW168 forest/objective와 evaluation helper다. Native event surrogate를 유지한다. Full1024 forward, settle512, live tail64/boundary960를 공통으로 유지하며 full1024-BPTT나 event detach를 이 pilot에 섞지 않는다.

같은 검증된 ordered512 TRAIN RGB/mask IDs/cache lineage를 사용한다. 512개 전체에 대해 RandomState(171001)의 연속 두 permutation을 생성하고 각 pass를 B16으로 나눈다. **64 logical Adam updates, 1024 optimizer image exposures, 512 unique images 각각 두 번**이다. MicroB4의 image losses를 sum/16하여 누적하고 logical update당 clip1 및 Adam1회, integration projection1회다. 두 arm의 init/context seed=171001, order, source/cache hashes와 parameter order는 같다.

Fresh Adam LR: context3e-4, graph/native-core3e-5, integration1e-3. 기존 source checkpoint와 frozen base를 strict-load한다. Checkpoint는 update 경계에서 model/Adam/RNG/order cursor 및 trainer/objective dependency hashes를 보존한다. Peer 데이터, VAL RGB/masks, reserve는 trainer에서 읽지 않는다. TRAIN masks를 쓴 supervised track이며 unsupervised leaderboard로 표시하지 않는다.

## 최소 진입 검증: task proxy로 학습을 대체하지 않기

실제 source와 zero-final 모델의 같은 TRAIN RGB/RNG/B4 및 inference B1에서 gamma/A/six traces/Q와 SW165 energy/merge/labels parity를 확인한다. TRAIN B4와 VAL B1 수치 모드를 서로 bitwise 같다고 요구하지 않는다. 같은 mode의 production Q와 rho product는 원래 classifier 연산 순서로 일치해야 한다. 별도 cached rho/Q 혼합은 금지한다.

CPU unit checks는 product-dead positive edge에서 새 positive rho gradient가 살아 있는지, negative-Q loss/gradient가 기존과 같은지, amax ties/empty classes/pair mass가 명세와 같은지만 검사한다. 운영 사전등록 명확화: 별도 disposable GPU updates를 추가하지 않고, **등록64회 중 첫 두 실제 committed updates**를 실제 gradient admission으로 사용한다. 양 arm은 검증된 같은 fresh source/zero-init에서 시작하며 update0 경계를 저장한다. 각 첫 두 update의 finite loss/parameter gradients, actual S까지의 registered loss gradient 및 context/graph/Kuramoto credit을 기록하고, 등록된 dead/nonfinite credit 조건이면 해당 committed boundary와 rejection을 보존하여 중단한다. 통과한 update1/2는 그대로 update64까지 이어지며 reset/restart하거나 노출을 다시 더하지 않는다. 첫 step zero-final encoder의 hidden gradients=0은 예상하며 두 번째 step에서 실제 경로를 본다. Dendric/membrane/각 integration scalar의 raw credit·event occupancy/active-gate transitions를 정확한 이름으로 기록하되, 모든 scalar에 임의의 최소 norm을 요구하지 않는다. 실제 SW148 boundary save/load/Adam/cursor 복원은 CPU regression으로 검사한다. 별도 GPU source-restoration smoke를 수행했다고 주장하지 않는다.

Nonfinite 또는 새 positive→actual S 경로가 완전히 dead이면 구현/과학적 진입 실패로 닫고 lambda/LR/clamp를 구제하지 않는다. 그 외에는 **8 TRAIN loss 감소를 64-update 학습의 필수 gate로 삼지 않는다.** Fixed TRAIN8 positions256..263의 initial/final native Q loss, candidate loss, clamp blocking, rho 분포, 두 mask FG/IoU를 기록하지만 checkpoint/arm/order 선택에 쓰지 않는다. 두 arm 모두 사전 고정64 updates를 완료한다.

## 고정320 task 판단

추론은 학습 loss와 독립적으로 원래 native positive-product Q, 실제 prepared coupling A, 학습된 graph의 actual exp(log_coupling_gain), 동일 SW165 induced 4-neighbor greedy solver를 쓴다. SW173의 frozen-bound-gain 진단 정책을 learned-model evaluation로 몰래 가져오지 않는다.

두 arm마다 아래 두 output을 함께 제출한다.

1. **Conditional:** frozen original SourceT의 source_F와 learned Q/A를 사용하는 attribution output. 원본 SourceT rollout 또는 검증된 그 support artifact가 추가로 필요하며 extra compute를 명시한다.
2. **Standalone:** 같은 learned rollout의 native CPU-QCC foreground와 learned Q/A를 사용하는 practical output. Teacher/source F가 inference에 필요하지 않다.

IDs1320..1639, native128→4px modal GT/32grid, 기존 foreground/background 정책과 세 metric을 유지한다. 모든 최종 prediction을 hash freeze한 뒤 GT를 읽으며 initial SourceT/SW164 seed1 reference .7709320986/.6149934303/.6305234820을 bound archive로 검증한다. SourceT의 원래 QCC-only .755273과 SW165 solver output을 혼동하지 않는다. Frozen SW163 learned-head 실패 기록은 historical reference일 뿐 이번 matched control이 아니다.

확대 조건은 **conditional와 standalone 각각** candidate FG가 (a) SW164/source seed1, (b) 새64-update native_Q control보다 높고, common-image paired10,000 bootstrap(seed175) delta의 95% lower bound가 각 비교에서 >0인 것이다. 두 output 모두 foregroundIoU/objectIoU means가 source 및 matched control보다 낮지 않아야 한다. 모든 mean/CI/실패 arm을 보고하며 좋은 head만 고르지 않는다. Initial/final actual labels의 변경량도 기록하되 작은 변경을 곧 성능 개선으로 해석하지 않는다.

64-update 결과가 이 gate를 실패하면 이 recipe를 닫고 passes/LR/seed/lambda sweep이나70k 확대를 하지 않는다. 실패는 모든 component objective/학습 예산의 불가능성 증명이 아니다. 통과한 경우에만 seed0/2의 동일 recipe와 all-three aggregate를 새로 등록하고, 그 뒤 fresh native32/70k 및 unique-data scaling 실험을 정한다. .87 평균, all-gate causal contribution, fresh native70k superiority는 이번 seed1 supervised pilot만으로 입증되지 않는다.

````

</details>

## EVALUATION_QUEUE_KO.md

<details>
<summary>EVALUATION_QUEUE_KO.md</summary>

Source SHA256: `cb132e896d4f74d627291d239c61628a09127109a039f669b961459b9aa830a1`

````markdown
# SW0175 학습 후 자동 평가 연결

두 V2 학습이 모두64 updates를 완료하면, 독점 유휴 GPU를 확보해 각 모델의
고정320 예측을 자동 실행하도록 두 prediction supervisor를 등록했다.
별도 CPU scorer는 두 모델의 예측 파일과 protocol이 고정·검증된 뒤에만 GT를 읽는다.

로컬·서버 평가 검사 각10개와 실제 Source-T/SW164 reference·baseline score SHA
검증을 통과했다. 학습 중인 V2 source는 수정하지 않았다.
원본 foreground를 유지하는 conditional 출력과 같은 learned rollout의 QCC foreground를
사용하는 standalone 출력을 모두 제출한다. 추론은 실제 학습된 graph gain을 사용한다.
checkpoint state 외의 mean/std/clip·patcher·frozen geodesic 구성도 학습 manifest와 비교한다.

세 개의 실제 coordinator PID와 선행 작업 대기 상태를 확인했다. 이는 큐 등록 증거이며
평가 점수가 아니다. seed1 supervised pilot의 고정320 점수와 paired bootstrap이
완료된 뒤에만 다음 확대 여부를 판단한다. SW165 최고 기록은 유지한다.

증거: `results_archive/evaluation_queue_started_20261015/`.

````

</details>

## EVALUATION_RECOVERY_KO.md

<details>
<summary>EVALUATION_RECOVERY_KO.md</summary>

Source SHA256: `efcdb91f140b05f95ddae4a598d7d4c06160729d688703b4a4ed84fc1fd01677`

````markdown
# SW175 학습 완료와 고정 320장 평가 복구

native_Q 대조군과 critical_component 후보 모두 seed 1의 **64 optimizer updates를 완료**했다. 두 최종 checkpoint와 optimizer/boundary, 원본 학습 소스는 보존했다. 평가 복구에 따른 추가 학습은 없다.

이전 평가 큐는 입력 검증에서 종료됐다. 평가 코드가 source-cache manifest SHA를 학습 binding의 최상위에서 읽었으나 실제 train_v2 생산자는 `bindings.source_proof.source_cache_manifest_sha256`에 기록한다. 실제 캐시 변경은 없었고 두 모델 모두 중첩 provenance와 현재 캐시 SHA가 일치한다. 예측 및 VAL 정답 채점 전 실패했으므로 그 큐의 성능 점수는 없다.

수정은 새 `predict_evaluate_v2.py`, `prediction_queue_v3.py`, `score_queue_v3.py`와 새 출력 경로에 적용했다. 기존 실패 버전과 로그를 보존했다. 새 코드는 두 모델의 중첩 SHA를 모두 검증하며 검증을 생략하지 않는다. 로컬·서버에서 각 11개 CPU 계약 검사를 통과했다. 특히 서버에서는 **완료된 실제 두 checkpoint로 전체 prepare_prediction_inputs_cpu 및 strict 모델 로드, 최종 state SHA, 고정 encoder/전처리/geodesic 구성 일치를 확인한 후** GPU 평가를 등록했다. CPU 검증에서 VAL mask는 읽지 않았다.

고정 native32 320장에 대해 conditional Source-T foreground와 standalone learned-Q foreground의 두 endpoint를 각각 생성한다. 두 모델의 전체 예측을 확정한 뒤에만 정답 mask를 열고 세 지표 및 사전 등록된 paired bootstrap을 계산한다. 후보의 두 endpoint 모두 기존 SW164 seed1과 대조군을 넘는 등록 조건을 만족해야 다른 seed로 확장한다. 현재는 최종 평가 대기이며 최고 모델 SW165를 교체하지 않는다.

실제 완료 학습 manifest, TRAIN8 진단, 이전 큐 terminal 증거, 수정 소스 SHA, 전체 실제 CPU 로드 결과 및 새 PID 관측은 `results_archive/completed_training_evaluation_recovery_20261015/`에 기록했다. TRAIN8은 학습 진단이며 검증 성능으로 사용하지 않는다.

````

</details>

## NEXT_HYPOTHESIS_KO.md

<details>
<summary>NEXT_HYPOTHESIS_KO.md</summary>

Source SHA256: `d7219f081502e9ecc2d200016e3aa75374f028b461678d33f12fc1827ac67dae`

````markdown
# SW175 평가 후 조건부 다음 가설

이 문서는 다음 후보에 대한 설계 검토이며 새 실험의 실행 또는 성능 결과가 아니다. SW175 고정320 평가가 완료되기 전에는 해당 후보를 학습하지 않는다.

SW173에서는 전체1024-step gradient가 tail64와 크게 다른 방향이고 크기도 컸다. SW174에서 raw-event surrogate backward 경로만 제거해도 폭증과 음의 방향 정렬이 거의 남았다. 따라서 event surrogate 하나가 원인이라는 설명은 기각하며 같은 작은 loss 조정을 반복하지 않는다.

SW175 후보가 두 endpoint의 등록 조건을 통과하지 못하면, 다음 단일 후보는 **전체 settled frames에 local phase-parameter gradient를 주는 학습 정책**이다. 각 Kuramoto 전이의 theta_prev 입력만 detach하고 그 step이 내보낸 theta output은 live history에 둔다. 실제 코드에서 과거 history는 다음 Kuramoto 입력으로 재사용되지 않고 sinusoidal carrier/delayed gate에서 읽히므로, Kuramoto 전이 사이의 연속 Jacobian만 끊으면서 각 delayed gate에서 해당 local step의 parameter로 흐르는 gradient는 유지할 수 있다. D/M/event/reset의 장기 gradient까지 모두 제거한다는 주장은 하지 않는다.

Forward 값은 바꾸지 않는 approximate recurrent gradient 정책이다. six traces, actual Q/A 및 두 mask가 원본과 같은지 먼저 확인하고, 작은 graph 검사로 successive theta gradient 차단과 delayed gate의 local parameter gradient 보존을 증명해야 한다. 이후 실제 GPU에서 gradient 크기·방향과 첫 update의 자원 사용을 측정한다. 안정화 효과는 아직 입증되지 않았다.

loss는 기존 critical-component, source 및 512장×2 TRAIN 순서, B16/micro4, optimizer/LR/clip, 64 updates를 고정한다. 완료된 SW175 critical-component 모델은 해당 조건과 초기/RNG/boundary lineage가 모두 맞는 경우에만 tail64 대조군으로 재사용한다. 후보는 실패한 가중치가 아닌 동일한 fresh source에서 시작한다.

확대는 proxy loss 감소로 결정하지 않는다. 동일 고정320에서 conditional/standalone FG-ARI가 기존 source와 tail-control 대비 paired bootstrap 조건을 통과하고 두 IoU를 유지해야 한다. 실패하면 이 정책도 닫는다. 성공해도 모든 gating의 필요성, FG-ARI .87, native32 fresh70k 및 데이터 증가 효과는 별도의 검증 대상이다.

````

</details>

## RECOVERY_KO.md

<details>
<summary>RECOVERY_KO.md</summary>

Source SHA256: `e4d9220ad148ce7490df65ecf723ae1076cae1ea776cc34fe10346313b443845`

````markdown
# SW0175 첫 실행 입력 adapter 오류

두 arm은 GPU 1/2에서 원본 모델을 구성했으나, 초기 TRAIN8 진단의
`forest_routes`에서 `support must be boolean`으로 종료됐다.
실제 캐시의 `source_support`는 object label grid이며 foreground boolean이 아니다.
기존 SW171 caller의 `np.asarray(support) > 0` 변환을 SW175에서 빠뜨린 구현 오류다.

로컬·서버 CPU 13개 검사는 통과했지만, 해당 검사는 실제 캐시의 multiclass support를
새 loss caller에 전달하는 경우를 다루지 못했다. source/cache admission 성공을
실제 입력 adapter 검증으로 확대 해석한 점을 바로잡는다.

기존 로그·초기 checkpoint·source manifest는 보존한다. 첫 두 worker와 parent의
종료를 확인하고, checkpoint 메타데이터로 두 arm 모두 update 0임을 확인한다.
이는 모델 성능의 음성 결과가 아니다. 세 seed 최고 기록은 SW165 그대로다.

수정은 caller의 foreground 변환 복구와 새 실행 namespace 분리에 한정한다.
모델·loss 수식·TRAIN IDs·64-update schedule은 바꾸지 않는다.
재실행 전 multiclass support 회귀 검사와 실제 TRAIN cache의 caller 검사를 추가한다.
첫 실행 증거는 `results_archive/failed_v1_20261015/`에 기록한다.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `c90e6865a3554d1a7d0ea151caada648920e34ad7238b3f05708bedb79859537`

````markdown
# SW175 결과: critical-component 손실 후보 종료

두 seed1 모델 모두 64 updates를 완료했다. 동일한 고정320 이미지와 native32 계약으로 예측을 확정한 뒤 정답을 채점했으며, 완료 예측과 정답에서 세 지표·bootstrap을 독립 재계산했다. 아래 결과는 세 seed 평균이 아니다.

| 모델/출력 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0164 | 0.770932 | 0.614993 | 0.630523 |
| native_Q_conditional | 0.728783 | 0.614993 | 0.595189 |
| critical_component_conditional | 0.725044 | 0.614993 | 0.579214 |
| native_Q_standalone | 0.699792 | 0.448026 | 0.538209 |
| critical_component_standalone | 0.674915 | 0.399856 | 0.509791 |

conditional은 기존 Source-T foreground를 유지하면서 학습한 실제 Q/A로 물체를 연결한다. standalone은 학습한 spike trace의 native QCC로 전경도 생성한다. 두 경우 모두 후보가 기존 SW164 seed1과 대조군을 넘지 못했다. conditional 후보의 대조군 대비 FG 차이 95% CI는 [-0.012715, 0.005511], standalone은 [-0.039863, -0.010130]이다.

조건부 목표 두 가지를 모두 실패했으므로 seed0/2 또는 70,000장으로 확대하지 않고 이 손실 변경을 종료한다. 기존 최고 SW165는 유지한다. 손실을 바꾸는 것만으로 학습의 일반화나 spike foreground가 개선된다는 증거는 없다. 이 결과만으로 모든 gating이 불필요하다고 결론내리지는 않는다.

고정 TRAIN8 사후 진단은 학습 전후의 gate/spike·연결·partition 변화를 설명하기 위한 것이며 모델 선택이나 종료 판정을 뒤집는 데 사용하지 않는다. 다음 후보는 NEXT_HYPOTHESIS_KO.md에 기록한 전체 settled-window local phase credit이다. 작은 loss/threshold/epsilon 재조정은 이번 recipe의 구제책으로 진행하지 않는다.

검증된 최종 score SHA: `7311ee86f0ad30a3670edf8bf798f379f200fcd0372cd199e5b0af95fbe5fd02`. 원본 결과·독립 검증·terminal PID 관측은 `results_archive/completed_score_20261016/`에 보존했다.

````

</details>

## TRAIN8_AUDIT_KO.md

<details>
<summary>TRAIN8_AUDIT_KO.md</summary>

Source SHA256: `5f05c3b02536d0e3a92de8bdf2ba96405592ec2f98df696a17073a98e8a888db`

````markdown
# 사전 등록된 TRAIN8 사후 진단 준비

`train8_audit.py`는 TRAIN512의 고정 위치 256..263에 대해 원본 Source-T+zero-context, 완료된 native_Q, 완료된 critical_component를 비교한다. 새 학습이나 checkpoint 선택에는 사용하지 않는다.

기록할 항목은 두 손실, conditional/standalone의 세 지표, 양성 route의 pair mass와 component rho 분포, 실제 Q 곱의 모든 component 편미분이 0인 pair mass, gate/spike의 평균 진폭과 event 비율, 원본 대비 label permutation에 불변인 partition ARI다. Q=0이라는 사실만으로 모든 gradient가 막혔다고 판정하지 않는다. 정답은 이미 등록된 TRAIN 캐시에 한정한다.

`train8_queue.py`는 수정된 score_queue_v3의 실제 완료 상태와 score SHA, 공개 score validator를 확인한 뒤에만 독점 GPU를 확보한다. 고정320 평가보다 먼저 GPU를 사용하지 않는다. 모델별 state SHA가 추론 전후 같아야 하며 추가 optimizer update는 0이다.

로컬 CPU 검사 12개가 통과했다. 성공 경로의 실제 supervisor 호출을 mock으로 끝까지 실행해 score에 묶인 worker command가 Popen까지 전달되는지 확인했고, launch 전 오류의 terminal 기록과 기존 상태 보존도 확인했다. **이는 서버 실행이나 실제 GPU 진단 완료의 증거가 아니다.**

현재 새 파일의 서버 전송 시도는 SSH banner timeout으로 실행되지 않았으며, 실제 서버 CPU 입력 준비 및 큐 등록은 아직 확인되지 않았다. 연결 복구 후 실제 완료 모델과 TRAIN 캐시로 CPU admission부터 검증하고 등록한다. 실행 중인 고정320 평가와 학습 산출물은 건드리지 않는다.

## 후속 완료 (2026-10-16 UTC)

연결 복구 후 네 파일의 실제 배포와 서버 CPU 12개 검사, 실제 캐시 입력 준비를 확인했다. 고정320 채점이 완료되고 공개 validator로 재확인된 뒤 GPU1에서 TRAIN8 진단을 실행했다. 부모·worker 모두 정상 종료했고 immutable 보고서 SHA가 terminal 상태와 일치한다. 각 행에서 평균을 독립 재계산했다. 위의 서버 미실행 기록은 당시 상태이며 현재 완료 증거는 `results_archive/completed_train8_audit_20261016/`에 있다.

| 모델 | TRAIN8 conditional FG | TRAIN8 standalone FG | standalone foreground IoU | 양성 pair mass의 zero-partial 비율 |
|---|---:|---:|---:|---:|
| critical_component | 0.604887 | 0.606784 | 0.385742 | 0.113366 |
| native_Q | 0.561564 | 0.581960 | 0.564056 | 0.051811 |
| source_T_zero_context | 0.645323 | 0.643618 | 0.597822 | 0.044274 |

이 표는 학습에 사용된 8장의 설명용 진단이며 일반화 성능이 아니다. 최종 판정은 RESULT_KO.md의 고정320 음성 결과를 따른다. 두 loss와 event/rho 원자료는 함께 보존했으며, 이 결과만으로 gating의 인과적 필요성을 입증하지 않는다.

각 component의 active gate 조건부 spike event 비율을 8장 평균하면 원본·대조군·후보 모두 0.9998 이상이다. 따라서 이번 학습으로 membrane의 binary event가 새롭게 유용한 선택성을 얻었다는 증거도 없다. 실제 S는 연속 gate 진폭을 포함하므로 이 비율을 S의 평균 진폭과 혼동하지 않는다. 후보는 양성 route의 모든 component 편미분이 0인 pair mass 비율도 원본 0.0443에서 0.1134로 증가했다. 이는 해당 8장과 해당 route에서의 관측이며 전체 데이터나 모든 gate의 원인을 단정하는 수치는 아니다.

````

</details>

## TRAINING_START_KO.md

<details>
<summary>TRAINING_START_KO.md</summary>

Source SHA256: `12d808ec1089377df8ecb7abc68d3489532fcd6e9c0eddaa0c5c9bacda732fd3`

````markdown
# SW0175 비교 학습 시작

V1의 boolean-support 변환 누락을 새 V2 source에서 수정했다. 모델·loss 수식·
TRAIN IDs·학습 순서는 유지하며 실패 소스와 update0 checkpoint를 보존했다.

로컬과 서버에서 15개 CPU 검사를 통과했다. 추가로 실제 TRAIN cache의
8개 multiclass support와 feasible target을 새 caller에 전달해, 명시적 `labels > 0`
결과와 두 loss가 일치하고 gradient가 유한함을 확인했다. 이 검사의 Q/rho는 합성 입력으로,
성능 평가나 실제 source rollout 결과가 아니다. 검증 코드를 결과 폴더에 함께 보존한다.

두 arm은 동일 원본 SourceT와 zero-final context 초기 상태에서 시작했다.
원래 native-Q loss와 critical-component positive loss를 각각 512 TRAIN images,
2 passes, B16/micro4, 총64 updates로 비교한다. 실제 GPU 1/3 worker가 확인됐고,
두 arm의 첫 committed update도 확인했다. cache 우회 환경은 실제 worker에서 확인했다.
첫2 updates의 credit 검사는 학습에 포함하며 추가 GPU updates를 수행하지 않는다.

고정320 평가가 완료되기 전까지 성능 향상이나 새 최고 모델을 주장하지 않는다.
Supervised seed1 pilot이며 70k 신규 학습 또는 세 seed 결과가 아니다.
현재 최고는 SW165의 0.8043 / 0.6100 / 0.6511로 유지한다.

실행·검증 증거: `results_archive/started_v2_20261015/`.

````

</details>

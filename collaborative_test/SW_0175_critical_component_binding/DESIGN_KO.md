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

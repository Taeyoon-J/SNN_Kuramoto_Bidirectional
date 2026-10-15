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

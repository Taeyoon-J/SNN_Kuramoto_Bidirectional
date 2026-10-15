# SW0173 — native Q gradient의 horizon/direction fidelity 1회 감사

2026-10-15 prospective diagnostic design. 구현·GPU 실행 없음. SW171/172는 등록된 prefix 조건 미통과로 닫은 상태이며 뒤집지 않는다. Highest165는 유지한다. 이 단계는 **TRAIN-mask supervised objective의 진단**, 새 모델 학습/VAL 성과가 아니다. 새 loss/threshold/surrogate/encoder/GT 범위를 바꾸지 않는다.

## 판단할 질문

171/172의 native Q task gradient는 실제로 nonzero였지만, 작은 unseen TRAIN8 mean loss는 initial보다 나빴다. Aux instance loss 감소와 Q 악화는 경로 간섭/gradient 오류의 증명이 아니다. Source `SW0134_native_spike_binding/rollout.py:late_rollout`은960에서 모든 recurrent states를 detach하지만 실제 parameter update는 다음 full1024 rollout의 burn-in도 바꾼다. 이 누락이 실제 descent direction을 바꿀 가능성을 **full1024 BPTT와 같은 forward에서** 검정한다. MG의 signed surrogate 및 critical-route 변화도 대안 원인이다.

먼저 새로운8-update sweep을 하지 않는다. Native actual forward를 고정한 **1회 finite optimizer-direction audit**로 full-horizon 학습이 해볼 만한지를 판정한다. Lower proxy loss만으로 FG-ARI .87/스케일 향상을 주장하지 않는다.

## source·데이터·resource

SourceT seed1/원래 membrane-reset-threshold/171 zero-final RF31 context model을 fresh init한다. 169/170 모델이나 completed171/172 trained weights는 load하지 않는다. Actual171 `Q_only`와 같은 trainable families/LR/context seed171001, frozen original base, fixed T contrast와 registered stats를 사용한다. Aux instance loss/lambda는 사용하지 않는다.

Audit cohort는 original ordered512 TRAIN positions0..3, transfer probes는256..259다. 모두 TRAIN이고 VAL1320..1639/금지 reserve90640..90959는 열지 않는다. 실제 RGB/GT32/source-F/feasible-target lineage를 검증한다. F/target은 source cache의 original CPU-B4 support를 그대로 쓴다. **Forward arithmetic은 이 감사 내내 B1**이며 cached GPU-B4 Q와 같다고 주장하지 않는다.

첫 번째 cohort image에서 native32 B1/full1024 BPTT의 actual backward 한 번을 resource probe한다. 기존 factorized coupling/geodesic chunk/checkpoint를 그대로 사용하고 peak allocated/reserved/time을 기록한다. OOM/nonfinite이면 이 full-BPTT 경로는 resource failure로 중지한다. 다른 N/T/precision/horizon으로 몰래 줄이지 않는다. 나머지 비용은 이 측정으로 먼저 공개한다. Fixed4+4 범위를 임의로 늘리지 않는다.

## 두 gradients, 같은 값

`late_rollout(...,total_steps=1024,live_tail_steps=64)` 대 `live_tail_steps=1024`다. Existing API는 이미 boundary0/full-grad를 지원한다. Static gamma/graph/drive/coupling도 두 경우 grad enabled로 준비한다. Full mode는 theta initialization부터 모든 state credit가 live이고 tail mode만960 detach한다. No new forward equations.

같은 source/input/RNG/B1에서 all-six trace values/actual Q/actual A/initial SourceT-SW165 partition과 loss가 **정확히 같아야** 한다. Same-source 반복 forward 두 번도 기록하여 numerical jitter를 관측한다. 값이 다르면 gradient horizon 비교로 해석하지 않고 technical stop한다. Cache layout/CPU-support 모드를 동일성 검사의 상대값으로 혼동하지 않는다.

각 horizon에서 네 cohort images의 Q-only `constrained_maximin_loss`를 image mean으로 backward/accumulate한다. 즉 **virtual audit B4 with micro1**이다. 원래 training B16 gradient 자체라고 부르지 않는다. Parameter union/aliases는 helper `gradient_fidelity.py`의 실제 API로 검증한다. Kuramoto/actual `dendric_layer`/membrane/context/graph/a_d/a_m/b의 raw gradient norms와 두 horizons의 cosine를 따로 보고한다.

## virtual Adam directions와 physical one-sided FD

두 gradient에서 같은 original state에 fresh Adam(global clip1,171 group LRs)을 **복제상태에서 한 번** 적용하고 기존 a_d/a_m projection까지 거친 delta64/delta1024를 만든다. 원본 weights/buffers/RNG와 optimizer 상태는 복원하며 checkpoint 학습 update로 저장하지 않는다. Directions는 실제 projected parameter delta이고 synthetic norm direction이 아니다.

각 direction마다 **미리 고정한 s={1/16,1} 두 값 모두** theta+s*delta를 평가한다. s를 보고 learning rate를 고르지 않는다. Source a_d/a_m=0에서 theta-s*delta는 physical [0,1] 밖일 수 있으므로 실제 model audit는 **one-sided FD**다. 기존 pure CPU helper의 symmetric central FD를 그대로 쓰거나 minus perturbation을 재projection하여 central FD라고 부르지 않는다. Delta가 이미 projected이므로 segment s∈[0,1]는 feasible하며 evaluation마다 다른 projection을 적용하지 않는다.

Predicted change P_h(s)=s*sum(g_h*delta_h), actual change D_h(s)=L_Q(theta+s*delta_h)-L_Q(theta). 실제 evaluation은 **unmodified full1024 forward**이며 전체 trajectory를 다시 계산한다. Grad-enabled/native kernel mode를 baseline과 맞추고 loss/values의 provenance를 유지한다. Same RNG/no stochastic data transform이다. Cohort4와 transfer4에 동일 delta를 적용한다. 두 gradients의 cross dot-products도 추가 backward 없이 보고한다. 모든 trials 뒤 원본 state/RNG hash를 다시 확인한다.

각 actual Q에서 (a) forest를 다시 구한 native loss, (b) original source forest edges/pair weights를 고정한 loss를 둘 다 계산한다. 추가 rollout은 필요 없다. Predicted gradient는 source에서 선택된 forest의 subgradient이므로 route switching과 parameter/surrogate fidelity를 구분한다. Actual native standalone 및 source-F conditional SW165 masks의 **TRAIN** FG-ARI/두 IoU도 initial/trials 모두 기록한다. 작게 변한 loss만으로 masks가 개선됐다고 쓰지 않는다. Hard labels가 한 step에서 변하지 않는다고 gradient를 무효로 단정하지도 않는다.

Component membrane minus actual .06*exp(b)의 margin 분포와 g>0에서 actual E를 기록한다. Source/trial의 공통-active samples에서 event flips를 센다. E가 바뀌면 그 finite-step 변화에 threshold crossings가 포함되므로 smooth local derivative proof로 쓰지 않는다. g=0이 만든 S0를 event crossing으로 세지 않는다. Source sampled E가 all-on이면 그 범위만 쓰고 dataset 전체로 일반화하지 않는다.

## 같은 forward에서 clamp 영향도 측정

Production과 **같은 B1 component loop**로 signed rho를 계산하고 positive-clamped product Q의 same-mode parity를 확인한다. Critical positive forest edge에서 product Jacobian은 실제 autograd로 구한다(clamp boundary derivative를 임의의 >0 indicator로 대체하지 않음). 모든 component partial이0인 edges의 (1) raw edge fraction, (2) represented positive pair mass fraction, (3) positive-loss |dL/dQ| blocked mass fraction을 따로 보고한다. Weight가 이미 포함된 dL/dQ에 pair mass를 다시 곱하지 않는다.

Blocked edge 수만으로 중요 supervision이 사라졌다고 주장하지 않는다. SW155/SW160은 이미 broad same-FG/BG positive-rho loss를 포함했다. 이 결과 없이 이름만 바꾼 component-positive loss를 바로 재시도하지 않는다. 이 감사는 새로운 component/signed-Q/clamp-removal intervention을 승인하지 않는다.

## 사전 결정과 끝나는 조건

Full-horizon 다음 pilot을 제안할 **필요 조건**은 full1024 direction의 predicted change<0, actual recomputed-route mean loss가 cohort/transfer에서 모두 s=1/16 및1에 감소, transfer의 decrease가 tail64 direction보다 크다는 것이다. Fixed-route 결과도 같은 방향이어야 한다. 변화가 observed same-state repeat jitter보다 작으면 inconclusive로 표시하고 epsilon/sample을 확대해 유리한 결과를 고르지 않는다. Small-scale event flips가 있으면 truncation 효과만 분리했다고 주장하지 않는다. Routing/activation crossing 또는 nonlinear finite-step effect의 가능성을 명시한다.

* Full gradient만 유효한 실제 descent를 복구하면: **새로 등록한** full1024-vs-tail64 matched short training을 제안한다. 같은 B1 microbatch accumulation/B16 source/order/LRs/Q-only objective, original SourceT initialization이어야 한다. B4→B1 차이가 있으므로 old171 control은 자동 재사용하지 않는다. Source first64 two passes=8updates만 먼저 수행한다.
* 둘 다 실제 descent가 되는데 transfer가 나쁘면: source fit과 sample transfer 문제로 보고, full-horizon을 해결책으로 자동 확장하지 않는다.
* Full도 실제 방향을 복구하지 못하거나 surrogate/route ambiguity/resource가 남으면: 이 full-BPTT 학습 제안은 중지한다. 추가8updates·새 epsilon·positive-rho/threshold rescue를 자동 queue하지 않는다. 명확한 negative diagnosis 자체가 이번 테스트의 종료 결과다.

필요 조건 통과 후의 새8update pilot은 unseen **TRAIN64 positions256..319**의 actual connected-partition FG-ARI를 source와 matched control에 대해 비교해야 한다. Q-loss 감소만을 유일한 continuation gate로 다시 쓰지 않는다. 두 support outputs/FG·IoU와 paired uncertainty를 사전 등록한 뒤 평가하며 closed171/172를 소급 통과로 바꾸지 않는다. 그 실제 task gain이 없으면64updates/seed/native70k로 확장하지 않는다. VAL fixed320 success/.87/gating necessity/scaling은 이 감사나 TRAIN 결과로 주장하지 않는다.

원본 source functions는 불변이며 새 helper/measurement만 별도 namespace에서 실행한다. Generated outputs는 중앙 `collaborative_testing/prepare_output.ps1`과 immutable receipts를 따른다. Huge trajectory/manifest를 console에 dump하지 않고 fixed metrics/state/vector hashes와 필요한 arrays를 artifact로 저장한다.

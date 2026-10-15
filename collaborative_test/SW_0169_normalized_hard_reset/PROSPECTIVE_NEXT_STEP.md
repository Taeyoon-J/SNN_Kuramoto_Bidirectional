# Conditional next hypothesis — causal synaptic-current readout

2026-10-15. **조건부 아이디어이며 아직 새 실험을 등록하거나 실행하지 않았다.** SW169 full320 판정과 독립 검증 후 최종 설계를 검토한다. 기존 source/recipe/실행은 변경하지 않는다. 현재 최종 평가 점수나 후보의 성공 결과를 주장하지 않는다.

## 현재 증거의 범위

`results_archive/completed_training_20261015/server_records.json`의 fixed TRAIN8에서 두 arm initial event density 평균은 .06143558, final live .05846727 / detached .05785352이다. Q mean은 initial .00178529, final .00185593/.00171727이다. Initial standalone foreground patch 수 min/mean/max는0/2/8, final live0/5/16 및 detached0/4/16이다. 전 이미지가 정확히0이라고 쓰지 않는다. 이 기록의 fixed8 maximin loss는 .49625620→.49695632/.49814903이며 epoch minibatch loss와 다른 통계다. Dataset 전체 또는 VAL 성능의 결론은 아니다.

`spike_classifier.py:spike_synchrony_affinity`는 시간 평균을 뺀 component별 Pearson을 positive-clamp한 후 네 값을 곱한다. 동일 rho라면 Q>.5에는 각각 rho>.8409가 필요하다. Q mean에는 비퇴화한 diagonal의 약1/1024 기여도 포함된다. 따라서 sparse coding에서 off-diagonal 관계가 매우 약하다는 증거는 있지만, within-object timing jitter가 원인이라는 증명은 없다. Sparse code 자체의 정보 손실과 instant zero-lag readout mismatch를 구분해야 한다.

## 고정된 한 가지 구조 가설

원본 SourceT에서 시작한 SW169 normalized hard-reset 모델에 actual emitted S=E*g만 입력받는 causal state를 둔다:

`Z_0=0; Z_t=exp(-1/16)*Z_(t-1)+(1-exp(-1/16))*S_t`.

16-step time constant는 이 아이디어에서 고정하며 GT/활동률/VAL로 고르지 않는다. 다른 width·threshold·clamp 조합을 탐색하지 않는다. Z는 **actual-spike-derived synaptic current**이며 binary spike라고 부르지 않는다. RGB/gamma/theta 또는 직접 membrane 값을 readout에 추가하지 않는다. Raw S의 기존 dendrite feedback과 raw E의 hard reset은 그대로다.

Z는 full1024 동안 계산하고 settled512에서 기존 centered positive-product affinity Q_Z를 만든다. Maximin TRAIN loss와 SW165 signed energy/induced-connected solver 모두 같은 Q_Z를 사용한다. Actual A/gain의 경로는 그대로다. Control은 동일 모델의 identity synapse Z=S다. 두 arm 모두 encoder/graph/core/기존12 scalars, TRAIN512, B16 micro4,64updates, 기존 loss/optimizer/order,full1024/settle512/tail64를 공유한다. Z도960 boundary에서 detach하며 이후 live recurrence credit는 유지한다. 새 decoder/head/목표밀도/auxiliary loss는 없다.

Conditional-support SourceT-F 진단과 standalone own-Q_Z-QCC 출력은 모두 유지한다. Native32/modal4px/fixed320/세 지표/GT-last/금지 reserve 조건은 바꾸지 않는다. Source/base event-history의 변화는 SW169 구조에서 오고, 새 비교가 분리하는 것은 causal synaptic readout과 그 학습 credit다. Filtering이 연속 gate/rate만 복원하여 binary-event/history 기여가 여전히 없을 가능성을 명시한다.

## 판별·최소 admission·대조군 재사용

최종 학습 모델뿐 아니라 **동일 init, optimizer0의 filter-only endpoint**도 사전 명시해야 한다. Frozen initial identity/filter 양쪽을 동일 actual rollout에서 읽고 predictions를 먼저 고정한다. 이것으로 단순 readout transformation과 학습 후 추가 개선을 분리한다. 초기 endpoint를 보고 width/모델/checkpoint를 선택하지 않는다. 최종 설계에서 평가 순서와 train expansion gate를 먼저 확정한다.

TRAIN-only feasibility는 source/cache/actualS lineage, identity branch의 기존 forward/Q/loss parity, full1024 causal filtering/settle/boundary 처리, finite·비상수 Z/Q(상수/allzero 출력이 아닌지), actual Q_Z→Z→S→trainable families의 finite actionable gradients, 동일 init 두 disposable logical updates의 실제 optimizer changes를 확인한다. Direct state partial은 실제 state references로 검사한다. Fixed8의 positive/negative critical-edge distributions와 loss는 기술하되, 아직 보지 않은 candidate의 task score를 위한 숫자 cutoff를 여기서 만들지 않는다. Admission 성공은 object binding/gating efficacy의 증거가 아니다. Candidate를 본 뒤 cutoff를 만들어 통과시키지 않으며 최종 등록에서 필요 조건을 먼저 고정한다.

Identity 학습은 SW169 `history_live`를 재사용하면 중복64updates를 줄일 수 있다. **실제 init/weights, raw forward, identity Q/loss, data IDs/order, B4 grouping, optimizer states/steps/projection, RNG 및 truncation까지 동일함이 검증될 때만** 가능하다. 새로운 branch가 RNG/normalization/loss arithmetic을 바꾸거나 해당 증명이 없으면 fresh matched identity control이 필요하다. 실패한 control을 다른 seed나 checkpoint로 대체하지 않는다.

최종 설계에서 filter-only 대비 learned improvement와 identity 대비 효과, SW165 baseline 보존 및 두 출력의 paired FG/IoU gates를 명시한 후만 실행한다. 성공하더라도 E-forced1/history/reset 실제 동역학 개입 없이 발화/메모리의 필요성을 주장하지 않는다. .87/세 seed/fresh native32 70k/positive data scaling/unsupervised 성과는 각각 별도 증거가 필요하다.

# SW0218: 실제 production partition의 energy ranking 학습

217에서 actual Q의 같은-object pair 정렬 AUC는 .9237-.9778이나, false merge의
양수 native energy 비율은89.59-99.58%였다. 같은 에너지를 높이는 분할은 FG를
개선하지 못했다. 고정한 에너지 optimizer 추가 탐색 대신 실제 readout이 쓰는
Q/A/gain의 에너지 순서를 TRAIN 정답과 맞춘다.

## 기존 실패와 차이

191은 잘못된 쌍의 component rho 제곱/production Q 제곱 surrogate였다.
199는 최종 Q의 pair-balanced Bernoulli loss,204는 component별 prototype QCE,
206과214-216은 별도 관계 classifier였다. 이번에는 별도 head나 decision
threshold를 만들지 않고 원래 signed energy 자체를 미분한다. 이것이 더 좋다는
주장은 실험 전 가설이다. 작은 margin/LR/epoch rescue sweep은 하지 않는다.

## 정확히 맞춘 source와 control

204와 동일한192 TRAIN8 seed0 model, 실제 Adam moments/steps2, CPU/CUDA RNG에서
시작한다. 새 TRAIN indices8..15의 B4 두 proposal, micro1, 같은5개 learning
rate/clip1/projection, finite-Adam policy(유한 uphill step도 유지)를 사용한다.
Component objective는204의 실제 함수를 재사용하고 변경하지 않는다.
Control은 이미 완료된204이며 TRAIN/초기 model/Adam/RNG/예산이 일치해야 한다.
기존 최고 warm192와도 비교한다. 이 seed0 mini는 독립 세 seed 목표의 대체물이
아니다. 효과가 확인되기 전에 전체320/70k를 추가 등록하지 않는다.

## Frozen TRAIN reference와 objective

각 proposal의 before-step no-grad native rollout에서 production spike readout을
실제로 실행해 예측 전경과 partition을 얻는다. 그 전경 안에서 각 GT label의
4-connected 조각을 reference로 삼는다. GT0인 false-positive patch도 그대로
포함한다. 전경 누락을 정답으로 복구하거나 GT object count를 inference에 넣지
않는다. Reference는 TRAIN GT를 이용한 구체적 feasible partition이며 optimum,
성능 상한 또는 predictor가 아니다.

해당 전경에서 D_ij = same(pred)_ij - same(reference)_ij를 정한다. Q/A/gain은
원래 signed_pair_energy와 같은 식, float64, Q_EPS1e-6 및 기존NATIVE_NODES1024를
사용한다. 4096이나 전경 크기로 정규화 상수를 바꾸지 않는다.

Loss = component_QCE204 + ReLU(1 + sum(E_current * D) / scale).
scale = max(sum(abs(E_before) * abs(D)), 2 * disagreement_pairs * Q_EPS).
단위 margin과 coefficient1을 고정한다. D=0이면 미분 가능한0을 반환한다.
분자/분모 모두 unordered pair를 두 번 세므로 factor2는 상쇄된다. 실제 source
partition, reference, D와 scale은 before/gradient/after에서 고정하고 parameter
변경 후 다시 mining하지 않는다. 이는 ranking surrogate이며 structured-inference
bound나 미분 가능한 FG-ARI라고 하지 않는다.

원래 production inference, graph, encoder, 모든 gating은 유지한다. Gamma를
core와 별도로 고정하지 않는다. 기존5개 parameter family를 실제 함께 update한다.
T1024/settle512/live-tail64, gradient/no-grad 같은 RNG loss parity를 검증한다.
추가 gradient-only 데이터가 없고 logical TRAIN exposure는 inherited8+new8=16이다.
Auxiliary 자체의 emitted-spike gradient와 각 proposal의 첫 active image에서
실제 parameter-family gradient RSS도 확인한다. Auxiliary의 실제 S/parameter
credit이 적용된 step에서 확인되지 않으면 에너지 학습 성공으로 보지 않는다.
이는 gate별 인과 기여 증명이 아니다.

## 실행 및 판정

소스 전송 전 CPU operator/finite-difference/context/transaction/GT barrier tests,
서버에서도 같은 tests와 actual model/Adam/RNG CPU admission을 통과해야 한다.
GPU0..3 exclusive lease, foreign owner 양보와 proposal/이미지 checkpoint resume.
기존 파일과 이미 실행 중인 ancestor source는 바꾸지 않는다.

동일 native64 canonical128/no crop/2px modal GT, fixed16 offsets0,2,..30,
RNG193000+offset, 원래1024-step spike inference. Candidate protocol을 고정한 뒤
DEV GT를 읽고 기존204 control/warm과 paired10000 bootstrap으로 비교한다.
확대 gate는 control 대비 FG+.01 이상 및95% CI lower>0, warm 대비 FG 감소 없음,
두 비교의 ObjectIoU 감소 없음, 실제 새 update/CNN 변화/auxiliary parameter
credit이다. FG-only 유용 후보는 별도 보존한다. 최종 .90 판정은 전체320의
독립 세 seed 평균에서만 한다. Mask-supervised 실험이며 unsupervised 승리,
신규 native64 TRAIN70k 또는 데이터 scaling 증거라고 하지 않는다.

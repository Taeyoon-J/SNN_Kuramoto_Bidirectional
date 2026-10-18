# SW0220: 새 objective 전환 시 Adam 상태의 한 번 제한된 진단

218은 실제 spike/모든5개 family의 auxiliary gradient가 존재했지만 FG를 개선하지
못했다. 첫 새 proposal의 같은-RNG combined TRAIN loss는2.819465->2.947188로
상승했다. 이전 objective의 Adam 상태를 물려받은 영향인지 아직 모른다.
Loss weight/LR/margin/epoch sweep 대신 같은 gradient에서 optimizer 상태만
비교한다. Reset은 first/second moments와 step counter를 모두 초기화하므로
특히 momentum 하나만의 인과 효과라고 하지 않는다.

실제 source192 TRAIN8 seed0 model/Adam step2/원래 CPU-CUDA RNG를 복원한다.
218의 첫 새 TRAIN indices8..11, B4/micro1, 같은5개 learning rate/clip1,
원래1024/settle512/live64,218의 component+energy-ranking loss다. Before-step
production partition/reference/scale은218 helper로 한 번 mining하고 고정한다.
4장 중 마지막 값/GT object count를 inference에 넣지 않고 DEV GT도 열지 않는다.

한 번의 gradient forward/backward에서 clipped gradient를 저장한다. 두 arm은
같은 initial model/RNG, 같은 gradient/context/projection에서 시작한다.
Inherited arm은 실제 Adam moments/counter를 유지한다. Reset arm만 state를
비우며 LR/betas/epsilon/parameter groups는 유지한다. 각 임시 step 후 같은
RNG의 실제 full-forward loss를 측정한다. 새 gradient를 arm별로 구하지 않는다.
원래218의 before/inherited-after/context hashes 재현 여부도 따로 보고한다.

Surrogate gradient와 실제 projected parameter delta의 내적, family delta,
Adam counters, 실제 forward loss를 기록한다. Hard spike의 straight-through
surrogate 내적은 실제 binary forward의 미분/단조 감소 보장이 아니다.
두 arm 뒤 model/Adam/RNG를 원래 source 상태로 정확히 복원하고 digest를
검증한다. 원래 source/218 weights/checkpoint/masks는 바꾸지 않는다.
임시 optimizer proposal은2번, committed update는0이다. 고유 TRAIN4장,
native full-forward image exposures16(사전4+gradient4+두 post8), DEV 평가0.
새로운 학습 모델/FG score/70k 결과를 만들었다고 하지 않는다.

Local/server tests에서 actual nonempty Adam의 opposite-state 사례, 동일 gradient,
state/counter reset, 예외·nonfinite 시 exact transaction restoration, projection 뒤
delta, JSON serialization, 실제 control import와 GPU command를 검증한다.
실제 CPU source model/Adam/RNG/frozen-control admission 후 exclusive idle GPU0..3
queue로 실행한다. 외부 owner에게 양보하며 source fingerprint를 유지한다.

Fixed follow-up criterion은 reset의 실제 loss가 before와 inherited-after 모두보다
max(1e-7,1e-6*abs(before)) 이상 낮은 것이다. 이것은 TRAIN-only 후속 학습의
근거일 뿐 validation 개선이나 .90 달성이 아니다. 두 arm이 모두 나쁘면 이
optimizer-state recipe는 닫고 LR/beta/반복횟수 micro-rescue를 하지 않는다.
조건을 통과해도 실제 segmentation 개선은 새 matched mini와 full320 세 seed로
확인해야 한다. 현재 full320 .848615/.592533/.672047 incumbent은 유지한다.

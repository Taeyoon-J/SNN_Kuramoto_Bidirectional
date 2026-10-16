# SW0185: QCE + local spike-connectivity margins

184번 TRAIN16은 단일 원인을 입증하지 못했다. 다만 같은 물체/배경의 Q
연결이 실제 QCC .50 기준을 넘도록 하는 학습 신호가 QCE에 명시되어 있지
않다. 상대 class competition은 유지하면서 실제 공간 graph의 연결/경계에
credit을 주는 보조 목적을 별도 테스트한다. 175번 forest loss 단독 정책이나
종료한182/183의 rescue sweep으로 취급하지 않는다.

목적: 기존 class-balanced leave-one-out production-Q CE + 공간 보조 loss.
4-neighbor native32 grid에서 TRAIN modal labels가 같은 edge는 네 signed
component rho 각각을 최소 .55**(1/4)로 유도한다. 그러면 그 edge의 positive
product Q는 .55 이상이 된다. 다른 class의 edge는 production Q를 .45
이하로 유도한다. 각각 squared hinge, positive class별/negative unordered
class-pair별 평균, 두 항 평균, 보조 weight1로 고정한다. BG class0 포함.
QCE 온도.1, 기존 모델/optimizer/batch/clip/readout는 유지한다.

모델은 기존 Source097 초기화+zero context에서 seed0/1/2별 fresh 시작한다.
TRAIN512×2/B16/64 proposals, strict actual full-forward loss descent와
rollback을 사용한다. 첫2 proposals에서 실제 S→loss/four-family credit와
full-forward parity, 최종 실제 Adam 및 모델 tensor 변경을 검사한다.
기존181 QCE-only는 고정 비교 대상으로 보존한다. accepted update 수가 다를
수 있음을 명시하고, 70k 신규 학습으로 부르지 않는다.

평가는 기존 native32 fixed320 actual S→Q→QCC→SW165 readout를 그대로
사용한다. 세 seed·두 endpoint 예측이 확정되기 전 GT scoring 금지.
primary standalone 평균FG가181보다 높고 paired-image95%CI 하한>0,
Slot 대비FG/FIoU/ObjectIoU 평균 모두 엄격히 초과하면 확대한다. Conditional
FG와 세seed각각, foreground관련오류도 공개한다. 원래 .87/70k/gating 인과
기여 목표는 별도이며 이 pilot의 성공으로 대체하지 않는다.

실패하면 margin/weight/threshold 작은 sweep으로 이 recipe를 연장하지 않는다.
현재 이 문서는 preregistration이며 GPU 학습 시작 증거는 별도 기록한다.

# SW0186: 원본 RGB encoder 공동 학습 + 기존 QCE

185번 공간 보조 loss는 기존181 대비 FG 개선 gate를 통과하지 못했으므로
종료했다. 이번에는181의 QCE-only/readout와 초기 tensor를 유지하면서,
현재 source CNN을 명시적으로 frozen으로 설정한 제약 하나를 바꾼다.

Source097→native32→SourceT + 동일 zero-final RF31 context 초기화에서
원본8-channel RGB CNN의 requires_grad를 켜고 독립 source_base Adam group을
추가한다. 기존 네 group/rate는 유지하며 source_base LR은 이전160 설계의
3e−5로 고정한다. QCE 온도.1, TRAIN512×2, B16/micro4,64 proposals,
T1024/settle512/live-tail64, clip1, strict actual full-forward descent/rollback,
세 source seed0/1/2와 입력 순서도181과 동일하다. 185 보조 loss는 사용하지
않는다. CNN의 초기 tensor/forward는 원래 모델과 정확히 같아야 한다.

Source160의 다른 loss/readout 조건에서 joint encoder가 .7590이었던 결과는
이번 QCE 조건의 성공 근거가 아니다. 이번 결과로 discrete gamma ordering
최적성/순서 불변성이나 많은 데이터의 향상 효과가 입증됐다고 하지 않는다.
TRAIN mask를 사용하는 supervised continuation이며 fresh70k/unsupervised가
아니다. 원래 .87/70k scaling 및 각 gating의 유용성 목표는 계속 남는다.

실행 전 실제 세 seed CPU factory와 기존181 초기 model SHA 일치, empty Adam,
CNN forward parity와 gradient→source CNN→실제 Adam 변경, five-family 중복
없음, checkpoint restore 및 composite가 아닌 QCE full-forward parity를
검사한다. 첫 두 실제 GPU proposals에서는 loss→S, 다섯 family의 finite
positive gradient/proposed delta와 full-forward parity를 검사한다. 최종
모델은 원본 CNN의 초기/최종 SHA와 실제 Adam steps도 기록·검증한다.
Frozen 비교에서는 의도적으로 trainable인 base_encoder만 제외하며,
feature mean/std/clip, patcher, geodesic_contrast는 계속 고정한다.

평가는 기존 actual S→native Q→QCC→SW165 readout, fixed320 native32 modal
계약이다. 세 seed와 conditional/standalone 예측을 모두 확정한 뒤만 GT를
채점한다. Primary gate는 standalone FG mean>SW181 mean 및 paired-image
95%CI 하한>0, Slot 대비FG/FIoU/ObjectIoU 세 mean 엄격 초과다. 원본 CNN의
실제 변경/Adam credit도 확인해야 한다. Conditional과 모든 seed를 공개한다.
독립 검산은185 v2의 명시적 Slot archive loader를 사용한다.

실패하면 이 조건의 작은 LR/epoch sweep으로 연장하지 않고 종료한다.
GPU는 exclusive idle 장치만 사용하고 타 사용자가 오면 해당 장치를 양보한다.

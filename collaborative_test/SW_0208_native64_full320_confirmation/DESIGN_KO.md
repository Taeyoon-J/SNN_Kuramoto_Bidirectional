# SW0208: 204와 warm 모델의 전체320 확대 평가

204의 seed0 고정16 FG-ARI .900539와 동일16 warm192 .924401을
전체320으로 확대한다. 추가 학습은 하지 않는다. 두 모델은 독립 seed가
아니며, 본 실험만으로 세 seed 평균 .90 또는 70,000장 신규 학습 성공을
주장하지 않는다. supervised 모델이며 unsupervised 기록에 넣지 않는다.

- 입력: canonical HDF5 전체128 RGB, IDs1320..1639, crop 없음.
- 출력: native64 4096 patch, 기존1024-step spike readout/settle512 그대로.
- 모델: 실제204 완료 checkpoint와 실제192 TRAIN8 완료 checkpoint.
- RNG: 두 arm 모두 이미지 offset별193000+offset. 기존193/204 mini와
  같은 구간은 같은 RNG이며, 나머지288장의 RNG는 이번에 미리 확장 등록한다.
- 분할: offset modulo4의 네 작업. 각 작업은 두 arm을 실행한다. 모든
  GPU0..3에 exclusive lease로 대기하며 외부 사용자가 들어오면 양보한다.
- GT: 예측 전체640개 검증 후 mask를 읽는다. 2px modal, 동률 작은ID.
  기존 comparable Slot 세 seed의 전체320 native128 예측을 modal64로 평가.
- 모델/입력/소스/각 행 hash를 기록한다. 기존 mini와 겹치는 모든 행의
  label map은 정확히 같아야 한다. 점수는 독립 구현과 일치해야 한다.
- 점수 세 개와 paired bootstrap95% CI(10000회, seed208)를 공개한다.
  확대 학습 우선 후보는 전체320 FG-ARI가 높은 arm이다. 선택에 쓴
  validation320 결과는 독립 test 성능이라고 부르지 않는다.
- 70,000장 확대 학습은 이 평가의 후속 단계이며 현재 실행 내용과
  혼동하지 않는다. 전체320 seed0 .90도 세 seed 목표 달성은 아니다.

이 실험은 mini 개선 gate를 우회해 개선을 주장하는 실험이 아니다.
사용자가 요청한 절대 .90 결과의 범위를 확인하는 별도 confirmation이다.

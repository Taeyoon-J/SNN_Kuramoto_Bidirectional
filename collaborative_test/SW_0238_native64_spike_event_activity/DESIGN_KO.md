# SW0238: 현재 native64 spike event 활동 진단

235번에서 검증한 실제 SW0232 누적512회 모델을 그대로 사용한다. 각 seed의
기존 TRAIN 순서 첫2장만 T1024로 추론해 마지막512 step의 actual component
spikes/gates를 관찰한다. 진단의 입력은 RGB와 실제 trace이며 라벨·성능 평가는
사용하지 않는다. 기존 checkpoint 복원에서 과거 TRAIN artifact를 검증할 수 있지만
이 진단의 정답 입력으로 쓰지 않는다. 추가 학습·mask 변경은 없다.
과거 native32 all-on event 관측을 현재 native64에 그대로 적용하지 않는다.

S=event*gate에서 gate>0일 때만 event occupancy와 transition을 관측한다.
gate=0에서는 event를 추정하지 않는다. 실제 S와 gate의 bitwise equality 및
component-shared gate 여부를 기록한다. 이6개 TRAIN trace의 관찰은 성능 결과나
counterfactual membrane/gate 제거의 인과 증명이 아니다. 후속 loss 선택의 진단이다.

독립 actual checkpoint CPU admission 뒤 기존 독점 GPU 큐를 사용한다. 외부 사용자가
접근하면 우리 worker만 양보하고 완료한 이미지부터 재개한다. 기존237번은 변경하지
않으며70,000장 실험 범위 확인에 의존하는 학습/추론도 시작하지 않는다.

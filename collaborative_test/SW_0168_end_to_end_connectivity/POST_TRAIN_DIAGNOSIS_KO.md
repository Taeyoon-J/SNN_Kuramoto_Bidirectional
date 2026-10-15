# SW0168 학습 후 physical gating 진단

2026-10-15. Sol6.1 설계 검토와 root의 실제 완료 artifact 수집에 근거한다. 고정320 평가 결과와 분리한 TRAIN-only 진단이다.

고정 TRAIN8에서 두 arm 모두 학습 전후 settled event-on fraction1, event-transition fraction0이었다. 이 기록에서는 `S=E*g`가 `S=g`로 줄어든다. Membrane/integration parameter의64회 update와 nonzero surrogate credit은 확인됐지만, **실제 발화 선택이 분할에 기여했다는 증거는 아니다.** 전체512장·70k·VAL에 대한 포괄적 진술이나 FG-ARI 성능 예측으로 확장하지 않는다. `dendric_layer` raw norm은 진단 이름 필터 누락으로 미측정이며 optimizer update 기록과 구분한다.

이는 SW168의 encoder·graph·native-core joint training이 실행되지 않았다는 뜻이 아니다. 두 최종 모델의64회 history/optimizer/checkpoint/boundary 검증은 통과했고 두 TRAIN loss도 감소했다. 다만 'gradient가 흐른다'와 '해당 physical forward가 물체 구분에 필요하다'는 별도 검증 대상이다.

평가 후 조건부로 검토할 단일 다음 가설은 **source TRAIN component RMS로 정규화한 입력, 느린 membrane 적분, binary hard reset**이다. 또 작은 threshold 조정을 반복하는 대신 current 단위와 실제 charge history/reset을 바꾸는 방향이다. 초기 시간상수는16 step, 정규화된 threshold 시작값은.5를 검토하며 rate target·밀도 맞추기·threshold sweep은 넣지 않는다. 이는 아직 prospective 아이디어이며 SW168과 source-output parity를 주장하거나 실행을 등록한 것이 아니다.

같은 forward/capacity에서 recurrent-history live/detached 학습을 비교하면 credit의 역할을 분리할 수 있다. 실제 forward 필요성을 주장하려면 학습된 모델에서 history-off/reset-off 개입을 별도로 확인해야 한다. 먼저 소규모 TRAIN에서 finite mixed events/transition과 실제 current/history credit이 생기는지 확인하고, untrained FG가 떨어진다는 이유만으로 곧바로 기각하거나 activity를 맞추는 sweep을 하지 않는다. 이 방향은 과거 DC-history 제거 또는 threshold-only 변경과 구분한다.

SW168의 conditional/standalone 평가를 먼저 마친 뒤 recipe 유지/종료와 다음 실험을 결정한다. 이 문서는 .87 평균, 모든 gating 기여, 70k scaling 또는 Slot Attention 승리를 보장하지 않는다.

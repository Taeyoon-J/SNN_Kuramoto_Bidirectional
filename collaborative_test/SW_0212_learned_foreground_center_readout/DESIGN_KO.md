# SW0212: 학습된 전경과 비국소 center assignment 결합

211 결과를 아직 읽기 전에 등록하는 후속 readout 비교다. 211의 원래 spike
전경 보존 조건을 바꾸지 않는다. 별도212에서만 이미 학습된 foreground
출력을 사용한다. Core/encoder/graph/gates와211 head 모두 재학습하지 않는다.

207의 전경 제거는 공간 연결을 끊어 object split을 증가시켰다. 211은
offset으로 떨어진 patch도 같은 center에 배정한다. 따라서 이 비국소 배정과
학습 전경을 결합하면 foreground 오류를 고치면서 split을 피할 수 있는지
확인한다. 결과를 미리 성공으로 가정하지 않는다.

- 211 primary128 head 고정. 32/64 snapshot 중 선택하지 않는다.
- 동일16 DEV/seed2/native64, 같은268-channel cache만 사용. RGB나 GT count
  입력 없음. 211 학습 mask32장 외에 추가 학습/GT supervision 없음.
- foreground logit>=0 (sigmoid>=.5)로 고정한다. DEV threshold sweep 없음.
- heat/offset/center threshold .3, radius1, capacity10은211과 동일하다.
- CPU에서 frozen head를 실행한다. 모든16장의 원래 foreground를 사용한
  replay가 저장된211 primary128 mask와 bitwise 동일해야 비교를 허용한다.
  수치 차이로 replay가 다르면 점수를 내지 않고 관찰 결과를 보존한다.
- 새 foreground를 사용한 모든16 mask를 먼저 저장·검증하고 GT를 읽는다.
  원래 baseline, 211 primary128, 새212를 같은 evaluator로 비교한다.
- FG-ARI 우선/두 IoU/count 및 fallback을 모두 보고한다. CI는211과 같은
  paired16 bootstrap이다. 유용한 한 지표 개선도 보존하며, 전체320/세seed
  .90 또는 데이터 scaling/gating 인과 증거로 해석하지 않는다.
- 211 supervisor가 실제 살아 있는지 확인하며 대기한다.211 실패/종료에
  결과가 없으면 재시작하지 않는다. GPU 작업을 추가하지 않는다.

이 실험은 frozen classifier readout 비교이며 새 training test가 아니다.

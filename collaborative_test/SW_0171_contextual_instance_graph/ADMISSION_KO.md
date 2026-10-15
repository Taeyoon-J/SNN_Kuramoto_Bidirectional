# SW0171 학습 진입 검증 — 2026-10-15

실제 Frontier GPU1에서 source-preserving contextual encoder 진입 검증을 통과했다. 최적화 업데이트는 0회이며 성능 향상이나 최종 평가 결과가 아니다.

- Original SourceT와 zero-residual context의 동일 TRAIN RGB/RNG B4에서 gamma, native normalized z, A, 실제 Q, six traces, own QCC와 SW165 solver energy/merge log/labels가 정확히 일치했다. B1 TRAIN parity도 통과했다. VAL RGB/GT는 읽지 않았다.
- 원본 source 전체 상태와 base encoder의 전후 SHA가 일치했다.
- TRAIN positions0..15의 네 B4 평균 loss에 대해 실제 gamma gradient norm 합의 비율로 lambda=0.35786938902827736을 고정했다. Q와 instance loss 모두 양의 유한 gradient norm을 가졌다.
- 다음 학습의 고정 TRAIN8 진단은 positions256..263이며, 초기값은 첨부 independent_verification.json에 기록했다. 학습 성능 판정에 사용할 VAL 결과가 아니다.
- 로컬 및 서버 CPU 검사 26개 통과. 실제 원본 TRAIN512 캐시 소비 및 source admission도 통과했다.

첫 큐는 `python -m` 진입 시 `__name__`이 `__main__`이 되는 실행 경로를 놓쳐 child import 전에 종료됐다. 모델/CUDA/optimizer는 실행되지 않았다. 기존 실패 기록을 보존하고 canonical module 이름을 고정했으며 해당 진입 이름 회귀 검사를 추가했다. 수정 큐는 별도 v2 runtime에서 단 한 번 실행되어 정상 완료했다. preflight/model/loss는 수정하지 않았다.

원본 server 기록, terminal PID/보고서 SHA 연결 및 독립 수치 검증은 `results_archive/completed_admission_20261015/`에 있다. 다음 단계는 등록된 Q-only 대 Q+instance의 동일 8-update 비교이며, 아직 학습 결과는 없다. incumbent는 SW165 그대로다.

# SW0175 학습 후 자동 평가 연결

두 V2 학습이 모두64 updates를 완료하면, 독점 유휴 GPU를 확보해 각 모델의
고정320 예측을 자동 실행하도록 두 prediction supervisor를 등록했다.
별도 CPU scorer는 두 모델의 예측 파일과 protocol이 고정·검증된 뒤에만 GT를 읽는다.

로컬·서버 평가 검사 각10개와 실제 Source-T/SW164 reference·baseline score SHA
검증을 통과했다. 학습 중인 V2 source는 수정하지 않았다.
원본 foreground를 유지하는 conditional 출력과 같은 learned rollout의 QCC foreground를
사용하는 standalone 출력을 모두 제출한다. 추론은 실제 학습된 graph gain을 사용한다.
checkpoint state 외의 mean/std/clip·patcher·frozen geodesic 구성도 학습 manifest와 비교한다.

세 개의 실제 coordinator PID와 선행 작업 대기 상태를 확인했다. 이는 큐 등록 증거이며
평가 점수가 아니다. seed1 supervised pilot의 고정320 점수와 paired bootstrap이
완료된 뒤에만 다음 확대 여부를 판단한다. SW165 최고 기록은 유지한다.

증거: `results_archive/evaluation_queue_started_20261015/`.

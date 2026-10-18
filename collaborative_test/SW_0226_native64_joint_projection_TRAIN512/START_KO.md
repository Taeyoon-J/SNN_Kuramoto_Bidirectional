# SW0226 시작: 공동학습 TRAIN512 비교

로컬 10개·Linux 10개 테스트와 3개 CLI 검사를 통과했다. 세 seed 모두 실제 CPU
admission 및 projection gradient 검사를 통과했다. 원래 source teacher의 초기
within-object variance와의 최대 차이는 4.37e-10이고, TRAIN64 prefix의 RGB/GT/
teacher와 원래 223번 checkpoint/protocol/mask를 검증했다. 세 TRAIN512 큐가 등록됐고
관측 당시 seed1 worker가 GPU3에 배정됐으며 나머지 supervisor도 살아 있었다.
이 기록은 모델 성능 결과나 optimizer 완료 기록이 아니다.

encoder 전체와 실제 core 입력 projection을 같은 128회 업데이트로 학습하며,
고유 이미지 수만 223번의 64장 대신 512장으로 늘린다. source, loss, LR, batch16,
readout, DEV16 및 세 seed를 유지한다. 총2048 image exposures로 예산을 맞춘다.
새 TRAIN512의 최종 mask를 모두 동결한 뒤 기존 TRAIN64와 비교한다. 70,000장 학습이나
full320 확인은 이번 미니에 포함하지 않으며 최고 full320 기록도 갱신하지 않는다.

225번은 별도로 계속 실행 중이다. Linux 큐/worker에는 큰 배열 쓰기의 THP 지연을
피하도록 프로세스별 메모리 정책을 적용한다. 전역 설정과 모델 계산식은 바꾸지 않는다.
빈 GPU0..3만 독점 lease로 사용하며 다른 compute 작업이 나타나면 본인 worker만 양보한다.

## 후속 실제 GPU 학습 확인

server_unix1792363053에서 seed1 worker PID2665968가 GPU3에서 실제 encoder 및
core projection 128 Adam updates를 완료한 training_report를 확인했다. 다른 core
업데이트는 0이고, 이후 원래 spike readout 평가를 실행 중이다. 이는 최종 평가
점수가 아니며 나머지 두 seed는 빈 GPU를 기다리는 큐에 있다. 근거는
results_archive/deployment_v1/gpu_training.json이다.

# SW0179: QCE의 seed 0·2 재현성 검증

완료·독립 검산된 standalone 세 seed 평균은 **FG-ARI 0.761620 /
Foreground IoU 0.469308 / Object IoU 0.584656**입니다. seed 0·1의 FG는
높았지만 seed 2가 0.616842로 낮아져 등록한 primary gate에 실패했습니다.
기존 최고 unsupervised SW0165를 유지합니다.

전체 비교·신뢰구간·한계는 [RESULT_KO.md](RESULT_KO.md)에 있습니다.
이하 학습·큐 등록 근거에 기록된 실행 단계는 모두 완료됐고 재시작하지
않았습니다. 최종 점수·독립 검산·예측 protocol은 `results_archive/score_v1/`입니다.

SW0178 seed 1에서 standalone FG-ARI 0.8180 / Foreground IoU 0.4375 /
Object IoU 0.6864였던 product-QCE 설정을 변경 없이 반복합니다. 각 seed의
원래 SW0097 checkpoint에서 시작하고, TRAIN512 이미지와 정답 mask로
2회 추가 학습합니다. supervised 실험이며, 70,000장 신규 학습 결과나
unsupervised 성능으로 해석하지 않습니다. seed 1 결과는 기존의 검증된
SW0178 결과를 사용합니다.

실제 서버 CPU 검증에서 세 seed의 checkpoint가 서로 다름을 확인했고,
seed 1 초기 모델 상태는 SW0178과 정확히 일치했습니다. 각 seed의
zero-context gamma, 빈 optimizer, checkpoint 저장·복원도 확인했습니다.
학습·예측·점수 계산과 각 큐의 테스트 23개가 서버에서 통과했습니다.

seed 0·2의 예측 큐도 서버에 등록했고 실제 부모 프로세스가 학습 완료를
기다리는 것을 확인한 뒤, 학습 완료 후 빈 GPU에서 자동 예측을 마쳤습니다.
다른 사용자의 GPU 작업과 겹치면 양보하도록 구성했습니다. 예측 코드와
동일 seed의 실제 SourceT 기준 파일도 CPU에서 검증했습니다.

CPU 점수 계산 큐도 실제 프로세스로 등록했습니다. 새 seed 0·2와 기존
SW0178 seed 1의 예측이 모두 확정된 후에만 평가 정답을 읽습니다.
비교할 SW0165·Slot Attention의 세 seed 기준 파일과 평균도 실제 서버에서
검증했습니다. 최종 성능 점수와 별도 검산까지 완료됐습니다.

고정 32×32 patch·320장 평가에서 세 seed 평균을 SW0165와 같은 조건의
Slot Attention에 비교합니다. 예측을 모두 확정한 뒤 평가 정답을 읽습니다.
FG-ARI 향상이 특정 seed에만 나타나는지와 두 IoU의 tradeoff를 함께 보고합니다.
사전 기준과 한계는 [DESIGN_KO.md](DESIGN_KO.md), 실제 시작 근거는
`results_archive/start_v1/`, 평가 준비와 큐 등록 근거는
`results_archive/eval_v1/`에 기록했습니다.
학습 완료 검증과 각 최종 manifest는 `results_archive/train_v1/`에 있습니다.

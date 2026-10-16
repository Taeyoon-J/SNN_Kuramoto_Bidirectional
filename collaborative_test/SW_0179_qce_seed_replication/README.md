# SW0179: QCE의 seed 0·2 재현성 검증

seed 0·2 학습이 모두 완료됐습니다. 실제 최종 checkpoint와 atomic boundary의
모델 tensor 일치, 두 pass의 고유 512장 노출, Adam step 64를 독립 검증했습니다.
최종 평가 점수는 아직 없으며 예측 큐가 후속 작업을 진행합니다.

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
기다리는 것을 확인했습니다. 각 학습이 완료되면 빈 GPU에서 자동으로
예측하며 다른 사용자의 GPU 작업과 겹치면 양보합니다. 예측 코드와
동일 seed의 실제 SourceT 기준 파일도 CPU에서 검증했습니다.

CPU 점수 계산 큐도 실제 프로세스로 등록했습니다. 새 seed 0·2와 기존
SW0178 seed 1의 예측이 모두 확정된 후에만 평가 정답을 읽습니다.
비교할 SW0165·Slot Attention의 세 seed 기준 파일과 평균도 실제 서버에서
검증했습니다. 아직 새로운 성능 점수는 없습니다.

고정 32×32 patch·320장 평가에서 세 seed 평균을 SW0165와 같은 조건의
Slot Attention에 비교합니다. 예측을 모두 확정한 뒤 평가 정답을 읽습니다.
FG-ARI 향상이 특정 seed에만 나타나는지와 두 IoU의 tradeoff를 함께 보고합니다.
사전 기준과 한계는 [DESIGN_KO.md](DESIGN_KO.md), 실제 시작 근거는
`results_archive/start_v1/`, 평가 준비와 큐 등록 근거는
`results_archive/eval_v1/`에 기록했습니다.
학습 완료 검증과 각 최종 manifest는 `results_archive/train_v1/`에 있습니다.

# SW0179: QCE의 seed 0·2 재현성 검증

서버 학습 시작을 실제 프로세스와 GPU 점유로 확인했습니다. seed 0은 GPU 1,
seed 2는 GPU 2에서 실행 중입니다. 결과는 아직 없습니다.

SW0178 seed 1에서 standalone FG-ARI 0.8180 / Foreground IoU 0.4375 /
Object IoU 0.6864였던 product-QCE 설정을 변경 없이 반복합니다. 각 seed의
원래 SW0097 checkpoint에서 시작하고, TRAIN512 이미지와 정답 mask로
2회 추가 학습합니다. supervised 실험이며, 70,000장 신규 학습 결과나
unsupervised 성능으로 해석하지 않습니다. seed 1 결과는 기존의 검증된
SW0178 결과를 사용합니다.

실제 서버 CPU 검증에서 세 seed의 checkpoint가 서로 다름을 확인했고,
seed 1 초기 모델 상태는 SW0178과 정확히 일치했습니다. 각 seed의
zero-context gamma, 빈 optimizer, checkpoint 저장·복원도 확인했습니다.
trainer/queue 테스트 11개가 서버에서 통과했습니다.

고정 32×32 patch·320장 평가에서 세 seed 평균을 SW0165와 같은 조건의
Slot Attention에 비교합니다. 예측을 모두 확정한 뒤 평가 정답을 읽습니다.
FG-ARI 향상이 특정 seed에만 나타나는지와 두 IoU의 tradeoff를 함께 보고합니다.
사전 기준과 한계는 [DESIGN_KO.md](DESIGN_KO.md), 실제 시작 근거는
`results_archive/start_v1/`에 기록했습니다.

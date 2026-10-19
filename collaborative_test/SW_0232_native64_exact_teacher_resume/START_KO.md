# SW0232: 실제 accepted checkpoint에서 이어 학습

SW0223의 실제 128회 checkpoint에서 encoder·gamma projection·Adam·history·Torch RNG를 복원하고, 같은 TRAIN64와 teacher loss로 384회 추가해 누적 512회를 비교한다. 재계산으로 원래 128회를 재현하려던 SW0231과 구분한다. 동일성 기준은 완화하지 않는다.

새 학습 예산은 384 updates / 6,144 exposures이며, 상속된 128회를 포함하면 512 updates / 8,192 exposures다. graph·나머지 core/gating·spike/membrane readout은 그대로 유지한다. mask-supervised native64 미니 실험으로, 70,000장 신규 학습이 아니다.

세 seed를 고정 16장에서 baseline 및 실제 SW0223 128회 모델과 비교한다. FG 개선과 paired CI, 두 IoU를 함께 확인한다. 전체 320장이나 FG-ARI .90 달성으로 해석하지 않는다. 로컬 16개 검사는 통과했으며 서버 Linux 검사와 세 seed의 실제 accepted checkpoint CPU 복원·gradient 검사를 통과해야 GPU 큐가 등록된다. 실제 시작 단계는 START_EVIDENCE.json에 기록한다.

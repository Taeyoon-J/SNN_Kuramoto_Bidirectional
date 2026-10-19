# 239번 서버 검증 완료 및 단일 GPU 큐 등록

로컬 및 Linux 서버 검사 각각 10개, 서버 실행 명령 검사, seed0/1/2의 실제
checkpoint 복원과 encoder·projection gradient 검증을 통과했다. 각 seed에서
기존 source teacher의 실제 target cache와 새 streaming 계산 결과도 bitwise 일치했다.
원본 source와 teacher, input statistics 및 CPU 검증 단계의 optimizer 불변성을 확인했다.

등록 시점에 독립 검증된 전체 320장·세 seed 결과 중 최고인 227번을 시작점으로
고정했다. 그 점수는 .858916 / .605286 / .674313이다. 237번은 이 시점에 아직
세 seed 전체 평가가 완료되지 않아 초기 모델 선정에서 제외했다. 중간에 다른
checkpoint로 초기 모델을 바꾸지 않는다.

새 단계는 고유 TRAIN70,000장·10 complete passes·BATCH16으로 seed당
43,750개 새 Adam update를 수행한다. Adam은 새 데이터 단계에서 초기화한다.
mask-supervised teacher loss와 기존 readout을 유지하며 나머지 core 가중치는 고정한다.
현재까지의 source 학습 이력을 포함한 fine-tuning이며 scratch 학습이라고 부르지 않는다.

**대기 큐 등록을 확인했으며, 기록 시점에는 GPU worker와 새 optimizer update가 없다.**
한 supervisor가 한 GPU만 임대하고 seed0→1→2를 차례대로 처리한다.
다른 실험과 다른 GPU 사용자를 중단하지 않는다.

학습 후 같은 320장·같은 native64 평가 계약으로 세 점수와 seed 평균을 보고한다.
예시 이미지는 사전에 정한 1320/1321/1322이며 ours와 Slot 모두 seed0를 사용한다.
검증 집계는 QUEUE_VALIDATION_EVIDENCE.json에, 서버 경로·PID·파일 메타데이터를
포함한 원본 기록은 비공개 runtime 폴더에 보관했다.

# SW0240: 동일한 70,000장 학습의 입력 I/O 개선

239번은 실제 GPU gradient·native64 forward 및 첫 Adam checkpoint를 검증했다.
그러나 무작위 batch를 HDF5에서 읽을 때 이미지 축에 큰 chunk로 압축된 데이터를
반복 해제하는 CPU 병목이 확인됐다. 모델이나 loss를 바꾸는 실험이 아니다.

## 입력 캐시

cache.py는 기존 239번 data.py의 canonical RGB와 modal patch label 함수를 그대로
사용한다. source chunk에 맞춰 **TRAIN 구간 0..999와 1640..70639만** 순차적으로
읽는다. 예약 평가 구간의 mask는 캐시에 넣기 위해서도 읽지 않는다.

RGB는 uint8 그대로, patch label은 손실 없이 uint8로 저장한다. batch 반환 시
label을 기존 int64로 복원한다. 원래 TRAIN ID에서 cache 위치로만 매핑하며,
seed별 shuffle과 batch 구성 및 이미지·mask tensor는 바꾸지 않는다. 모든 block의
tensor를 저장 후 확인하고 파일 hash와 완전 사용 범위를 완료 manifest에 남긴다.
캐시는 생성 artifact이며 Git에 올리지 않는다.

로컬 검사 2개가 전체 TRAIN ID 완전 포함·평가 구간 배제·chunk 정렬·무작위 순서의
batch identity를 확인했다. 실제 전체 캐시 생성과 원본 batch 대비 bitwise 비교 및
읽기 시간 측정은 별도로 수행해야 한다.

## 이어서 학습할 때 필요한 조건

239번의 등록된 코드·selection·checkpoint를 몰래 바꾸거나 reader를 숨겨서
교체하지 않는다. 동일 학습의 명시적 I/O successor로 출처를 기록한다.
최고 확정 모델 227번에서 시작한 기존 239번 초기 모델·teacher·loss·learning rates·
seed 순서·70,000장 10 passes·총 43,750 logical updates·readout·평가 320장도 유지한다.

239번은 캐시와 successor가 검증될 때까지 실행을 유지한다. 전환할 때는 본인 239번
supervisor/worker만 중단하고 마지막 원자적 checkpoint를 동결한다. 그 model·Adam·
RNG·이미지 사용 이력을 실제로 복원해, 학습을 처음부터 다시 하지 않고 이어간다.
수행되지 않았던 seed1/2는 기존 선정 모델에서 같은 계획을 시작한다.
동시에 이 70,000장 작업에 GPU 두 개를 쓰지 않도록 이전 worker 종료를 확인한다.

새 binding은 원래 239번 admission 및 보존 checkpoint의 SHA를 명시하고,
inherited update와 successor update를 구분한다. 합친 이력으로 각 pass 70,000장
완전 사용과 최종 Adam step 43,750을 검증해야 한다. 실제 checkpoint 재개 동등성과
단일 GPU 큐·서버 preflight 전에는 successor 학습을 시작하지 않는다.

현재는 입력 캐시를 생성하는 단계이며, successor trainer·재개 검증·큐 등록은
아직 완료하지 않았다. 239번의 모델·loss·작업은 이 단계에서 변경하지 않았다.

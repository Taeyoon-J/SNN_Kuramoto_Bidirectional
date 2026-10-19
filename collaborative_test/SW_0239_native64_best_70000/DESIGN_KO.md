# SW0239: 최고 native64 모델의 실제 TRAIN70,000 확대 학습

사용자는 70,000장 전체 추론이 아니라, 최고 모델을 70,000장으로 학습한 뒤
분리된 고정 평가 데이터의 점수와 mask를 확인하도록 명시했다.

## 고정 범위

- canonical HDF5 clevr_10-full.hdf5의 TRAIN IDs는 0..999와 1640..70639,
  정확히 서로 다른 70,000장이다. 평가·검증 예약 구간 1000..1639는 제외한다.
- native64×64 patch, 기존 T=1024·settle=512의 spike 기반 instance readout을 유지한다.
- 같은 320장 IDs 1320..1639에서 FG-ARI / Foreground IoU / matched Object IoU를
  세 seed 각각 및 평균으로 보고한다. 이 구간은 재사용 validation이며 untouched test라고
  표현하지 않는다. mask 예시는 사전에 정한 IDs 1320, 1321, 1322를 사용한다.
- 한 GPU만 임대하고 seed0→1→2를 순서대로 진행한다. 다른 실험의 GPU는 점유하지 않는다.
  외부 사용자가 나타나면 본인 worker만 양보하고 checkpoint에서 이어간다.
- 전체 320장·세 seed·독립 검증이 완료된 후보 중 FG-ARI 평균이 가장 높은 모델을
  시작점으로 고정한다. 세 지표가 comparable Slot 평균을 넘는 조건도 확인한다.
  237번이 미완료라면 현재 확정 최고인 227번/223 checkpoint가 대상이다.
  시작점 선정 후 모델과 provenance를 고정하고 중간에 다른 후보로 교체하지 않는다.

## 확대하는 학습 방식

현재 최고 모델의 recipe인 encoder 전체와 실제 8→4 sensory projection 학습을
유지한다. 그 외 core 가중치, graph/readout 규칙과 loss 계수는 바꾸지 않는다.
mask-supervised source object-frequency teacher와 원래 gamma anchor를 유지하며,
새 데이터 단계의 Adam은 기존 learning rates 3e-5 / 3e-4 / 3e-4로 초기화한다.
따라서 이전 stage의 Adam을 정확히 이어간 실험이라고 부르지 않는다.

BATCH=16, 10 complete passes, seed당 43,750개의 새 optimizer update와
700,000회 이미지 노출을 등록한다. 첫 pass에서도 70,000장 각각을 정확히 한 번
사용한다. 전체 RGB/teacher 배열을 메모리에 쌓지 않고 batch 단위로 읽는다.
teacher는 원래 source 모델에서 고정하며 student를 teacher로 덮어쓰지 않는다.
학습 mask와 평가 mask의 접근 경로를 구분한다.

기존 source teacher와 동일하게 teacher를 CPU에서 계산한다. 실제 source의
기존 target cache와 bitwise 일치 및 student loss·gradient를 preflight에서
확인해야 한다. 검증되지 않은 teacher 구현으로 GPU 학습을 시작하지 않는다.
학습 상태·Adam·RNG·이력의 checkpoint 재개와
전용 단일 GPU 큐도 서버 실행 전에 검증한다.

## 초기 데이터 경로 검증 기록

streaming 데이터 경로의 로컬 검사 5개를 통과했다. 기존 `clevr_mask_patch`와
합성 mask의 label 계산도 bitwise 일치했다. 서버의 실제 HDF5에서 첫 seed0 batch와
마지막 seed2 batch를 읽어 RGB·mask geometry, 순서 및 기존 patch label 구현과의
bitwise 일치를 확인했다. 근거는 LOCAL_DATA_PREFLIGHT.json과 SERVER_DATA_PREFLIGHT.json이다.
이는 전체 70,000장의 데이터 내용 검사나 실제 모델·optimizer의 CPU admission을
대체하지 않는다. 위 데이터 검증 기록은 학습 시작 증거가 아니다.
실제 CPU admission·큐 등록·학습 시작·완료는 별도 실행 기록으로 확인한다.

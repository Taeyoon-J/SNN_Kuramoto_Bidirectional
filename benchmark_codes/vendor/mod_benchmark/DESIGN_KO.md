# SW0250: SW0240의 두 데이터셋 전이 benchmark

사용자 승인: 기존 240번의 세 독립 seed에서 추가 학습한다. benchmark/와 benchmark.md는 f9301ed 원문을 그대로 고정하며 코드·데이터 adapter는 이 폴더에만 작성한다. 보호 파일 11개의 SHA는 BENCHMARK_LOCK.json으로 실행 전후 확인한다.

데이터는 /work/USERS/tkim1/mod의 올바른 multi_dsprites(colored_on_colored, HWE) 및 tetrominoes(EHW) 계열이다. mdsprites 계열은 사용하지 않는다. 원래 manifest의 TRAIN 0..5999, validation 6000..6999, TEST 7000..7999를 유지한다. 별도 다운로드나 무작위 재분할은 하지 않는다.

각 데이터셋에 대해 zero_shot / finetune × seed0/1/2, 총 12개 작업이다. 기존 portable SW0240의 SW0249_matched35000_terminal에서 시작한다. zero_shot은 학습 없이 기존 전체 checkpoint를 사용한다. finetune은 TRAIN 6,000장×10회(60,000노출, batch16, 3,750 actual Adam updates), fresh Adam, LR 3e-5/3e-4/3e-4, gradient clip1로 encoder 전체와 실제 sensory projection만 추가 학습한다. 나머지 graph/core/integration/readout과 feature 통계는 고정한다. 기존 frozen teacher의 frequency denoising + gamma anchor mask-supervised objective를 유지한다. Teacher encoder/frequency 연산은 student와 같은 GPU에서 no_grad로 수행하여 CPU 병목을 줄인다. 새 encoder나 precomputed gamma로 checkpoint의 encoder를 교체하지 않는다.

고정 10회 종료 checkpoint만 평가하며 validation 선택·early stopping·loss 탐색은 하지 않는다. TEST mask는 학습에 사용하지 않는다. 각 1,000장 TEST prediction protocol을 완성한 다음 scoring에서 pixel truth를 읽는다. 원본 preflight.py는 실제 평가를 시작할 때마다 변경 없이 실행한다.

입력은 원본 benchmark/make128.py로 nearest 128×128 RGB를 준비한다. Native model은 두 데이터셋 모두 실제 64×64 SW0240을 유지한다. Multi-dSprites는 native64 예측을 그대로 평가한다. Tetrominoes는 원본 35픽셀에서 빈 cell이 생기는 grid64 GT를 만들지 않는다. 학습 시 기존 grid32 TRAIN partition만 2×2 반복하여 모델의 64×64 frequency에 대응시킨다. 평가 시 native64 예측을 원본 benchmark/make_targets_grid.py의 modal 규칙으로 32×32로 변환하고, 그 grid32를 같은 원본 함수의 35픽셀 cell edge로 확장한다. 이 고정 readout adapter를 zero_shot과 finetune에 동일 적용한다. Native32로 core를 재구성하거나 node parameter를 변환한 별도 모델 실험은 하지 않는다.

Pixel FG-ARI / foreground IoU / Hungarian-matched object IoU는 보호된 benchmark/evaluation.py의 원본 함수를 직접 호출한다. 추가로 pixel MBO(일대일 Hungarian matching과 구분), pred_fg, true_fg, native output group 수를 보고할 수 있다. 실제 TEST split의 oracle grid ceiling을 원본 modal/expand/evaluation 함수로 측정한다. 원래 validation300에서 측정한 0.9409/1.0000과 TEST ceiling을 혼동하지 않는다. 세 seed 평균과 sample standard deviation을 보고한다. 추가 학습 없는 모델도 source CLEVR 학습이 mask-supervised였다는 사실을 명시한다. 공개 논문 baseline 재현 없이 이 결과만으로 논문 모델을 이겼다고 주장하지 않는다.

실행 전 Linux 단위 검사·실제 두 데이터셋 CPU gradient 검사·TRAIN image0의 실제 CUDA forward/readout smoke를 통과해야 한다. 작업별 immutable binding, resumable optimizer/encoder/projection boundary, row별 prediction SHA, terminal model SHA와 dataset/file SHA를 보관한다. 원본 데이터·원본 checkpoint·다른 사용자의 프로세스는 수정하지 않는다. 여유 메모리 8GiB 이상의 GPU에서 공유 실행하며 외부 사용자가 나타났다는 이유로 실행 중인 작업을 자동 중단하지 않는다. 완료 후 구조/loss 탐색을 이어가지 않는다.

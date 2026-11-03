# 261 encoder·core·head 학습 — 252와 유사한 두 역전파 경로

이 문서는 **학습 가능한 실행 코드**를 설명한다. 학습 완료나 FG-ARI 향상 결과를
의미하지 않는다. 기존261 변환·추론 코드와 기존 checkpoint는 보존한다.

## 정확한 학습 경로

```text
경로 A: 이미지 → encoder → gamma64 → 실제64 core → phase/spike loss
                   ↑──────── core를 통과하는 역전파 ─────────┘

경로 B: 같은 core 활동·gamma → detach → context head → soft mask/RGB decoder
                                               ↑──────── RGB 복원 loss ──┘
```

- 기본 실행은 **encoder와 core 모두 업데이트**한다. 단순 head-only trainer가 아니다.
- Body loss는 head에 연결되지 않는다. Head loss는 encoder/core에 전달되지 않는다.
  활동 특징뿐 아니라 head로 들어가는 **gamma 경로도 detach**한다.
- Encoder pooling은 원래 CUDA adaptive-average-pooling forward를 그대로 사용한다.
  CUDA 기본 backward의 비결정적 atomic 누적 대신, 동일 평균 영역의 정확한 선형
  Jacobian을 행렬곱으로 적용한다. Forward·pooling 범위·loss를 바꾸지 않는다.
- Encoder를 학습하려면 core의 그래프를 통과해야 한다. 기본1024step forward 중
  마지막64step에 gradient를 유지하는 truncated BPTT다. Graph/drive 준비도 live다.
- Signed partition·후보 생성·최종argmax는 미분 불가능하며 no-grad다. 학습에는
  head의 soft assignment를 사용한다. Body의 별도 동기화 목적함수가 실제 core를 학습한다.

### 252와 같은 것과 다른 것

| 항목 | 252 전체 학습 | 이번261 두 경로 학습 |
|---|---|---|
| Body loss → encoder/core | 연결 | 연결 |
| 최종 head loss → body | detach로 차단 | 활동·gamma 모두 detach로 차단 |
| Body 목적함수 | phase/spike + mask 기반 주파수 teacher·gamma anchor | **phase/spike만 사용** |
| Head 목적함수 | 정답 mask CE+Dice | **원본 RGB reconstruction MSE** |
| TRAIN 정답 mask | 사용 | **읽지 않음** |

261은 unsupervised를 유지한다. 252의 지도학습 teacher/CE/Dice를 그대로 복사하지 않는다.
Core의 phase/spike 목적함수는 기존252/253 full-benchmark와 같은 정의·계수다:

`L_body = L_phase + 5 * L_positive_component_spike_Q`

두 항 모두 기존 `UnsupervisedS2NetLoss`에서 bimodality6, balance10,
coherence0.5, collapse1, target_density0.867, grid64를 사용한다. Phase PLV는
component 위상을 평균하는 기존 `combine='mean'`을 유지한다.

`L_head = mean((RGB_reconstructed - RGB_target_patch_mean)^2)`

Head는 공개260의 `context_rgb` 구조를 실제4096개 위치의 활동에 연결한다.
고정 content encoder와 prior/residual scale은 기존 설계대로 고정하며,
head embedding·context CNN·decoder를 학습한다. 배경 지정/삭제를 추가하지 않는다.

## 환경과 입력

저장소 **전체와 공개 checkpoint들**이 필요하다. 기존 requirements를 사용한다.
각 seed는 새 Python 프로세스에서 실행한다. 입력 HDF5의 `image` 필드는
`uint8 [N,128,128,3]`이어야 한다. 학습기는 `mask` 필드를 열지 않는다.
실제 데이터 경로 예:

`/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5`

기본 TRAIN pool은0–999 및1640–70639(실제 데이터 길이 이내)다.1000–1639는
학습에서 제외한다. `--train-count`는 **고유 이미지 수**, `--epochs`는 반복 횟수다.
동일한 이미지 집합을 seed 간 공유하고 epoch 순서는 seed별로 섞는다.

## 1. 기존 가중치에서 전체 미세조정 — 기본 실행

저장소 루트에서:

```bash
pip install -r benchmark_codes/requirements.txt
CUDA_VISIBLE_DEVICES=0 python -B benchmark_codes/native64_core/train.py \
  --data /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output /work/USERS/$USER/SW0261_joint_seed0 \
  --seed 0 --init pretrained --train-count 128 --epochs 10 \
  --steps 1024 --settle 512 --tail 64 --device cuda
```

기본 Adam: encoder3e-6, core3e-5, head1e-3. Body/head optimizer가 별도이며
각 gradient norm을1로 clipping한다. Physical batch는1이다.
정확한 geodesic 연산의 row chunk는 메모리를 줄이기 위해 기본8이다.
`--geodesic-chunk 32`로 기존 실행 chunk를 사용할 수도 있다. 위치 수·거리 공식은 같다.
Decoder는 기본16개 그룹씩 activation checkpointing한다(`--decoder-chunk`).
RGB 복원식과 loss는 같고, 그룹별 합산 순서의 부동소수점 차이는 있을 수 있다.
예를 들어10,000장×10회는 `--train-count 10000 --epochs 10`이다.
Core/encoder가 변하므로 이전 spike cache를 재사용하지 않고 매 이미지 재계산한다.

## 2. 무작위 초기화에서 전체 학습

```bash
CUDA_VISIBLE_DEVICES=0 python -B benchmark_codes/native64_core/train.py \
  --data /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output /work/USERS/$USER/SW0261_random_seed0 \
  --seed 0 --init random --train-count 128 --epochs 10 --device cuda
```

모델은 공개 bundle의 구조 정의를 사용하지만 encoder·native64 core·integration·
head·decoder 가중치를 새로 만든다. Core는 독립적인64-grid tensor로 초기화하고,
shared dendritic bias는 기존 full-benchmark처럼0으로 시작한다. 정규화 통계는
해당 초기 encoder의 선택한 TRAIN 특징값 전체에서 scalar mean/std로 다시 계산한다.
통계·고정 content 표현은 이후 고정한다. 별도 mask teacher는 만들지 않는다.

## 3. 중단 후 이어 학습

```bash
CUDA_VISIBLE_DEVICES=0 python -B benchmark_codes/native64_core/train.py \
  --data /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output /work/USERS/$USER/SW0261_joint_seed0 \
  --seed 0 --init pretrained --train-count 128 --epochs 10 --resume --device cuda
```

원래와 같은 초기 checkpoint·split·학습률·steps/tail을 사용해야 한다.
Epoch 수는 목표 총 횟수다. 두 Adam 상태·epoch 내부 cursor·RNG를 복구한다.
기본 첫update/8update마다/epoch 끝에 atomic checkpoint를 저장한다.
`--save-every 1`이면 매update 저장한다. 저장 간격 사이의 미저장 작업은 재실행된다.
서로 다른 실행이 같은 output을 공유하면 안 된다.

## 4. 일부 모듈 고정

- `--freeze-core`: core 가중치는 고정하지만, body loss가 core를 통과해 encoder를 학습.
- `--freeze-encoder`: encoder 고정, core와 head 학습.
- 두 옵션 모두: body 학습을 끄고 head만 학습. Encoder/core 상태를 바꾸지 않는다.

## 5. 출력과 학습 checkpoint 추론

`initial.pt`, `resume.pt`, `binding.json`, `training.jsonl`, `progress.json`,
`complete.json`을 저장한다. 학습 체크포인트에는 실제64 core·encoder·head,
정규화 통계·두 optimizer·복구 상태가 들어간다. 원래 공개 checkpoint를 덮어쓰지 않는다.
`resume.pt`는 마지막 학습 상태이며, validation-best나 성능 향상 모델로 주장하지 않는다.

```bash
python -B benchmark_codes/native64_core/portable.py \
  --checkpoint /work/USERS/$USER/SW0261_joint_seed0/resume.pt \
  --image example128.png --output trained_labels64.npy --device cuda
```

새 checkpoint도 architecture 재구성을 위해 저장소에 포함된 동일 seed의 원래 공개
260/259 checkpoint가 필요하다. 실제 학습 가중치는 `resume.pt`에서 strict-load한다.
Large native64 tensors·Adam 상태로 학습 checkpoint는 공개 head checkpoint보다 커진다.

## 검증 범위와 한계

`training.py`는 실제 backward마다 body→head gradient가 없고,
head→encoder/core gradient가 없는지 검사한다. 기본 joint 모드에서는 body의
encoder·core gradient, head gradient가 finite/nonzero인지 검사한다.
이는 경로 검증이며 FG-ARI 향상이나 모든 gating의 인과적 기여를 입증하지 않는다.

기본 full1024-step/64-tail과 짧은 diagnostic 설정은 구분해서 기록한다.
짧은 검증 예시는 `--steps 64 --settle 32 --tail 4 --train-count 1 --epochs 1`이다.
짧은 run의 loss/속도를 기본 학습 성능으로 인용하지 않는다.
Graph/Q4096×4096 및 checkpointed geodesic 역전파는 메모리·시간 비용이 크다.
효과적인 loss 가중치·학습률·학습량은 별도 실험이 필요하다.

### 이번 코드의 실제 서버 검증

[validation/two_path_training.json](validation/two_path_training.json)에 근거를 보존했다.

- TITAN RTX, seed0, 실제4096개 위치에서 기본1024step/settle512/tail64로1update.
  Encoder·core·head 모두 finite/nonzero gradient를 확인했다.
- 별도64step/tail4 diagnostic에서 기존 가중치2update(중간 재개), random 초기화1update.
  세 모듈의 실제 가중치 변경, 두 Adam의 실제 step과 재개, 새 checkpoint loader를 확인했다.
- Pooling forward는 원본과 bitwise 동일하고 backward는 CPU 기본 연산과 수치 동등하다.
  그룹 분할 decoder의 loss·gradient도 원래 context_rgb와 수치 동등함을 확인했다.
- 이는 실행·경로 검증이다. 전체 데이터 학습, seed3개 성능 평가, FG-ARI 향상은 수행하지 않았다.

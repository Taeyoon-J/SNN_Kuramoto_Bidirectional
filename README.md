# patch_final — SW0240

240번의 실제 실행 구조를 메인 패키지와 함께 모은 독립 폴더입니다.
`patch_v2_sw`나 `collaborative_test`를 Python 경로에 추가하지 않아도 실행합니다.
원본 연구 저장소는 변경하지 않았습니다.

```text
patch_final/
  snn_kuramoto_bidirectional/   기존 core 계산 모듈
    sw240/                    240번 encoder·적분·rollout·readout·loss·loader
  predict.py                  학습된 240번으로 mask 생성
  train.py                    기존 teacher/loss로 추가 학습
  tools/export_from_research.py  기존 연구 checkpoint를 독립 형식으로 1회 변환
  config/                     실제 checkpoint에서 내보낸 설정
  checkpoints/sw240_seed0.pt  실제 240번 seed0 최종 가중치·통계·teacher
  validation/                 실제 checkpoint 복원·기존 mask 일치 검증
  tests/                      독립 실행·gradient·readout 검증
  SOURCE_PROVENANCE.json       복사한 코드의 원본과 SHA
```

## 기본 모델

```python
from snn_kuramoto_bidirectional import load_model

model = load_model("checkpoints/sw240_seed0.pt", device="cuda:0")
labels = model.predict(rgb_uint8)  # 입력 [B,128,128,3], 출력 [B,64,64]
```

240번의 최종 비교 종료 checkpoint를 사용합니다. 이 endpoint는 기존 240번의
70,000장 학습 이후, 최종 네 모델 비교에서 35,000장×10회를 추가 학습한 버전입니다.
세 seed 평균은 **0.889187 / 0.754775 / 0.686438**이며, 이 숫자는 seed0 하나의
점수가 아닙니다. 실제 가중치를 불러와야 그 학습 모델이 됩니다.
`SW240Model()` 생성만으로는 학습된 모델이 되지 않으며, checkpoint 없이
`predict`를 실행하면 오류로 막습니다.

구조는 RGB → 학습된 원본 CNN + context residual encoder → 고정 학습 통계로
정규화·clip → 64×64 gamma → 이미지별 graph와 sensory projection → Kuramoto
→ sinusoidal gate → phase 기반 dendrite·membrane 적분 → spike → foreground
support → signed connected partition → object mask입니다.

- 8 feature channels, 4 oscillator components, 4,096 patches, graph top-k 512.
- 고정 contrast 8의 squared geodesic distance와 원래 연산 순서를 보존했습니다.
- 1,024-step 실제 spike rollout, 첫 512 step 제외 후 readout.
- 각 이미지는 자신의 graph를 만듭니다. Graph 파라미터는 이 학습 단계에서 고정합니다.
- 이 모델의 spike→phase pulse / graph feedback은 꺼져 있습니다.
- 240번은 **mask-supervised**입니다. 임의의 unsupervised loss로 대체하지 않았습니다.

## 실행

```bash
cd patch_final
python -m pip install -r requirements.txt
python predict.py --checkpoint checkpoints/sw240_seed0.pt --input images.npy --output outputs/masks.npz --device cuda:0
```

입력은 uint8 RGB `[B,128,128,3]`의 `.npy` 또는 128×128 이미지 한 장입니다.
자동 crop/resize를 하지 않습니다. 출력 label 0은 배경이고 양수는 object ID입니다.
고정 평가 이미지 ID 1320+i를 재현할 때 inference seed는 `193000+i`입니다.
같은 GPU·PyTorch·CUDA 조건에서의 전체 mask 일치는 별도 검증 결과를 확인하세요.

현재 로컬 검증은 6개를 통과했습니다. 독립 import, native64 encoder/projection
gradient, 고정 core 유지, 실제 phase/dendrite/membrane rollout, foreground 보존,
전체 모델 형상의 checkpoint 저장·복원과 SHA 오류 거부를 확인했습니다.
로컬 단위 검증의 checkpoint roundtrip에는 합성 상태를 사용했습니다.
별도로 실제 240번 seed0 최종 checkpoint를 내보내 로컬에 저장했고,
전체 모델 SHA와 teacher 복원 SHA, 실제 loss·gradient의 유한성도 확인했습니다.
독립 폴더만 import한 서버 실행에서 미리 정한 평가 이미지 1320/1321/1322의
64×64 mask가 기존 최종 결과와 모두 정확히 일치했습니다(다른 patch 0개).
이는 3장 재현 검증이며 전체 320장이나 seed1/2를 새로 평가한 결과는 아닙니다.
상세 검증은 `validation/actual_seed0_replay.json`과 `actual_cpu_restore.json`에 있습니다.

현재 `checkpoints/sw240_seed0.pt`가 포함되어 있습니다(약 202MB).
`checkpoints/manifest.json`에 파일 SHA와 모델 SHA를 기록했습니다.
이 가중치는 로컬 폴더에 보관하며 GitHub `patch_final` 브랜치에는 코드와
검증 기록·checkpoint 명세를 올립니다. GitHub에서 코드를 받은 경우 가중치는
아래 exporter로 원본 연구 서버에서 내보내거나 기존 파일을 별도로 복사해야 합니다.
다른 seed가 필요하거나 가중치를 다시 내보낼 때는 원본 저장소가 있는
서버 환경에서 한 번 내보냅니다. 이 도구만 기존 연구 코드와 학습 산출물을 사용하며,
내보내기가 끝난 뒤에는 실행에 원본 저장소가 필요 없습니다.

```bash
python tools/export_from_research.py --legacy-root /path/to/patch_v2_sw --output checkpoints --seed 0
```

`--seed 1` 또는 `--seed 2`로 해당 seed를 내보낼 수 있습니다. 이미 존재하는 파일은
덮어쓰지 않습니다. `config/sw240_architecture.json`은 구조 template이고,
실제 가중치 로딩에는 bundle 내부의 checkpoint별 설정을 사용합니다.

내보낸 파일은 전체 core·encoder·적분 파라미터·feature 통계·실행 설정·원래 teacher를
포함합니다. 최종 checkpoint의 SHA와 복원된 전체 모델 SHA를 검증합니다.
기존 `boundary.ptx`는 일부 encoder/projection만 저장한 파일이므로 단독으로
이 폴더에 복사하는 것으로는 충분하지 않습니다.

## 추가 학습

```bash
python train.py --checkpoint checkpoints/sw240_seed0.pt --dataset /path/to/clevr_10-full.hdf5 --output outputs/continuation --train-count 35000 --epochs 10 --device cuda:0
```

기존 frozen teacher의 class별 effective-frequency center와 gamma anchor를 사용합니다.
Whole encoder와 실제 sensory projection만 학습합니다. Fresh Adam, batch 16,
LR 3e-5 / 3e-4 / 3e-4, gradient clip 1을 유지합니다. 35,000장 선택과 순서,
마지막 8장 batch, 평가 영역 1000–1639 제외도 최종 비교와 같습니다.
입력 checkpoint부터 **새 추가 학습**을 수행하는 명령이며 과거 전체 학습 이력을
처음부터 재현하거나 기존 최종 점수를 보장하는 명령은 아닙니다.
학습 실행은 사용자가 위 명령을 실행할 때만 시작됩니다.

## 검증

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

기존 실험·큐·로그·데이터셋은 복사하지 않았습니다. 원본 메인 패키지의 계산 모듈과
240번에 필요한 추가 구현만 포함했습니다. 복사 과정에서는 package 상대 import만
정리했고, 기존 phase 적분·spike rollout·readout·loss의 계산을 유지했습니다.

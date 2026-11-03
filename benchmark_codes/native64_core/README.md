# 실제 64×64 core — SW0261

SW0260의 **32×32 core + 64×64 readout**을 실제 **64×64 core + 64×64 readout**으로 변환한 구현입니다. 4,096개 위치마다 4개 oscillator component, dendritic state, membrane state, spike가 직접 계산됩니다. 32×32 spike나 최종 mask를 늘려 쓰지 않습니다.

**학습 코드 추가:** [TRAINING_KO.md](TRAINING_KO.md)에 encoder·core·head를 학습하는
두 역전파 경로와 실행 명령이 있습니다. 실행기는 [train.py](train.py), loss와 경로 분리는
[training.py](training.py)입니다. 아래의 초기화·검증 기록은 기존261 변환 단계의 기록이며,
새 학습의 성능이나 수렴을 의미하지 않습니다.

## 변경 및 초기화

| 부분 | 이번 구현 |
|---|---|
| Encoder | 기존 SW0259의 동일 seed 가중치·정규화 유지 |
| Gamma | 실제 64×64 위치, `[B,8,4096]` |
| Core | 4,096개 위치, component 4개, 기본 추론 1,024 step |
| Graph | 이미지별 계산, top-k 512, 거리 단위 0.25 |
| Core 초기화 | 공개 checkpoint의 원본 16×16 core에서 공간상 4×4 자식으로 매핑 |
| 공유 가중치 | gamma projection, graph network, shared dendritic projection 등 그대로 복사 |
| 위치별 가중치 | omega·kappa·tau를 부모 위치에서 복사하되, 별도 위치 tensor 원소로 생성 |
| Pair 가중치 | direction learner를 두 공간축 모두 매핑 |
| Integration | 동일 seed의 a_d·a_m·b 유지 |
| Spike grouping | 실제 4,096개 spike에서 Q 계산, 실제 graph와 결합 |
| Grouping 정규화 | 기존 고정 계수 1,024를 실제 위치 수 N=4,096으로 변경 |
| Head | 학습된 SW0260 context_rgb 가중치 재사용, 입력 활동은 실제 native64의 `[4096,144]` |

**해상도를 변환한 초기화입니다. native64로 새로 학습한 checkpoint나 성능 결과가 아닙니다.** 기존 SW0260의 FG-ARI 0.8069를 이 모델의 점수로 인용하면 안 됩니다. 자식 위치의 초기 가중치는 같지만, 입력 gamma·graph·상태는 각 위치에서 계산합니다. 학습하면 위치별 가중치가 달라질 수 있습니다.

기존 SW0259/SW0260 코드, checkpoint와 `benchmark/` 평가 코드는 수정하지 않습니다. 원본 checkpoint가 이미 저장소에 있으므로 큰 변환 checkpoint를 중복 업로드하지 않고 로딩 시 결정적으로 구성합니다.

## 실행

`patch_final` 저장소 전체를 받은 뒤 저장소 루트에서 실행합니다. 기존 bundle과 같은 Python·PyTorch 환경을 사용합니다. 각 실행은 새 Python 프로세스에서 시작하세요.

```bash
python -B benchmark_codes/native64_core/verify.py \
  --seed 0 --device cuda --report native64_seed0.json

python -B benchmark_codes/native64_core/portable.py \
  --checkpoint benchmark_codes/readout64_optimization/checkpoints/sw0260_context_seed0.pt \
  --image example128.png --output labels64.npy --device cuda
```

입력은 정확한 uint8 RGB 128×128 이미지입니다. 출력은 64×64 정수 grouping mask이며, 배경 지정이나 제거는 하지 않습니다. seed1·2도 대응 checkpoint로 실행할 수 있습니다.

`verify.py --image example128.png`를 지정하면 실제 graph 생성 → 1,024 step core → spike affinity → grouping → head → 최종 mask까지 추가 검증합니다. 기본 검증은 매핑, gamma, native64 routing, head 역전파를 검사합니다. 이때 실제 optimizer update는 하지 않습니다.

## 학습 인터페이스와 제한

`portable.load(..., trainable=True)`는 core·integration·head의 가중치를 학습 가능하게 반환합니다. encoder는 기본적으로 고정입니다. core 역전파가 필요하면 `found.rollout(gamma, live_tail_steps=64)`처럼 명시해야 합니다. 기본 rollout은 추론용 `live_tail_steps=0`입니다.

최종 signed partition은 기존처럼 미분 불가능합니다. 따라서 단순히 head의 RGB loss를 호출한다고 core까지 학습되는 것은 아닙니다. 새 `train.py`는 별도의 phase/spike body loss로 encoder·core를 학습하고, 활동·gamma를 detach한 RGB loss로 head를 학습합니다. 기본 native64 변환 checkpoint가 이 방식으로 이미 학습됐다는 뜻은 아닙니다.

Graph와 Q는 각각 4,096×4,096입니다. 기존 32×32에 비해 pair 행렬 크기가 16배이며, 기존 정확한 geodesic 연산은 더 큰 계산량이 필요합니다. 서버에서 batch1으로 검증합니다. 큰 batch의 메모리 요구량은 별도 측정이 필요합니다.

검증 근거는 `validation/`에 기록합니다. 전체 평가 점수는 아직 측정하지 않았습니다.

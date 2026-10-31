> UPDATE: proposal withdrawn in peer commit08e384a. Supported run.py training/queue commands now default to physical1. Batched forward with independent per-image objectives remains an explicit experimental option (--microbatch auto/2/4), not a validated faster default. The earlier automatic-default description below is historical. The replacement fused-bmm proposal has not yet been applied here.

# REQUEST_0010 반영: 앞으로의 학습·검증 방법

이 폴더에서 **새로 시작하는 학습**은 physical microbatch를 자동 선택합니다.
`run.py train`과 `run.py queue`의 기본값은 `--microbatch auto`입니다.
이미 실행 중인 서버 작업이나 과거 checkpoint를 자동으로 전환하지 않습니다.

## 유지하는 과학적 조건

- Effective batch는16, 고유 TRAIN3,500장×10epochs, 총2,190Adam updates입니다.
  마지막12장도 버리지 않습니다. 작은 admission은 기존2updates·4장입니다.
- FP32, TF32 비활성화,1024-step forward와 마지막64-step 역전파를 유지합니다.
- 이미지 ID와 epoch로 만든 seed를 이미지별로 전달합니다. 배치 위치에 따라
  초기 phase가 달라지지 않도록 하고, gamma_noise도 이미지별 RNG로 생성합니다.
- PLV·spike·teacher·CE/Dice loss와 Hungarian matching은 **각 이미지마다** 계산합니다.
  각 loss를 실제 logical batch 길이로 나눈 뒤 physical chunk마다 역전파합니다.
  optimizer는16장 처리 후 한 번만 업데이트합니다.
- 253은 TRAIN mask를 읽지 않고,252의 최종 mask feature detach 및254의 live
  gradient 연결도 유지합니다. 평가 데이터·평가 공식·예측 mask 생성 규칙은 같습니다.
- Dense coupling을 sparse로 바꾸지 않습니다. 작더라도0이 아닌 연결과 phase lag를
  없애는 변경은 이 최적화에 포함하지 않습니다.

## 자동 선택과 메모리

GPU의 남은 메모리에서 보수적인 예약량을 만족하는4→2→1 순서로 선택합니다.

| Physical microbatch | 필요한 예약량 |
|---:|---:|
|1|11,000MiB|
|2|16,000MiB|
|4|28,000MiB|
|8 (명시적 실험 옵션)|52,000MiB|

따라서24GB TITAN RTX에서는 빈 GPU 기준 보통2, 큰 GPU에서는4를 선택합니다.
다른 작업과 공유 중이면1로 낮출 수 있습니다. 이 값은 실제 최고 메모리 사용량이
아니라 큐의 보수적인 예약량입니다. 장치별 최적 배치는 아래 측정으로 확인하세요.
실행 중 OOM을 반복하며 재시작하는 자동 fallback은 넣지 않았습니다.

선택한 크기는 `binding.json`, `launch.json`, `training_report.json`에 기록됩니다.
재개할 때 `auto`는 해당 작업에 이미 기록된 크기를 유지합니다. 코드·배치 크기가
다른 과거 결과를 완료된 동일 실험으로 취급하지 않으며 SHA 검증도 유지합니다.

## 실행

새 output을 사용합니다. 기존 서버의 live 실험 디렉터리에 덮어쓰지 마세요.

```bash
python -B run.py --config configs/frontier.json --output /scratch/new_benchmark preflight --dataset all

# 모든 새 작업에 자동 microbatch 선택을 적용하는 기본 큐
python -u -B run.py --config configs/frontier.json --output /scratch/new_benchmark queue --gpus 0,1,2,3

# 단일 작업: 먼저 기존 admission 검사를 수행한 뒤 본 학습
CUDA_VISIBLE_DEVICES=0 python -B run.py --config configs/frontier.json --output /scratch/new_benchmark train --model 253 --dataset clevr --seed 0 --admission
CUDA_VISIBLE_DEVICES=0 python -B run.py --config configs/frontier.json --output /scratch/new_benchmark train --model 253 --dataset clevr --seed 0

# 통제 비교가 필요하면 크기를 명시하고 서로 다른 output을 사용
CUDA_VISIBLE_DEVICES=0 python -B run.py --config configs/frontier.json --output /scratch/mb1_control train --model 253 --dataset clevr --seed 0 --microbatch 1
```

Admission의 logical batch는2라서 physical4의 메모리를 검증하지는 않습니다.
4나8로 큰 GPU를 사용할 때는 별도의 실제 native64 측정을 먼저 수행하세요.

```bash
# GPU에 공간을 확보하고 실행. 실행 중인 다른 학습을 중단하는 기능은 없습니다.
CUDA_VISIBLE_DEVICES=0 python -B profile_microbatch.py --output /scratch/mb_profile253 --model 253 --sizes 1,2,4 --count 4
CUDA_VISIBLE_DEVICES=0 python -B profile_microbatch.py --output /scratch/mb_profile254 --model 254 --sizes 1,2,4 --count 4

# 이미지별 seed·loss·gradient 논리를 빠르게 검증하는 CPU 축소격자 검사
python -B validate_microbatch.py --output /scratch/microbatch_cpu_controls.json
```

Profile은 동일한 이미지·초기 가중치·seed·loss scaling으로 속도, peak memory,
loss, gradient와 Adam 업데이트 후 가중치를 비교합니다.4장은16장 logical batch의
일부를 계산하는 진단이며, 완전한 학습 결과나 segmentation 점수가 아닙니다.
CPU 검사는16-grid의 수치적 통제 실험이고, 실제64-grid의 GPU 가속을 증명하지 않습니다.
예약량과 실제 속도는 GPU, 동시 작업, 모델에 따라 달라집니다.

## 현재 확인된 범위

네 모델의 CPU 통제 검사에서 microbatch1·2·4의 per-image loss/gradient와
이미지별 noise seed 보존을 확인했습니다. 실제 native64에서는 기존 공유 GPU의
남은 메모리 한도 내에 microbatch2가 들어가지 않았습니다. **2배 가속이나 동일한
최종 FG-ARI는 아직 확인하지 않았습니다.** 실제 측정 기록과 코드 SHA는
`validation/microbatch_revision.json`을 확인하세요. 원본 제안은 저장소 루트의
`REQUEST_0010_microbatch.md`에 보존합니다.

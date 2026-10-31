# benchmark_info — 모델·학습·평가·과거 실험 통합 기록

스냅샷: **2026-10-31T19:35:39.082255+00:00**. 이 README는 과거 checkpoint와 현재 새로 학습하는 모델을 구분하기 위한 연구 안내서다. 현재 실행 코드는 [`benchmark_codes`](../benchmark_codes/README.md), 독립 240 추론 패키지는 [`snn_kuramoto_bidirectional/sw240`](../snn_kuramoto_bidirectional/sw240/)에 있다.

## 1. 먼저 구분해야 하는 세 가지

1. **과거 학습된 모델의 평가:** 기존 checkpoint를 그대로 사용한다. 추가 학습 없는 전이 평가도 여기에 포함된다.
2. **과거 checkpoint에서 부분 추가 학습:** encoder·sensory projection 또는 mask head 등 지정한 부분만 학습한다. core 전체를 처음부터 학습한 결과가 아니다.
3. **현재 SW0255 전체 학습:** encoder·graph·활성 core/gating·학습 가능한 readout을 각 데이터셋에서 무작위 초기화부터 독립적으로 학습한다. 과거 240/252/253/254는 이 실험에서 **구조/loss 계열명**이며 과거 최고 가중치를 불러온다는 뜻이 아니다.

과거 최고 점수와 현재 점수를 비교할 때는 이 구분, grid, supervised 여부, 고유 이미지 수, 누적 노출량, 업데이트 수, 평가 이미지·픽셀/패치 규약을 함께 확인해야 한다. 같은 모델 번호만으로 동일 실험이라고 판단하지 않는다.

## 2. 현재 SW0255 확정 조건

| 항목 | 현재 실제 실행 조건 |
|---|---|
| 모델 우선순위 | 252 → 253 → 254 → 240 |
| 데이터 우선순위 | 각 모델의 CLEVR → Multi-dSprites → Tetrominoes를 먼저; 네 모델의 이 12조건 완료 후 Objects Room → CLEVRTex의 8조건 |
| 고유 TRAIN | 데이터셋마다 같은 3,500장; 각 모델에 동일한 선택 ID 적용 |
| 반복 횟수 | 10 complete epochs = 35,000 image exposures |
| batch | effective 16, physical microbatch 1; 마지막 12장도 포함 |
| 업데이트 | epoch당 219, 조건당 2,190 Adam updates |
| seed | **현재 seed0 한 개**, 총 20조건 |
| 시작점 | 데이터셋·모델별 random initialization; 학습된 source checkpoint 불러오기 없음 |
| 통계 | 자기 초기 encoder로 해당 TRAIN 3,500장만 사용해 fit; 정답 mask 사용 없음 |
| 평가 | 각 데이터셋의 고정 별도 평가 ID, 보호 evaluator 유지 |
| 완료 기준 | 실제 학습 + 전체 고정 평가 + 저장 checkpoint/학습 이력/예측 SHA/metric 독립 검증 |

처음의 35,000 unique ×10·세 seed 계획은 사용자 수정에 따라 **3,500 unique ×10**, 이후 **seed0 우선**으로 변경됐다. 기존 35k 실험 및 seed1·2 checkpoint는 검증 후 원래 위치에 보존했다. 이전 명세가 남은 `snapshot/`이나 goal 문구보다 [`configs/active_request.json`](../benchmark_codes/configs/active_request.json)이 현재 실행을 나타낸다. 70k/35k 크기의 부모 cache가 있다고 그 전체를 현재 학습하는 것은 아니다.

우선순위는 메모리 조건을 만족하는 작업의 병렬 입장을 허용한다. 모델별 앞 작업이 끝나기 전 다른 모델이 시작될 수 있다. 다른 사용자 작업을 중단하지 않으며, 여유 VRAM에서 함께 실행한다. GPU당 최대 3개·전체 최대 12개라는 상한과 별도로 native64 작업 11,000 MiB/native32 작업 3,000 MiB의 예약량을 적용한다. 실제 가능 작업 수는 이 예약량과 다른 사용자의 점유량에 따라 더 적다.

## 3. 네 모델의 현재 구조와 gradient 차이

| 현재 variant | native grid | encoder | TRAIN mask | 최종 mask head | 최종 mask loss → body/encoder | 배경 처리 |
|---|---:|---|---|---|---|---|
| 240 | 64×64 | CNN + context residual | 사용 | 없음 | 해당 head 없음; mask 기반 frequency teacher loss는 학습에 사용 | largest-component support + signed partition |
| 252 | 64×64 | CNN + context residual | 사용 | exchangeable NeutralRegionHead | activity features를 detach하므로 head에만 전달; 다른 body loss는 encoder도 학습 | 전용 배경 지정/최대 영역 삭제 없음 |
| 253 (165 계열) | **32×32** | 기본 8-channel CNN | **사용 안 함** | 없음 | mask loss 없음; **phase·spike loss는 encoder까지 역전파** | 기존 largest-component support + signed partition |
| 254 | 64×64 | CNN + context residual | 사용 | 252와 같은 head | **activity → gates/core → encoder까지 전달** | 전용 배경 지정/최대 영역 삭제 없음 |

**현재 학습된 core checkpoint를 공유하는 조건은 없다.** 같은 seed에서 일부 초기값이 같아도 별도 parameter/optimizer 상태이며 학습 중 서로의 가중치를 공유하지 않는다. 데이터·초기화·loss·gradient·optimizer/RNG·업데이트까지 같은 공통 구간만 재사용할 수 있다는 원칙이며, 현재는 다른 학습 경로를 같은 checkpoint로 대체하지 않는다.

253만 32×32인 이유는 기존 비지도 기준점 SW0165의 grid를 유지했기 때문이다. 구조적으로 64×64가 불가능한 것은 아니며, 이번 fresh 학습에는 기존 checkpoint 형상을 유지해야 하는 제약도 없다. 따라서 **현재 비교는 네 모델의 해상도까지 동일한 대조 실험이 아니다.** 특히 Multi-dSprites의 작은 경계에서 이 차이가 중요하다. 253의 64×64 별도 학습 결과는 이 스냅샷에 없다. 현재 grid를 과거/후속 결과에 맞춰 소급해서 바꾸지 않는다.

### 실제 forward

```text
uint8 RGB [B,128,128,3]
  → encoder (8 feature channels)
  → TRAIN-fit mean/std 정규화, [-3,3] clip
  → patch pooling, gamma [8 channels × grid²]
  → image-conditioned graph + 8→4 sensory projection
  → Kuramoto phase → sinusoidal gate
  → phase-state dendritic/membrane integration → component spikes
  → 1,024-step traces (readout은 앞 512 step 제외)
  → positive-component spike correlation / spatial graph partition
  → 240·253: foreground support + signed connected partition
     252·254: 144D activity features + exchangeable region head
  → integer object-region masks
```

Graph는 이미지마다 입력 feature로 다시 만든다. graph 생성 가중치의 고정 여부와 모든 이미지가 같은 adjacency를 쓴다는 말은 다르다. native64는 4,096 patches, native32는 1,024 patches이며 patch마다 4 oscillator components를 사용한다. graph top-k는 각각 512/128, 공간 간격은 16/grid, squared-geodesic T의 contrast 8은 구조 상수다. 사용하지 않는 `geodesic_contrast` parameter를 학습됐다고 주장하지 않는다.

현재 학습은 1,024-step 실제 forward를 하되 마지막 **64 step의 recurrent gradient**를 보존하고 graph 생성의 학습 경로도 연결한다. 전체 1,024-step BPTT가 아니다. loss에 포함된 PLV/spike 상관량과 실제 object readout을 별개로 기록한다. gradient가 존재한다는 사실만으로 각 gate의 인과적 필요성이 증명되지는 않는다.

과거 최고 계열의 spike→Kuramoto pulse 및 synchrony→graph feedback은 꺼져 있었다. dendritic previous-spike 입력과 membrane reset 같은 국소 경로와 구분해야 한다. 현재 정확한 flag/연산 순서는 archive의 source와 [`FreshModel`](../benchmark_codes/vendor/full_benchmark/model.py), [`default_config`](../snn_kuramoto_bidirectional/sw240/model.py)에서 확인한다. 이름에 bidirectional이 들어간다는 이유만으로 모든 feedback이 활성화됐다고 해석하지 않는다.

gamma의 channel 순서와 patch 위치는 입력 표현의 일부다. 임의 permutation을 할 때 graph 위치나 projection을 함께 바꾸지 않으면 동일 출력을 보장하지 않는다. 현재 encoder는 학습 대상이며 gamma는 그 출력이다. gamma 자체를 정답 object ID라고 보거나 channel 재정렬만으로 grouping 문제가 해결됐다고 주장하지 않는다.

### 현재 loss와 optimizer

- 공통 body: phase-PLV objective + **5 × positive-component-spike Q objective**.
- 공통 PLV 설정: bimodality 6, balance 10, coherence 0.5, collapse 1, target density 0.867. 다른 관련 가중치는 실제 코드의 0 설정을 유지한다.
- 240/252/254: mask-defined frequency denoising + gamma anchor. teacher는 **자기 random 초기 상태의 frozen 복사본**이며 학습된 CLEVR teacher를 가져온 것이 아니다.
- 252/254: 정답 배경 ID 0도 다른 정답 영역과 동등하게 Hungarian matching한 class-balanced CE+Dice. 추론 시 정답 mask·물체 수·배경 역할을 넣지 않는다. hard proposals/matching/argmax는 이산 결정이다.
- 253: TRAIN mask와 mask teacher를 받지 않도록 runtime assertion. 평가 정답은 예측 고정 후에만 사용한다.
- Adam LR: encoder 3e-6, 다른 body/integration 3e-5, mask head 0.003. body/head는 별도 optimizer와 norm 1 clipping.
- 모든 선언된 활성 tensor의 finite/nonzero gradient 누적, 실제 parameter 변경, optimizer step과 저장/재개·RNG·학습 순서를 검증한다.
- 현재 수치 정밀도는 FP32이며 `torch.backends.cuda.matmul.allow_tf32=False`, `torch.backends.cudnn.allow_tf32=False`, AMP 없음이다. GPU에서 학습한다고 Tensor Core 고속 행렬곱을 활용하고 있다는 뜻은 아니다. 실제 Tensor Core kernel 사용량을 측정한 profiler 결과는 아직 없다.

이 objective는 기존 frozen-core 적응의 loss 하나를 단순히 35k에서 재사용한 비교가 아니다. 전체학습의 명세는 [`METHODS.md`](../benchmark_codes/METHODS.md), 수식/세부 계산의 최종 근거는 [`loss.py`](../benchmark_codes/vendor/full_benchmark/loss.py)와 보존된 source다.

## 4. 현재 실행 스냅샷과 완료 결과

| 모델 | CLEVR | Multi-dSprites | Tetrominoes | Objects Room | CLEVRTex |
|---|---|---|---|---|---|
| 252 | 학습 81.6% / GPU1 | 학습 88.4% / GPU1 | 학습 92.2% / GPU0 | 대기 | 대기 |
| 253 | 학습·평가 완료 | 학습 72.7% / GPU3 | 학습 75.1% / GPU3 | 대기 | 대기 |
| 254 | 학습 37.4% / GPU0 | 학습 33.0% / GPU2 | 학습 33.0% / GPU2 | 대기 | 대기 |
| 240 | 학습 23.2% / GPU3 | 대기 | 대기 | 대기 | 대기 |

진행률은 3,500개 고유 이미지의 처음 순회 비율이 아니라 **총 35,000 exposures 중 처리 비율**이다. 위 표는 2026-10-31T19:35:39.082255+00:00의 관측이며 실시간 dashboard가 아니다. 정확한 원본 관측은 [`CURRENT_STATUS.json`](CURRENT_STATUS.json)에 있다. 운영상 admission/smoke 완료를 과학적 학습·평가 완료로 세지 않는다.

현재 독립 검증 완료가 확인된 **새 전체 학습** 결과:

| variant / dataset / seed | 모델 grid | 평가 | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---|---:|---:|---:|
| 253 / our CLEVR / 0 | 32×32 | 고정 320장, native128 pixels | 0.428118 | 0.277787 | 0.240651 |
| 같은 새 예측을 과거 평가 형식으로 채점 | 32×32 | 고정 64-grid patch scoring | 0.457315 | 0.273522 | 0.248168 |

두 번째 행은 과거 최고 checkpoint를 다시 평가한 수치가 아니다. 두 행 모두 같은 새 random-trained 모델의 예측이다. **단일 seed 결과이며 세 seed 평균이 아니다.** 실제 학습 35,000 exposures/2,190updates, 10 epoch 순서, checkpoint·예측 SHA, 보호 metric의 독립 재계산을 확인했다. [전체 결과 JSON](../benchmark_codes/results/model253_clevr_seed0.json).

## 5. 과거 checkpoint와 결과 계보

### 5.1 기존 16-grid core가 현재 scratch 모델보다 좋을 수 있는 이유

기존 SW0253/SW0165는 학습된 SW0097 core와 reconstruction encoder를 고정해서 전이 평가했다. 16→32 mapping은 학습된 초기값을 자식 위치로 옮긴 것이다. 16→32는 2×2, 16→64는 4×4의 위치 확장이다. 값의 복사와 parameter를 영구적으로 묶어 공유하는 것은 다르며, 이후 실제 학습/고정 여부를 별도로 확인해야 한다.

확인된 source core의 후속 학습 구간:

| 구간 | 고유 이미지/선택 | 노출량 | Adam updates |
|---|---|---:|---:|
| SW0090에서 선택된 epoch01 | 70,000 | 70,000 | 4,375 |
| SW0095 | 같은 70,000 pool | 70,000 | 4,375 |
| SW0097 | 같은 pool의 4,096 | 4,096 | 256 |
| **확인된 후속 구간 소계** | **70,000 distinct pool** | **144,096** | **9,006** |

그 이전 core 학습과 encoder 사전학습은 이 소계에서 제외했다. 따라서 모델의 전체 수명 동안 정확히 144,096장만 봤다는 뜻이 아니다. 서버 manifest와 source SHA 연결을 확인한 설명은 [provenance 비교](../benchmark_codes/results/HISTORICAL_VS_SCRATCH.md)에 있다.

현재 3,500 distinct/35,000 exposures/2,190updates와는 학습량·다양성·초기화·학습 대상이 다르다. 새 253의 epoch 평균 loss는 46.1868→28.4788로 감소했고 마지막 두 epoch도 29.0075→28.4788이었다. encoder/core gradient·업데이트는 확인했지만 **loss 감소가 FG-ARI 증가를 증명하지는 않는다.**

기존 SW0090에서도 세 seed 평균 FG-ARI가 epoch1 0.774323, epoch3 0.752242, epoch10 0.677605였다. 더 오래 학습하면 항상 좋아진다는 증거는 없다. 노출량·고유 데이터·validation 성능·학습 경로를 구분해야 하며, 현재 저하를 데이터 수 하나의 원인으로 확정하거나 더 학습하면 반드시 복구된다고 약속하지 않는다.

### 5.2 대표 CLEVR 결과 — 서로 다른 계약을 한 순위표로 합치지 않음

모든 수치는 FG-ARI / Foreground IoU / Matched-object IoU 순이다. 아래 과거 값은 현재 scratch 결과와 별개다.

| 실험 | 조건 | 세 seed 평균 또는 명시된 범위 | 의미/주의 |
|---|---|---|---|
| SW0090 | native16, 70k selected epoch1 | .774323 / .548028 / .603848 | epoch10 FG .677605; 추가 반복의 단조 개선 아님 |
| SW0095 | 같은 70k 추가 1회, classifier-aligned spike loss | 상세 원본 기록 참조 | 학습된 core에서 continuation |
| SW0097 frozen arm | native16, 4,096장·256updates | .776348 / .577070 / .601537 | graph adaptive arm .769307 / .575107 / .596571보다 높았음; graph를 항상 고정해야 한다는 증명은 아님 |
| SW0165 | native32, fixed320, frozen signed-partition readout | **.804299 / .609975 / .651149** | 비지도 계열; 새 native32 전체학습 아님 |
| SW0165의 동조건 Slot 참고 | native32 fixed320 | .781652 / .204360 / .203165 | 64-grid Slot 참고와 혼합하지 않음 |
| SW0227 | native64 fixed320 | .858916 / .605286 / .674313 | mask-supervised encoder/projection 계열 |
| SW0237 | native64 fixed320 | .872680 / .626110 / .676331 | TRAIN64, 누적512updates; core/readout 고정 |
| SW0240 full70k | native64 fixed320 | **.888130 / .743444 / .684382** | 70,000 unique ×10; encoder/projection만 적응 |
| SW0249의 240 terminal | native64 fixed320 | **.889187 / .754775 / .686438** | 위 240 이후 동일35k×10 최종 추가 학습 |
| SW0251 | native64 fixed320 | .889629 / .784008 / .707530 | residual/background 보정 계열 |
| SW0252 | native64 fixed320 | **.892326 / .211509† / .738660** | 배경 역할 비고정 head, supervised; †raw FG IoU 의미 다름 |
| native64 동조건 Slot 참고 | fixed320 | .786902 / .198201 / .195203 | 기존 reconstruction-trained Slot checkpoint 참고 |

SW0249의 최종 네 모델 비교는 같은 TRAIN35,000×10·320평가·세 seed였다:

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 240 | .889187 | .754775 | .686438 |
| 237 | .887640 | .718838 | .682593 |
| 244 | .879873 | .788644 | .666593 |
| 245 | .879739 | .788727 | .666623 |

240·237의 FG 차이에 대한 이미지 paired bootstrap 구간은 0을 포함했다. 이것을 엄격한 우열이나 seed 모집단의 불확실성 추정으로 바꾸지 않는다. 이 최종 비교는 encoder와 sensory projection의 추가 학습이며 무작위 core 전체학습이 아니다. 240은 앞선70k 적응 이력도 있어 네 모델의 **평생 총 학습량**까지 같지는 않다.

현재 메인 패키지 `patch_final/snn_kuramoto_bidirectional/sw240`가 정리한 것은 **SW0249 terminal의 실제 240**이다. seed0/1/2 portable checkpoint는 각각 약202MB이며 로컬·서버에 보존한다. weights는 이 GitHub 문서에 업로드하지 않는다. 크기와 SHA는 [`models/legacy240_checkpoint_manifest.json`](models/legacy240_checkpoint_manifest.json)에 있다. `SW240Model()`만 생성하면 학습된 checkpoint가 되는 것이 아니다. 새 SW0255 scratch 240은 이 weights를 사용하지 않는다.

### 5.3 과거 두 추가 데이터셋의 전체 TEST1,000 결과

모두 세 seed 평균이다. 최초360장(ID7000–7359)과 추가640장(ID7360–7999)은 구분해 원본 기록에 보존했다. 아래는 전체1000장이며 고유35k fresh 학습 결과가 아니다.

| 모델 | 데이터 | 방식 | FG-ARI | Foreground IoU | Object IoU |
|---|---|---|---:|---:|---:|
| 240 / SW0250 | Multi-dSprites | 기존 checkpoint | .683310 | .662551 | .722921 |
| 240 / SW0250 | Multi-dSprites | encoder/projection 적응 | .770147 | .826195 | .777474 |
| 240 / SW0250 | Tetrominoes | 기존 checkpoint | .617843 | .460512 | .674918 |
| 240 / SW0250 | Tetrominoes | encoder/projection 적응 | .801673 | .712728 | .819070 |
| 252 | Multi-dSprites | CLEVR supervised checkpoint 전이 | .684124 | .096216† | .749783 |
| 252 | Multi-dSprites | 적응 후 head 학습 | .773413 | .096216† | .799309 |
| 252 | Tetrominoes | CLEVR supervised checkpoint 전이 | .754310 | .244898† | .741599 |
| 252 | Tetrominoes | 적응 후 head 학습 | .907381 | .244898† | .853011 |
| 253 / frozen165 | Multi-dSprites | target-data update 0 | .587447 | .692986 | .649447 |
| 253 / frozen165 | Tetrominoes | target-data update 0 | .810799 | .603889 | .801700 |

240 적응은 해당 TRAIN6,000×10, batch16, 3,750updates로 encoder/projection만 학습했다. 과거252는 여기에 TRAIN256×20, 5,120 head updates를 더한 방식이며 현재 전체학습과 다르다. 원래254는 최종 mask gradient를 encoder/projection에 연결한 별도256장 pilot이었다. 그 3개 작업은 SW0255 전환 때 checkpoint 보존 후 중단됐다. 확인되지 않은 과거254 최종 평균을 임의로 채우지 않는다.

†252/254는 모든 region ID를 양수로 유지한다. 보호 metric의 `prediction != 0`은 이미지 전체가 되어 raw Foreground IoU가 GT foreground 면적 비율과 같아진다. 따라서 240/251의 semantic foreground 검출 품질과 직접 비교하지 않는다. FG-ARI 및 원본 Hungarian Object IoU도 해당 label convention을 명시한 상태로 해석한다.

## 6. 데이터셋·분할·전처리·저장 위치

| 데이터 | 원본 크기/내용 | 현재 고정 평가 | 처리 및 유의점 |
|---|---|---|---|
| 우리 CLEVR | HDF5 native128 RGB, 최대10 objects 계열 | ID1320–1639, 320장 | TRAIN/VAL과 분리; 반복 사용한 validation 계열이며 untouched 최종test라 부르지 않음 |
| Multi-dSprites | 64×64, colored_on_colored, 최대5 sprites/겹침 | TEST7000–7999,1000장 | correct `multi_dsprites`; 원본 mask HWE, RGB nearest128; native64는 원본 pixel 해상도와 일치 |
| Tetrominoes | **35×35**, 물체3개, 검은 배경 | TEST7000–7999,1000장 | 원본 mask EHW; 64-grid 모델도 보호 modal32 경유 후 원본35pixel에 평가; 원본35를64개 비어있지 않은 셀로 나눈 것으로 취급하지 않음 |
| Objects Room | 64×64 | source36000–36999,1000장 | entities0–3 sky/floor/walls→배경0,4–6→objects1–3; in-distribution holdout, 공식OOD 평가 아님 |
| CLEVRTex Original | 공식 full50k, 240×320 원본 | official TEST source0–999,1000장 | RGB/GT 동일 center192 crop(64,24,256,216), RGB bilinear128/GT nearest128; 전체공식TEST5000장 중 고정subset |

현재 각 TRAIN3,500 IDs는 [`contracts/*_split.json`](../benchmark_codes/contracts/)에 전부 있다. Multi-dSprites/Tet의 부모 TRAIN은0–5999 및8000–36999, VAL6000–6999, TEST7000–7999다. Objects Room은 TRAIN0–34999/VAL35000–35999/TEST36000–36999다. CLEVRTex TRAIN source10000–49999에서 부모35k를 고정 선택하고 그중3,500을 선택했으며 VAL source5000–5999와 TEST0–999를 제외한다. cache row 번호와 원본 image ID는 같다고 추정하지 않는다.

| 자산 | 서버 경로 |
|---|---|
| 현재 실험 root | `/work/USERS/kevinswk/experiments/SW0255_full3500` |
| 현재 checkpoint | `.../jobs/model{240,252,253,254}/{dataset}/seed0/{initial,resume}.pt` |
| 보존된 이전35k 실험 | `/work/USERS/kevinswk/experiments/SW0255_full35000` |
| 부모 dataset cache | `.../SW0255_full35000/inputs/{multi_dsprites,tetrominoes,objects_room,clevrtex_original}` |
| 우리 CLEVR | `/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5` |
| CLEVR input cache | `/Data0/kevinswk/patch_v2_sw/collaborative_testing/generated/collaborative_test/SW_0240_native64_70000_cached_io/input_cache` |
| 동료 MOD 데이터 | `/work/USERS/tkim1/mod` |
| 공식 다운로드 보관 | `/work/USERS/kevinswk/datasets` |
| portable 240 exports | `/Data0/kevinswk/patch_v2_sw/collaborative_testing/generated/patch_final_exports/sw240_seed{0,1,2}.pt` |

잘못된 초기 `mdsprites` 계열과 올바른 `multi_dsprites` 계열을 혼동하지 않는다. 우리 HDF5 CLEVR, 동료의 with_masks CLEVR, Slot Attention의 외부 checkpoint 학습 자료를 같은 renderer·분포·전처리·분할이라고 추정하지 않는다. 과거 다른 계약의 `.8946/.2235/.2460` 같은 수치는 현재 native64 Slot 참고 행에 합치지 않는다. 정확한 실험별 출처는 아래 전 실험 원문/기록 archive에서 확인한다.

다운로드 URL·파일 크기·SHA는 [`acquisition_manifest.json`](../benchmark_codes/contracts/acquisition_manifest.json), 재현에 필요한 실제 파일 목록은 [`assets.json`](../benchmark_codes/contracts/assets.json)에 있다. COCO 다운로드 이력은 보존하지만 현재5데이터셋 matrix에는 COCO/OOD를 추가하지 않았다. 원본 dataset과 checkpoint binary는 이 폴더에 포함하지 않는다.

## 7. 평가 방법과 숫자의 해석

- 보호된 `benchmark/evaluation.py`, `make_targets_grid.py`, preflight 규약을 수정하지 않는다. 학습/입력 adapter와 평가 정답 변환을 구분한다.
- 예측은 GT 없이 만들고 integer masks와 protocol/SHA를 고정한 뒤 평가 정답을 연다. GT object count를 inference에 넣거나 mask를 보고 background를 고르지 않는다.
- **FG-ARI:** GT foreground 위치에서 정답 instance 분할과 예측 cluster 분할의 쌍 관계를 adjusted Rand index로 측정한다. 예측 object ID 번호 자체의 일치 여부가 아니다. 이를 위해 Hungarian matching을 하지 않는다.
- **Foreground IoU:** 보호 evaluator가 정의하는 foreground convention의 교집합/합집합. 252/254의 all-positive IDs는 앞서 적은 의미 차이가 있다.
- **Matched-object IoU:** 예측/정답 object 영역을 Hungarian matching해 측정하는 별도 지표다. 논문의 MBO와 자동으로 같은 값으로 취급하지 않는다.
- 단일 object 이미지를 일괄0점으로 넣거나 임의로 모두 제외하지 않는다. 보호 evaluator가 undefined로 판정한 경우만 해당 규칙대로 제외하고 valid_count를 보고한다. 과거 Multi-dSprites TEST1000에는 단일object230장이 있었지만 원본규약 평가의 유효 FG-ARI는1000개였다.
- 픽셀 평가와 modal patch 평가를 혼합하지 않는다. patch GT는 보호 builder로 만들고 prediction은 대응 cell 경계로 원본 pixel에 lift한다. 원본 정답 pixel mask는 변경하지 않는다.
- raw TFRecord를128 RGB로 resize했다는 말과 정답 mask 평가 기준을128로 바꿨다는 말은 다르다. dataset별 계약을 적용한다.
- soft mask는 region ownership 확률, hard mask는 discrete assignment다. RGB reconstruction에 hard mask를 곱한다고 soft mask가 되는 것은 아니다. 과거 Slot 결과도 어떤 mask tensor와 resize/argmax를 썼는지 원문 계약을 확인한다.
- 예전 제안된 any-object-in-patch 허용 점수, oracle merge/coverage, GT-assisted 분석은 canonical FG-ARI와 다른 진단이다. 서로 다른 점수를 최고 성능 행으로 교체하지 않는다.
- seed는 random initialization/order 등의 재현성 제어값이며 학습 방법 이름이 아니다. best seed, seed 평균, 작은 screening, fixed320, 전체TEST1000을 구분한다. 현재 seed0 결과에 세 seed 평균/표준편차를 붙이지 않는다.

## 8. 과거 실험 전체 기록을 찾는 방법

**과거 실험 252개 폴더의 문서 520개 + 공통/진단 문서 42개**, 최근 SW0250–SW0255의6계열 및 관련 연구 기록·코드 **총 5,441개 파일**을 원본 bytes로 보존했다. 원본 텍스트/코드 합계 146.3 MB, 압축 archive 34.1 MB다.

- [전 실험 색인](history/EXPERIMENT_INDEX.md): 각 실험 폴더별 문서/기록 수와 보존된 원문으로 이동.
- `history/experiments/`: 각 과거 실험의 README·START·RESULT·설계·인수인계 문서를 원문 그대로 보존. 시점별 상충 문구도 임의 수정하지 않았으므로 최신 결과 문서와 실제 증거를 함께 확인한다.
- [최근 모델/benchmark 문서 색인](history/RECENT_INDEX.md): SW0250–SW0255 관련 문서 및 현재 패키지 설명.
- [공통·인수인계·진단 문서 색인](history/OTHER_DOCUMENTS_INDEX.md): 개별 SW 실험 폴더 밖의 원문도 별도 보존.
- [전체 파일별 색인 CSV](history/RECORDS.csv): source 경로, ZIP member, 크기, SHA256.
- [전체 파일별 manifest](history/FILES.json): 위 원본 연구 텍스트/코드 기록의 검증 가능한 inventory.
- [원본 기록 압축 archive](history/research_records.zip): 실험 코드·Markdown·JSON 결과/설정·CSV·텍스트 로그 등의 원본 bytes. 대용량 raw dataset·가중치·numpy/tensor binary·cache·인증 자료는 포함하지 않는다.
- [스냅샷 출처와 범위](SOURCE_SNAPSHOTS.json): 두 저장소 HEAD, 포함/제외 범위, source별 통계, 파일 무결성. HEAD는 출처 식별이고 작업 트리 전체가 그 commit과 동일하다는 뜻은 아니다. 각 파일 SHA가 실제 보존 bytes의 기준이다.

`patch_v2_sw/collaborative_test/INDEX.md`는 일부 최신 실험보다 갱신이 늦어, 이 폴더의 색인은 INDEX만 복사하지 않고 실제 하위 실험 폴더 전체를 열거해 만들었다. 원문 중 이미 encoding이 손상된 문서는 복구했다고 주장하지 않고 원본 bytes를 보존했다. 숫자와 현재 설명은 검증 가능한 JSON/명세를 우선한다.

서버에만 있는 전체 checkpoint binary, raw images/masks, 모든 시점의 중간 checkpoint·서버 전용 장기 로그를 GitHub 문서에 모두 복제한 것은 아니다. 이들은 실험별 manifest/README의 서버 위치와 SHA 참조로 추적한다. 기록이 없는 실험의 완료나 결과를 추정해서 채우지 않는다. 현재 미완료 작업의 결과는 스냅샷 이후 별도 검증이 필요하다.

스냅샷 파일과 ZIP 안의 모든 원본 bytes·색인 링크는 저장소 root에서 `python -B benchmark_info/verify.py`로 검증할 수 있다. 검증기는 archive 안의 코드를 실행하지 않는다.

## 9. 실제 실행·검증·보존 원칙

재현은 [`benchmark_codes/README.md`](../benchmark_codes/README.md)를 따른다. Linux/WSL·CUDA·명시된 Python 환경 및 실제 datasets가 필요하며, 현재 새 학습에는 과거 학습된 checkpoint가 필수가 아니다.

```bash
cd benchmark_codes
python -B verify_bundle.py
python -B run.py --config configs/frontier.json --output /work/USERS/$USER/new_sw0255 preflight --dataset all
CUDA_VISIBLE_DEVICES=0 python -B run.py --config configs/frontier.json --output /work/USERS/$USER/new_sw0255 train --model 252 --dataset clevr --seed 0 --admission
python -u -B run.py --config configs/frontier.json --output /work/USERS/$USER/new_sw0255 queue --gpus 0,1,2,3
```

다른 머신은 [`materialize_assets.py`](../benchmark_codes/materialize_assets.py)로 SHA를 확인하며 datasets를 준비한다. 기존 live output root를 재현 실험의 출력으로 사용하지 않는다. 기존 checkpoint·raw예측·실패 이력은 보존하고, 관측 timeout을 작업 종료로 오인해 재시작하지 않는다.

SSH 연결 유지와 GPU 감시는 모델을 호출하지 않는 로컬 background observer가 담당한다. 현재 queue는 controller_v4이며 기존 worker를 유지한 채 여유 VRAM 입장 제한만 조정했다. 실제 새 test 시작/완료를 보고하고, 중간 epoch 진행 메시지를 성능 결과로 취급하지 않는다. 이 문서 작성·업로드를 위해 서버 학습을 중단하거나 모델/loss/데이터를 바꾸지 않았다.

## 10. 아직 증명되지 않은 것

- 각 gating이 다른 설명보다 필요하다는 인과적 기여. gradient/activation 존재와 ablation 성공은 다른 증거다.
- 더 많은 고유 데이터 또는 더 많은 반복이 항상 FG-ARI를 향상시킨다는 명제.
- 기존 최고점에 새 random 전체학습이 동일 예산에서 도달한다는 명제.
- 현재 253과 다른 모델의 해상도까지 같은 대조 비교.
- 이번 문서 시점의20조건 전체 학습·평가 완료, 또는 현재 seed0의 세 seed 재현.
- 서로 다른 supervision·학습 예산·renderer·split·mask convention을 통제한 Slot Attention/동료 모델 대비 보편적 우위.

과거 높은 점수를 지우지 않으며, 그 숫자의 학습 이력과 평가 범위를 현재 새 실험에 그대로 이전하지 않는다. 최종 결과는 원본 prediction/checkpoint/학습 이력과 보호 metric의 독립 검증으로 확인한다.

## 11. Vast.ai H100 비용 문의에 대한 참고 계산

2026-10-31 조회한 공식 안내의 표시값은 H100 SXM 최저 $1.73/시간·중앙값 $2.40/시간이며, PCIe 검색 매물 예시는 약 $2.40–2.55/시간이었다. 웹 안내의 수집/갱신 시각과 실제 예약 시각은 다르므로 현재 재고·최종 결제 가격으로 보장하지 않는다. [Vast 공식 가격 안내](https://vast.ai/article/how-much-does-it-cost-to-rent-a-gpu-in-the-cloud-live-pricing-guide), [PCIe 검색](https://cloud.vast.ai/?gpu_option=H100+PCIE).

현재 TITAN RTX를 공유하는252의 최근 실측 속도는 조건 하나의35,000 exposures에 약45–67시간이다. **H100에서 실제 측정한 처리량은 아직 없다.** 시간당 $2.50을 가정하면:

| H100 실측 가속률이라는 가정 | 조건 하나의 학습 시간 | GPU 비용 |
|---|---:|---:|
| 현재 공유 환경보다3배 | 15–22시간 | 약 $38–56 |
| 5배 | 9–13시간 | 약 $23–34 |
| 10배 | 4.5–6.7시간 | 약 $11–17 |

이 표는 시나리오 계산이지 H100 benchmark나 확정 견적이 아니다. 20조건이 모두 각10시간이면 GPU 비용만 약 $500이지만, 실제 모델 grid·데이터·평가 시간은 다르다. GPU 수를 늘리면 병렬 시간이 줄 수 있어도 비용은 GPU-hour 합계로 계산한다. 1,024-step 반복·CPU partition·FP32 연산 등이 있어 광고상의 Transformer/FP8 배속을 그대로 적용하지 않는다. 먼저 약1시간 실측으로 실제 처리량과 비용을 확인하는 것이 타당하다.

코드 확인 결과 현재는 matmul/convolution의 TF32를 모두 끄고 FP16/BF16 AMP도 사용하지 않는다. Tensor Core는 주로 행렬곱 가속용이며 sin·phase 업데이트·gate의 모든 원소 연산이 그 장치에서 수행되는 것은 아니다. H100의 일반 FP32 성능·메모리 대역폭 개선과 Tensor Core/FP8 광고 수치를 구분한다. TF32/혼합정밀도 전환은 현재 실행 중인 benchmark에 소급 적용하지 않고 별도 속도·정확도 검증이 필요하다. [NVIDIA GPU 연산 설명](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/), [PyTorch 수치 정밀도](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

평가, 환경 준비, storage와 데이터 전송 비용은 별도다. Vast는 host마다 가격이 다르고 stopped instance에도 storage 과금이 지속된다. [공식 과금 규칙](https://docs.vast.ai/guides/instances/pricing). 이 문서 작성 과정에서 유료 instance를 생성하거나 기존 서버 작업을 이전하지 않았다.

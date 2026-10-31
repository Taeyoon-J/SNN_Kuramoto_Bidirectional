# SW_0239_native64_best_70000

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## AUDIT_KO.md

<details>
<summary>AUDIT_KO.md</summary>

Source SHA256: `ea0c98f190111b1bc973026effcf8db904dd80bd9ea666841e14155addaf2ee5`

````markdown
# 완료 후 독립 검증

completion_audit.py는 학습 source fingerprint에 들어 있지 않은 별도 검사 도구다.
등록된 학습 코드·선정 모델·CPU admission·큐를 바꾸지 않는다. 검사 결과에는
이 도구 자신의 source SHA를 별도로 기록한다.

완료 시 실제 checkpoint의 Adam counter, 43,750개 update 이력과 각 pass의
고유 70,000장 완전 사용 및 seed별 shuffle 순서를 검증한다. 원래 모델에서
최종 checkpoint를 CPU로 복원해 전체 모델 hash와 고정 core·입력 통계를 확인한다.
이것은 전체 학습을 다시 실행해 재현하는 검사는 아니다.

세 seed의 완성된 prediction protocol과 Slot artifacts를 먼저 확인한 후,
실제 mask 1,920개를 독립 metric 구현으로 다시 채점하고 평균·paired CI·목표 판정을
검증한다. 사전 지정한 seed0·이미지 3장의 양쪽 모델 figure 6개도 확인한다.

로컬 검사 2개에서 완전한 합성 이력을 통과시키고 반복·누락·순서 변경·비정상 loss와
비교 배열 오류를 거절했다. 아직 실제 70,000장 학습 완료 자료에 이 검사를 실행한
결과는 없다. 서버 배포·실행과 실제 완료 증거는 이후 별도로 기록한다.

````

</details>

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `e6828b3e0d10d10454bdca2bdb015d7df40d3998a2a66bb64402736d0465a538`

````markdown
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

````

</details>

## QUEUED_KO.md

<details>
<summary>QUEUED_KO.md</summary>

Source SHA256: `00bf733c2cd1497db30c4dab3311b2ec8a8d4fa06ea835848813552a1bf632cf`

````markdown
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

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `9694161ea7ba8a976e3674bb0214a235f82b2480f77cc94ac38dbc01dd17f7f8`

````markdown
# SW0239 native64 best-model TRAIN70,000

Continue the highest independently audited full320 candidate among SW0227/SW0237,
using all 70,000 distinct registered TRAIN images for ten complete passes. Keep the
original mask-supervised source-frequency teacher, gamma anchor, actual sensory
projection learning and spike readout. Teacher computations stay on CPU, matching
the accepted target cache exactly. Adam resets for this new data stage.

One supervisor leases at most one GPU and runs seed0, seed1, seed2 sequentially.
Other experiments may use other GPUs. It yields only its own worker when a foreign
GPU user appears; the last committed Adam/model/RNG boundary supports resumption.

Tests, from the repository root:

```sh
python -B -m pytest -q --import-mode=importlib -p no:cacheprovider -W ignore::pytest.PytestConfigWarning collaborative_test/SW_0239_native64_best_70000
```

After source verification, select once, complete all three CPU admissions, then
start exactly one supervisor:

```sh
python -B -m collaborative_test.SW_0239_native64_best_70000.train --select
python -B -m collaborative_test.SW_0239_native64_best_70000.train --admit --seed 0
python -B -m collaborative_test.SW_0239_native64_best_70000.train --admit --seed 1
python -B -m collaborative_test.SW_0239_native64_best_70000.train --admit --seed 2
python -B -m collaborative_test.SW_0239_native64_best_70000.queue
```

The worker checks a real batch's GPU gradients and one actual native64 TRAIN
rollout before optimizer updates. Checkpoints commit every 32 updates and at every
complete pass. Up to 31 uncommitted updates may be replayed after interruption.

Each final seed predicts the fixed 320 validation images. Scoring reads evaluation
GT only after all three prediction protocols and comparable Slot artifacts have
validated. Report all three metrics and the seed mean, preserving the distinction
between mask-supervised ours and reconstruction-unsupervised Slot. Visualizations
use predetermined images 1320/1321/1322 and seed0 for both models.

Source/data preflight records alone do not establish queue admission or training.
Runtime artifacts belong in the generated artifact tree through results_archive.

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `cf64440fd8d41642ce55d27987f0f77e5e44e329587c180d88eafcf071aded96`

````markdown
# 239번 실제 학습 시작 확인

seed0가 GPU 2에서 실제 학습을 시작했다. GPU batch gradient와 native64 TRAIN
전체 forward 검증 후, 저장된 checkpoint에서 실제 Adam step 32와 이력 길이 32가
일치함을 확인했다. worker와 supervisor도 살아 있었다. 근거는 START_EVIDENCE.json이다.

이는 70,000장 학습 완료나 새 성능 점수가 아니다. 전체 계획은 seed당 70,000장·
10 complete passes·43,750개 logical Adam update이며 seed0→1→2를 GPU 하나로 처리한다.

첫 구간에서 HDF5 무작위 읽기의 CPU 압축 해제 병목도 확인했다. 동일한 RGB·mask·
학습 순서를 보존하는 입력 캐시를 240번에서 별도로 준비한다. 이 확인 시점에는
239번의 source·loss·초기 모델·학습 작업을 변경하거나 중단하지 않았다.

````

</details>

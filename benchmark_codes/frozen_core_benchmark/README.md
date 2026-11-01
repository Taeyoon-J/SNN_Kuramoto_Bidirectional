# SW0256: 기존 core 고정 + encoder 미세조정

사용자가 선택한 최종 조건입니다. 과거 SW0255의 무작위 초기화 전체 학습과 구분합니다.

| 항목 | 조건 |
|---|---|
| 모델 |252 supervised /253 unsupervised|
| 데이터 |우리 CLEVR HDF5, Tetrominoes, CLEVRTex Original|
| Seed |각각0·1·2, 총18개 독립 학습|
| 학습량 |조건당 고유 TRAIN35,000장×10epochs =350,000회 노출|
| 업데이트 |logical batch16, physical1, epoch 마지막8장도 유지:21,880updates|
| Core |같은 seed의 학습된 SW0097 native16을64로 strict mapping 후 고정|
| Encoder |252는 기존240의 contextual encoder,253은 원래 reconstruction CNN에서 미세조정|
| Loss |기존 phase/spike loss;252만 기존 frequency teacher와 detached CE/Dice head 추가|
| 정규화 |출발 checkpoint의 통계 유지, 평가 데이터로 재추정하지 않음|
| 역전파 |1024-step forward, 마지막64-step gradient. Core를 통과해 encoder로 전달|
| 평가 |CLEVR 고정320장·원본128pixel 및 기존64patch 보조 점수; 나머지 TEST1000장|

**Core 고정은 `no_grad`로 전체 core 연산을 끊는다는 뜻이 아닙니다.** Graph,
Kuramoto, dendrite, membrane, sensory projection, integration 파라미터는 고정하지만,
encoder를 학습하기 위해 gamma에서 실제1024-step dynamics까지의 연산 그래프는 유지합니다.
따라서 예전 cached-feature head-only 학습보다 오래 걸립니다. Graph 가중치는 고정이지만,
graph 자체는 이미지의 gamma로 매번 계산합니다.

16→64는 각16-grid 위치를4×4 자식 위치로 복사하는 공간 매핑입니다. Per-node
파라미터와 pairwise direction을 strict하게 옮기고 SC identity와 공간 거리만
새 격자에 맞춥니다. 학습된 source bias를0으로 초기화하지 않습니다. SW0130의
integration scalars는 기존 wrapper 초기값으로 고정하며, 기존 fixed8 squared
geodesic·top-k512·FP32/TF32off·이미지별 seed·readout 규칙을 유지합니다.

252의 최종 mask loss는 head만 학습합니다. Encoder는252의 다른 기존 body loss로
학습됩니다.253은 TRAIN mask를 받지 않으며 별도 supervised head도 없습니다.

## 출발 checkpoint와 완료 결과 재사용

`sources.frontier.json`은 세 core와 encoder·전처리·manifest의 실제 크기/SHA 목록입니다.
252는 같은 seed의 `sw240_seed*.pt`에서 encoder와 통계만 가져오며 그 안의
native64 core를 대신 사용하지 않습니다.253은 SW0097 폴더의 원래 encoder를 사용합니다.
252 head는 동일한 기존 architecture로 seed별 초기화한 후 새로 학습합니다.

두 모델의 **추가 학습량**을 같게 맞추는 실험입니다. 두 encoder의 사전 학습 이력은
서로 다르므로 lifetime 전체 데이터 노출량까지 같은 통제 실험이라고 주장하지 않습니다.

출발 가중치 재사용과 이번350,000회 노출 완료는 다릅니다. 기존256×20 head 학습,
6000×10 encoder 적응,3500×10 scratch 결과는 이번 완료 결과로 셀 수 없습니다.
모든 새 작업은 서로 다른 output/binding/optimizer를 가지며, 원본 checkpoint는 수정하지 않습니다.

## 실행

기존 `benchmark/`를 수정하지 않습니다. 새 output 디렉터리를 사용합니다.
서버 데이터는 기존 `configs/frontier.json`의 prepared_inputs를 읽습니다.

```bash
cd benchmark_codes
python -B run_frozen_core.py --output /work/USERS/$USER/SW0256_reproduce preflight
python -B run_frozen_core.py --output /work/USERS/$USER/SW0256_reproduce validate-cpu
CUDA_VISIBLE_DEVICES=0 python -B run_frozen_core.py --output /work/USERS/$USER/SW0256_reproduce train --model 252 --dataset clevr --seed 0 --admission
python -u -B run_frozen_core.py --output /work/USERS/$USER/SW0256_reproduce queue --gpus 0,1,2,3
```

다른 Linux/WSL 기기에서는 기존 `materialize_assets.py`로 데이터 cache를 받고,
원본 checkpoint에 접근 가능한 SSH 계정으로 출발 가중치15개도 받아야 합니다.
실제 가중치·인증 정보는 Git에 포함하지 않습니다.

```bash
python -B materialize_frozen_sources.py --ssh frontier --destination /scratch/sw0256_sources --sources-out /scratch/sources.local.json
# 기존 데이터 가져오기 절차로 생성한 configs/local.json과 함께 사용
python -B run_frozen_core.py --config configs/local.json --sources /scratch/sources.local.json --output /scratch/sw0256 preflight
```

큐는18개 모두에 실제 CUDA admission을 먼저 요구합니다. Admission은2updates·4장
학습과 고정 평가1장의 진단이며 최종 점수가 아닙니다. 진단에서 encoder/head의 실제
gradient·update, core 전체의 비트 단위 불변, checkpoint/Adam 복원을 확인합니다.
큐는 진단에서 측정한 reserved memory에 여유를 더해 다음 작업을 배치합니다.
GPU에 공간이 없으면 예약 상태를 유지하며 다른 사용자의 프로세스를 종료하지 않습니다.
종료된 미완료 worker를 확인 없이 자동 재시작하지 않습니다.

각 checkpoint에서 고정 core의 SHA를 검증합니다. 최종 독립 audit은 initial.pt와
terminal checkpoint의 모든 `wrapped.*` tensor를 직접 비교하고,350,000회 전체 이미지
순서·21,880업데이트·모든 예측 SHA·GT를 읽기 전 예측 고정·원래 metric 재계산을 확인합니다.
세 seed가 모두 끝난 데이터셋만 `reports/model{252,253}_{dataset}.json`에 평균과
표준편차를 작성합니다. 아직 이번18개 조건의 완료 성능은 없습니다.

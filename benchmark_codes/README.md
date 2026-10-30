# SW0255: current four-model benchmark, standalone reproduction

현재 실행 중인 실험을 **이 폴더만으로** 실행하기 위한 코드 묶음입니다.
다른 Git branch·과거 collaborative_test·원본 patch_final 코드 import가 필요하지 않습니다.
Linux/WSL2, Python 환경, CUDA GPU와 실제 데이터 파일은 필요합니다.
대용량 데이터·학습 checkpoint는 Git에 포함하지 않습니다. 데이터 위치·크기·SHA와
가져오는 실행 도구를 함께 제공합니다.

## 현재 확정된 실험

- 각 데이터셋의 **고유 TRAIN3,500장 ×10epochs =35,000 image exposures/seed**.
  이전35,000 unique ×10 계획은 사용자가 명시적으로 축소 승인했습니다.
- 네 모델 모두 **무작위 초기화**, seed0/1/2. 학습된240 checkpoint를 불러오지 않습니다.
- effective batch16, physical microbatch1, epoch 마지막12장도 학습합니다.
  epoch당219updates, seed당2,190updates입니다.
- 총4모델×5데이터셋×3seeds=60개 본 학습입니다.
- 먼저 CLEVR/Multi-dSprites/Tetrominoes의12조건을 완료한 뒤,
  Objects Room/CLEVRTex의8조건을 실행합니다. 각 단계에서 모델252→253→254→240 순입니다.
- 설정·loss는 과거240 checkpoint의 frozen-core 추가학습과 **다릅니다**.
  이 패키지는 현재 SW0255의 전체학습 재구현을 재현합니다.

| 모델 | native patch grid | 학습 mask 사용 | 최종 mask loss의 body gradient | 추론 배경 규칙 |
|---|---:|---|---|---|
|252|64×64|frequency teacher + CE/Dice head|head feature에서 detach|전용 배경 지정·삭제 없음|
|253 (165계열)|32×32|사용하지 않음|mask head 없음|기존 largest-component 배경 규칙|
|254|64×64|frequency teacher + CE/Dice head|encoder까지 연결|전용 배경 지정·삭제 없음|
|240|64×64|mask 기반 frequency teacher|최종 mask head 없음|기존 largest-component 배경 규칙|

253은 현재 실험의 실제32×32 구조입니다. 모든 모델이64×64라는 뜻이 아닙니다.
Multi-dSprites에서 이 차이가 성능에 영향을 줄 수 있습니다.

## 빠른 실행: Frontier에 데이터가 이미 있을 때

```bash
cd benchmark_codes
python -m venv .venv
source .venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install numpy==2.2.6 scipy==1.15.3 h5py==3.16.0 Pillow==12.3.0

# 보호 benchmark·데이터·실제 분할을 먼저 검사합니다.
python -B run.py --config configs/frontier.json --output /work/USERS/$USER/sw0255_reproduce preflight --dataset all

# 단일 seed의 실제 CUDA forward/backward·상태 변경·저장/재개·1장 평가 검사
CUDA_VISIBLE_DEVICES=0 python -B run.py --config configs/frontier.json --output /work/USERS/$USER/sw0255_reproduce train --model 252 --dataset clevr --seed 0 --admission

# 본 학습: 고유3500×10, 그 뒤 고정 평가 전체
CUDA_VISIBLE_DEVICES=0 python -B run.py --config configs/frontier.json --output /work/USERS/$USER/sw0255_reproduce train --model 252 --dataset clevr --seed 0

# 전체60개 큐: 자기 프로세스만 추적, 타 사용자 작업은 종료하지 않습니다.
python -u -B run.py --config configs/frontier.json --output /work/USERS/$USER/sw0255_reproduce queue --gpus 0,1,2,3
```

기존 live 실험 `/work/USERS/kevinswk/experiments/SW0255_full3500`를 output으로 쓰지 마세요.
동일 output·동일 code·동일 설정에서 명령을 다시 실행하면 저장 checkpoint부터 재개합니다.
오류로 끝난 큐 worker는 자동 재시작하지 않으며 로그를 확인한 뒤 처리해야 합니다.

## 다른 컴퓨터: 이 폴더 외의 소스가 없을 때

1. 위 Python/CUDA 환경을 준비합니다. 현재 서버의 정확한 버전은
   `contracts/environment.json`에 있습니다. 다른 CUDA/하드웨어에서 bitwise 일치를 보장하지 않습니다.
2. Frontier 데이터 읽기 권한이 있는 SSH alias를 설정하고 다음을 실행합니다.

```bash
python -B materialize_assets.py --ssh frontier --data-root /scratch/sw0255_data --config-out configs/local.json
python -B run.py --config configs/local.json --output /scratch/sw0255_run preflight --dataset all
CUDA_VISIBLE_DEVICES=0 python -B run.py --config configs/local.json --output /scratch/sw0255_run train --model 252 --dataset clevr --seed 0 --admission
python -u -B run.py --config configs/local.json --output /scratch/sw0255_run queue --gpus 0,1,2,3
```

도구는 데이터/cache와 보호 preflight가 요구하는 참조 파일을 가져와 SHA를 검증합니다.
**계정·SSH 키·비밀번호는 포함하지 않습니다.** 서버 접근 권한 없이 Git 코드만 내려받아
데이터까지 생기는 것은 아닙니다. 공식 원본 다운로드 URL은
`contracts/acquisition_manifest.json`, 실제 재현에 필요한 파일은 `contracts/assets.json`입니다.
우리 CLEVR HDF5는 별도의 데이터이므로 공식 CLEVR TFRecord로 임의 대체하면 동일 실험이 아닙니다.

## 학습 방법과 평가

상세 내용은 [METHODS.md](METHODS.md), 위치와 checkpoint 설명은 [DATA_AND_CHECKPOINTS.md](DATA_AND_CHECKPOINTS.md)를 참고하세요.
`contracts/*_split.json`은 실제 TRAIN/validation/evaluation ID 전체입니다.
`vendor/`는 실제 서버에서 읽은 원본 코드이며 `contracts/source_snapshot.json`에 SHA가 있습니다.
상위 저장소의 `benchmark/`를 수정하지 않고, 실행 당시 보호된 사본을 `vendor/benchmark/`에 포함했습니다.

`portable_runtime.py`는 파일 경로를 설정하고, phase2의 parent35k cache 검사와
실제3500 subset 검사를 연결합니다. 모델·loss·평가 공식을 바꾸지 않습니다.
`vendor/`의 역사적 보조 CLI를 직접 실행하지 말고 위 `run.py`를 사용하세요.

완료 후 결과는 `reports/model{번호}_{dataset}.json`에 seed별/평균/표준편차로 저장합니다.
```bash
python -B run.py --config configs/local.json --output /scratch/sw0255_run audit
CUDA_VISIBLE_DEVICES=0 python -B run.py --config configs/local.json --output /scratch/sw0255_eval evaluate --model 252 --dataset clevr --seed 0 --checkpoint /scratch/sw0255_run/jobs/model252/clevr/seed0/resume.pt
```

현재 전체 학습이 끝난 결과를 포함한 패키지는 아닙니다. 실행 코드를 재현하기 위한 패키지입니다.

## 포함된 실제 검증 기록

`validation/isolated_review.json`은 다른 프로젝트 소스를 import하지 않는 별도
서버 디렉터리에서 수행한 검증입니다.

- 77개 vendored source 파일이 실행 중 서버 원본과 SHA 일치.
- 네 모델×세 seed, 총12개 무작위 초기 상태·활성 parameter group이 원본과 일치.
- 다섯 데이터셋의3500개 TRAIN ID와 고정 평가 ID, 보호 preflight 통과.
- 다섯 데이터셋의 동일 mask 입력에 대한 보호 evaluator 결과 일치.
- 253/CLEVR은 실제 CUDA에서2updates·4image exposures, 모든 활성 tensor의
  gradient·parameter update, 저장/복원, 고정 평가1장까지 통과.

이 마지막 검사는 **CUDA smoke**이며 모델의 최종 benchmark 성능이 아닙니다.
같은 환경·동일 설정이어도 원래 CUDA backward의 atomic 연산 반올림 때문에
서로 다른 실행의 전체 학습이 bitwise 동일하다고 주장하지 않습니다.

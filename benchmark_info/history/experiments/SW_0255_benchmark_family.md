# SW0255 모델/benchmark 계열

## benchmark_codes/DATA_AND_CHECKPOINTS.md

<details>
<summary>DATA_AND_CHECKPOINTS.md</summary>

Source SHA256: `726147dabb963abc1e4445943107c1245b03e9460b1462c6da9ec308fabfb0d6`

````markdown
# Data locations and checkpoint requirements

## What is and is not required

**No existing learned checkpoint is required to start any current SW0255 job.**
The four names240/252/253/254 refer to architecture/loss families, not loading
their historical best weights. The teacher used by the supervised families is
made from their own fresh initialization.

To resume a newly trained job, keep that job's `initial.pt`, `resume.pt`,
`binding.json`, `effective_code.json` and history together. Do not transplant a
checkpoint across different data, code paths or backward recipes.

Legacy240 seed0/1/2 exports are optional historical assets, not training inputs:

`/Data0/kevinswk/patch_v2_sw/collaborative_testing/generated/patch_final_exports/sw240_seed{0,1,2}.pt`

Their file sizes/SHA are in `contracts/legacy_checkpoint_manifest.json`.
Do not load them into this scratch protocol and still call it scratch training.

Current checkpoint pattern:

`/work/USERS/kevinswk/experiments/SW0255_full3500/jobs/model{252,253,254,240}/{dataset}/seed{0,1,2}/{initial,resume}.pt`

Previous35k jobs were retired with their original files retained. Verification
records are in `snapshot/previous35000_retirement.json`. They are not silently
reused for the3500 subset.

## Dataset/cache paths on Frontier

| Asset | Location |
|---|---|
|Our CLEVR HDF5|`/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5`|
|Our CLEVR RGB/labels cache|`/Data0/kevinswk/patch_v2_sw/collaborative_testing/generated/collaborative_test/SW_0240_native64_70000_cached_io/input_cache`|
|Four prepared dataset caches|`/work/USERS/kevinswk/experiments/SW0255_full35000/inputs/{multi_dsprites,tetrominoes,objects_room,clevrtex_original}`|
|Corrected peer MOD artifacts|`/work/USERS/tkim1/mod` (use `multi_dsprites`, never `mdsprites`)|
|Protected preflight reference|`/work/USERS/tkim1/clevr/with_masks/targets_v1.pt`|
|Official raw Multi-dSprites|`/work/USERS/kevinswk/datasets/multi_dsprites/raw/multi_dsprites_colored_on_colored.tfrecords`|
|Official raw Tetrominoes|`/work/USERS/kevinswk/datasets/tetrominoes/raw/tetrominoes_train.tfrecords`|
|Official raw Objects Room|`/work/USERS/kevinswk/datasets/objects_room/raw/objects_room_train.tfrecords`|
|CLEVRTex Original full PNG/masks|`/work/USERS/kevinswk/datasets/clevrtex_original/data/clevrtex_full`|

`contracts/assets.json` is the executable inventory: exact file paths, size, SHA
and portable destination. It totals approximately16.6GB, copied only when moving
to a new data location. The70k/35k sizes of parent caches do not mean those many
images are trained in the current3500-subset experiment.

`materialize_assets.py` fetches those assets and writes a path config. On Frontier
you can use them in place with `configs/frontier.json`, without copying16.6GB.
Public source URLs and integrity evidence for all downloaded raw datasets are
in `contracts/acquisition_manifest.json`. COCO/OOD artifacts in that historical
acquisition log are not part of the current five-dataset matrix.

## Splits and preprocessing

- CLEVR parent train: fixed35000 selection from0–999 plus1640–70639; no reserved
  IDs1000–1639. Validation1000–1319, evaluation1320–1639. All3500 IDs included.
- Multi-dSprites/Tet parent train:0–5999 plus8000–36999. Validation6000–6999;
  TEST7000–7999. RGB nearest128. Multi-dSprites masks are HWE in raw source,
  Tet masks EHW; the corrected cache has been checked against peer pixel labels.
- Objects Room parent train0–34999; validation35000–35999; TEST36000–36999.
  Raw entities0–3 (sky/floor/two walls) merge to GT background0;4–6 map to1–3.
- CLEVRTex parent train is a fixed35000 subset of official10000–49999, selected
  with RandomState25535000. Validation source5000–5999, TEST source0–999.
  Compact cache row IDs map to raw source IDs via `*_parent_ids.npz`.
  RGB and mask both center-crop box(64,24,256,216), then RGB bilinear128 and mask
  nearest128. TRAIN targets use protected modal64. Official loader SHA is pinned.

All actual selected3500 IDs, validation IDs and evaluation IDs are stored in
`contracts/*_split.json`; the runtime checks them against the reproduced Dataset.
No answer-mask resizing is left implicit.

````

</details>

## benchmark_codes/METHODS.md

<details>
<summary>METHODS.md</summary>

Source SHA256: `7e044ae302603931469a3e329c3ab56fc4a8586ddf62655fd5926db1857cad25`

````markdown
# Exact current method

## Initialization and forward

`vendor/full_benchmark/model.py:FreshModel` constructs each model from random
seed255000+seed. No pretrained `torch.load` is involved. The shared dendritic
projection bias starts at zero for every variant. The supervised variants have a
contextual8-channel CNN;253 uses its base CNN. Each image is uint8 RGB128×128.
Own-INITIAL encoder statistics are fitted over exactly the chosen3500 TRAIN
images without opening masks (per-channel for240/252/254, global scalar for253).
Features are normalized, clipped to[-3,3] and pooled into patch γ.

Each patch has4 oscillator components. The learned image-conditioned graph uses
fixed8 squared-geodesic distance, spatial spacing16/grid, three geodesic stages,
and top-k512 at64-grid or128 at32-grid. The Kuramoto/sinusoidal/dendritic/membrane
and spike trace has1024forward steps; correlation/readout discards the first512.
Training backpropagates through the last64 time steps, including live graph
construction. It is **not1024-step full BPTT**. The factorized Kuramoto backend,
phase integration wrapper and all active groups are those in the frozen source.

No16-grid checkpoint is expanded in this experiment: each native grid is freshly
constructed. Fixed architectural constants/buffers are not learned. In
particular unused `geodesic_contrast` is explicitly excluded from trainable groups.

## Losses and backward

Every variant uses phase-PLV loss +5×positive-component-spike-Q loss, with
bimodality6, balance10, coherence0.5, collapse1 and target density0.867. Other
UnsupervisedS2NetLoss weights are zero as set in `full_benchmark/loss.py`.

240/252/254 also use mask-defined frequency denoising + γ anchor. Their frozen
teacher is a copy of **their own random INITIAL encoder/projection/frequency**,
not a learned CLEVR teacher. It makes per-truth-region frequency means and
class-balanced weights. Both denoising and anchor coefficients are1.

252/254 use the identical NeutralRegionHead: phase/spike/membrane activity yields
144features/patch; a LayerNorm/MLP embeds them, combines inference-derived region
proposals with12 deterministic farthest-point proposals, and predicts region
logits. All truth regions, including background ID0, are Hungarian-matched to
exchangeable predictions for class-balanced CE + Dice. Neither training nor
inference fixes a dedicated background slot. Hard proposal construction,
Hungarian matching and output argmax remain discrete.

- 252 detaches activity features before final mask-head loss; that loss trains
  the head. Other body losses still train the body.
- 254 keeps activity features live so final mask loss can reach the body/encoder.
- 253 receives no masks and no teacher; the engine asserts no TRAIN labels were
  opened. Masks are loaded only after frozen inference for evaluation.
- 240 has no final NeutralRegionHead; it uses the original largest-component
  foreground support and signed-energy partition readout.

Adam body LR3e-5; encoder LR3e-6; mask-head LR0.003 with separate Adam. Body/head
gradients are separately clipped to norm1. Each image objective is divided by the
actual logical-batch length. All active tensors must accumulate finite nonzero
gradient coverage, each active group must update, and optimizer restoration is
verified before evaluation. `training_report.json` records the actual evidence.

## Data order and budget

The immutable parent TRAIN pool is35000 per dataset. A sorted
RandomState2553500 choice selects3500 unique rows without replacement, shared
across variants/seeds. Each epoch shuffles these with
RandomState(249000+seed+10000×epoch). Inference/training initialization uses
255000+image_ID+100000×epoch. Ten epochs yield35000exposures and2190Adam updates.
The last12 images are retained. γ/graph/activation are computed anew per image.

`initial.pt` stores own INITIAL state/statistics. `resume.pt` atomically saves
model, optimizers, gradient coverage, cursor and RNG state after the first update,
every8updates and each epoch end. Existing pretrained weights are never reused.

## Fixed evaluation

Every image is predicted without GT. Native integer masks are frozen and hashed
before GT is opened for scoring, using the unchanged protected
`vendor/benchmark/evaluation.py` and `make_targets_grid.py`.

| Dataset | Fixed evaluation | Pixel scoring |
|---|---|---|
|Our CLEVR HDF5|IDs1320–1639,320 reserved images|native128pixels; historical64-grid patch scores also reported|
|Multi-dSprites colored_on_colored|TEST7000–7999,1000images|original64×64pixels|
|Tetrominoes|TEST7000–7999,1000images|original35×35pixels;64-grid output first modal32 then protected lifting|
|Objects Room|source36000–36999,1000images|original64pixels; in-distribution heldout, **not official OOD test**|
|CLEVRTex Original|official TEST source0–999,1000image subset|official center192crop and nearest128pixel mask|

Tetrominoes is35×35, not32×32. Native32 model253 predictions are lifted with the
same protected cell boundaries. Multi-dSprites uses the corrected `multi_dsprites`
artifacts; the old malformed `mdsprites` variants are never substituted.

FG-ARI uses only GT foreground pixels and excludes undefined one-object cases
according to the protected evaluator; it does not insert artificial zeros.
Foreground IoU and Hungarian-matched Object IoU use that same protected evaluator.
For252/254 all predicted region IDs are positive, including whatever region
captures GT background. Therefore raw Foreground IoU has this specific semantic
limitation; no post-hoc largest-region deletion is applied to improve that metric.

The active queue runs seed0 only for each of the20 conditions. Reports identify
this as a single-seed result; no multi-seed mean or standard deviation is claimed.
Previously started seed1/2 checkpoints are retained in place but not queued. The independent audit recomputes metrics from frozen masks
and verifies all epoch image orders, all35000exposures,2190optimizer steps,
checkpoint identities and saved prediction hashes.

The small-budget result is not a35000-unique/70000-unique result, not a
reproduction of Slot Attention's500000-step training, and not by itself a
controlled proof of each gating's causal contribution.

````

</details>

## benchmark_codes/README.md

<details>
<summary>README.md</summary>

Source SHA256: `c152b8a9008038a9877c019db9ebe0917e8c680084bfa188a5f3d1cf0aa30e8c`

````markdown
> Active protocol: 3500 unique TRAIN x10 epochs, seed0 only. Previously started seed1/2 checkpoints remain preserved and inactive. `configs/active_request.json` overrides the historical `snapshot/controller/REQUEST.json`.

# SW0255: current four-model benchmark, standalone reproduction

현재 실행 중인 실험을 **이 폴더만으로** 실행하기 위한 코드 묶음입니다.
다른 Git branch·과거 collaborative_test·원본 patch_final 코드 import가 필요하지 않습니다.
Linux/WSL2, Python 환경, CUDA GPU와 실제 데이터 파일은 필요합니다.
대용량 데이터·학습 checkpoint는 Git에 포함하지 않습니다. 데이터 위치·크기·SHA와
가져오는 실행 도구를 함께 제공합니다.

## 현재 확정된 실험

- 각 데이터셋의 **고유 TRAIN3,500장 ×10epochs =35,000 image exposures/seed**.
  이전35,000 unique ×10 계획은 사용자가 명시적으로 축소 승인했습니다.
- All four models start from random initialization. **Current queue: seed0 only.**
- effective batch16, physical microbatch1, epoch 마지막12장도 학습합니다.
  epoch당219updates, seed당2,190updates입니다.
- **4models x5datasets x1seed =20 scientific training jobs.**
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

# Queue all20 seed0 jobs; preserve prior seed1/2 checkpoints.
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

Each completed condition is independently audited in `reports/single_seed/model{number}_{dataset}.json`; no multi-seed mean is claimed.
```bash
python -B run.py --config configs/local.json --output /scratch/sw0255_run audit
CUDA_VISIBLE_DEVICES=0 python -B run.py --config configs/local.json --output /scratch/sw0255_eval evaluate --model 252 --dataset clevr --seed 0 --checkpoint /scratch/sw0255_run/jobs/model252/clevr/seed0/resume.pt
```

전체20조건은 아직 완료되지 않았습니다. 완료·독립 검증된 과학적 결과는
[results/README.md](results/README.md)에 seed별로 기록합니다.

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

## GPU queue capacity

The queue permits up to3 owned workers per GPU and12 across four GPUs.
Every active native64 worker retains an11000MiB reservation and native32 an3000MiB
reservation, including not-yet-allocated headroom. Memory fitting remains required.
This lets a queued native64 job share a GPU running two smaller native32 jobs;
other users and running training processes are never stopped for this admission.

Historical checkpoints and current scratch variants have different training histories. See [the provenance and budget comparison](results/HISTORICAL_VS_SCRATCH.md), including the distinction between encoder phase/spike gradients and final supervised-mask gradients.

````

</details>

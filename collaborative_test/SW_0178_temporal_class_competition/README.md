# SW0178: temporal class competition

## 상태

두 supervised 비교 학습 큐를 등록했습니다. 성능 평가는 아직 없으며 기존
SW0165 unsupervised 최고 모델은 유지합니다. 정답 modal mask는 TRAIN loss에만
사용합니다. 추론은 실제 emitted spike → native product Q → 기존 SW0165
readout으로 동일하며 정답 object 개수나 mask를 입력하지 않습니다.

- `mean_rho_ce`: 네 component의 signed correlation 평균으로 물체 간 경쟁을 학습.
- `product_q_ce`: 기존 positive-product Q로 같은 경쟁 loss를 학습하는 대조군.
- 동일한 초기 모델, TRAIN512 × 2 passes, B16/micro4, 64 updates.
- 새 decoder/head, SW0176 gradient hook, SW0177 polarity head는 없습니다.

서버 검증 18개 통과. 실제 TRAIN8의 ID/RGB/modal mask를 SW0171 데이터에
교차 확인했으며 두 loss의 cached-spike leaf gradient는 finite-positive입니다.
이는 모델 parameter credit의 증거가 아니며 첫 두 committed GPU update에서
실제 S gradient 및 네 optimizer family의 gradient/parameter 변화를 검사합니다.
두 실제 초기 모델의 state SHA와 0-update 저장·복원이 일치했습니다.

학습에 사용하는 vectorized loss는 원래 explicit loss와 값 및 affinity
gradient가 일치하는 fixture를 통과했습니다. 원래 explicit 구현의 실제 TRAIN8
CPU 검사도 완료됐으며 학습에는 vectorized 구현을 사용했습니다.

평가 조건과 사전 판정은 [DESIGN_KO.md](DESIGN_KO.md)에 있습니다. 두 arm은
각각 독립적으로 두 readout endpoint에서 SW0164 seed1 대비 FG-ARI 개선과
paired CI 양수, 두 IoU 유지 조건을 만족해야 확장합니다. 검증 예측을 모두
고정한 뒤에만 fixed320 GT를 엽니다. Reserve 데이터는 사용하지 않습니다.

검증 원문: `results_archive/cpu_v1/cpu.json`, 큐 등록:
`results_archive/cpu_v1/launch.json`. 등록 시 관찰은 `observed.json`이며 현재
상태를 보장하지 않습니다. GPU 소유권은 실행 중 실제 프로세스로 확인합니다.

동료 브랜치 `origin/patch_v2`를 이번 사이클에 다시 확인했습니다.
`cab6c676e24a149a45ea960586c04e579f78d384`로 이전 확인 이후 변경이 없습니다.

## 실제 GPU 학습 확인

GPU 1·2에서 두 학습 작업자가 실행됐습니다. 각 arm의 첫 두 committed
updates에서 실제 emitted S로의 loss gradient와 context encoder/graph/native
core/integration 네 optimizer family의 gradient 및 parameter 변화가 모두
finite-positive임을 저장 경계로 확인했습니다. 실제 모델 tensor가 초기 상태와
달라졌고 Adam step과 committed update 수도 일치했습니다. 검증 당시 두 arm
모두 5 updates가 저장됐습니다. 이는 전체 학습 완료나 성능 향상의 증거가
아니며 gating 제거에 따른 인과 기여를 증명하지 않습니다.

원문은 `results_archive/gpu_v1/first_two.json`, 독립 검증 코드는 `verify.py`,
실제 PID/GPU 관찰은 `observed.json`입니다. 검증에서는 validation GT를
읽거나 optimizer를 실행하지 않았습니다.

## 자동 평가 큐 등록

학습·평가 서버 검증 31개와 실제 중간 저장 모델 두 개의 CPU strict-load
검사를 통과했습니다. 모델 검사는 39-update 경계를 사용한 private fixture이며,
64-update 완료 artifact의 검증으로 간주하지 않습니다. 최종 artifact는 실제
학습 완료 뒤 다시 검증합니다. 이 준비 단계에서는 validation GT를 읽지 않았고
optimizer update나 GPU 추론을 실행하지 않았습니다.

예측 부모 두 개와 점수 부모 한 개를 서버에 등록하고 실제 PID/argv를
확인했습니다. 두 학습이 완료되면 실제 S/native Q로 conditional·standalone
예측을 각각 생성합니다. 성공한 arm의 fixed320 예측이 모두 고정되고 hash가
검증된 뒤 GT를 열어 세 지표와 paired bootstrap을 계산합니다. 한 arm이
학습 단계에서 거절돼도 다른 성공 arm은 독립 평가하며, 둘 다 거절되면
GT를 열지 않습니다. 현재 평가 점수는 없습니다.

CPU 검증과 큐 등록 원문: `results_archive/eval_v1/cpu.json`, `launch.json`.

## 학습 완료, 평가 대기

두 arm 모두 64 logical updates를 완료했습니다. 각각 고유 TRAIN512장을
두 번 사용한 1024 exposures, 실제 Adam step 64, 최종 모델 tensor digest,
최종 checkpoint와 atomic boundary의 tensor 일치 및 frozen source 유지가
검증됐습니다. 이 시점의 성능 점수는 없으며 최고 모델은 변경하지 않습니다.
두 예측 부모와 score 부모는 학습 뒤의 실제 artifact 검증·평가를 위해
계속 실행 중입니다. 학습은 재시작하지 않았습니다.

`results_archive/train_v1/audit.json`에 독립 완료 검증을 기록했고, `rho.json`과
`qce.json`은 각 최종 manifest입니다. `verify.py`는 검증 코드입니다.
느린 explicit loss의 실제 TRAIN8 CPU 검사도 완료됐습니다. Vectorized
구현과 loss 차이는 1e-6 미만, S leaf gradient norm 차이는 1e-8 미만입니다.
이는 실제 데이터의 loss·gradient norm 비교이며 전체 gradient vector의
동일성 주장은 아닙니다. 작은 fixture에서는 전체 affinity gradient를 비교했습니다.
참조 검사는 `reference.json`, 수치 비교는 `loss_check.json`에 보관했습니다.

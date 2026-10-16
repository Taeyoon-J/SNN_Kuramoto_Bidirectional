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
CPU 검사는 계산량이 많아 별도 실행 중이며 학습용 구현은 vectorized 버전입니다.

평가 조건과 사전 판정은 [DESIGN_KO.md](DESIGN_KO.md)에 있습니다. 두 arm은
각각 독립적으로 두 readout endpoint에서 SW0164 seed1 대비 FG-ARI 개선과
paired CI 양수, 두 IoU 유지 조건을 만족해야 확장합니다. 검증 예측을 모두
고정한 뒤에만 fixed320 GT를 엽니다. Reserve 데이터는 사용하지 않습니다.

검증 원문: `results_archive/cpu_v1/cpu.json`, 큐 등록:
`results_archive/cpu_v1/launch.json`. 등록 시 관찰은 `observed.json`이며 현재
상태를 보장하지 않습니다. GPU 소유권은 실행 중 실제 프로세스로 확인합니다.

동료 브랜치 `origin/patch_v2`를 이번 사이클에 다시 확인했습니다.
`cab6c676e24a149a45ea960586c04e579f78d384`로 이전 확인 이후 변경이 없습니다.

# SW_0035: adaptive object-count classifier baseline

Status: planned; code prepared while SW_0034 full validation is running.

## 목적

Classifier와 core 진단을 병행하기 위한 기준선입니다. 실제 membrane 또는
gate가 곱해진 spike pattern의 spectral eigenspectrum에서 이미지별 cluster
수를 정하고 patch를 묶습니다. GT object 수는 예측에 넣지 않습니다.

이 기준선은 엄격한 연구 단계 통과 조건이 아닙니다. count 추정, grouping,
background 판별 중 무엇이 약한지 수치로 보여 주고, core/loss 실험과
classifier 실험이 서로 다음 선택에 정보를 주도록 합니다.

## 비교 항목

- Normalized affinity eigenvalue의 후보 구간 내 최대 eigengap.
- Eigenvalue threshold count, 후보 범위로 clamp.
- 기존 membrane adaptive slots (`threshold=.7`, six initial slots).
- 고정 k10 spectral은 count 추정기가 아닌 grouping control.
- membrane, membrane×sigma1.5 spatial, gated-spike×spatial 신호.
- 세 patch metric과 함께 count MAE/bias, exact/within-one count,
  foreground fraction, 선택된 cluster-count histogram.

Largest group/slot을 background로 두는 현재 rule도 같은 기준점의 일부이며,
foreground fraction이 높으면 count와 별도로 background 병목으로 기록합니다.
Validation pilot에서 configuration을 비교하되 이미지별 GT count를 사용하지
않습니다. Pilot 결과를 본 뒤 full320으로 올릴 설정을 소수로 고정합니다.

## Reproduction

```bash
bash collaborative_test/SW_0035_adaptive_count_baseline/run.sh GPU_ID 64
```

Frozen SW_0003 seed-0 checkpoint, validation IDs 1320–1383, 256 steps and
64-step settle. No training, core change, reference-test selection, or goal claim.

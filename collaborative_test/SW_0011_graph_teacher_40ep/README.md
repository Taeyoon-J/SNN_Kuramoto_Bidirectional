# SW_0011 — graph 힌트 학습을 기본 모델과 같은 40 epoch로 연장

Status: detached training running on server GPU 3; validation and reference evaluation not yet done.

## 쉽게 설명하면

SW_0008에서 graph 힌트로 membrane을 학습한 모델과 동료 spike 분류법의
조합이 검증 세 지표를 모두 올렸습니다. 하지만 그 모델은 10 epoch만
학습했고 비교 기준은 40 epoch였습니다. 이번에는 **다른 설정을 유지하고
학습 길이만 40 epoch로** 맞춥니다. 잘게 쪼개지는 문제를 학습으로 줄일
수 있는지 확인합니다.

## 정확한 조건

- SW_0004와 같은 학습 데이터 gamma 첫 1000장, seed 0, batch 16,
  learning rate 1e-3, 64 time steps와 처음 32 settle.
- 위상 PLV loss와 graph-to-membrane synchrony loss weight 0.1,
  temperature 0.1, 실제 component spike 활성. 나머지 설정은 `run.sh` 참조.
- SW_0004 대비 epoch만 10→40. core 계산, architecture, loss 수식 변경 없음.
- 학습 뒤 검증 ID 1320–1639에서 동료 분류법을 먼저 비교하며, reference
  ID 1000–1319는 검증에서 읽기 설정을 고른 뒤 고정된 설정으로만 평가.
- 목표의 3-seed 평균에는 seed 1/2 검증도 별도로 필요.

# SW_0009 — 검증에서 선택한 분류법을 고정하고 reference에 적용

Status: seed-0 reference evaluation complete; goal not met.

## 쉽게 설명하면

SW_0008 검증에서 세 점수가 함께 좋아진 설정을 **더 이상 바꾸지 않고**
reference 이미지 320장에 적용합니다. 동료 방식으로 spike를 연결하되,
4개 oscillator 성분의 양의 동기화를 곱하고 연결 임계값은 0.50으로
고정했습니다. 가장 큰 연결 덩어리를 배경으로 둡니다.

## 고정 조건

- SW_0004 graph-teacher seed-0, 10-epoch checkpoint 그대로 사용.
- 실제 spike 256 step, 초기 64 step 제외, 성분별 centered signed correlation.
- 양의 correlation만 곱하고 0.50 이상인 patch 연결; 연결 덩어리별 물체.
- CLEVR HDF5 ID 1000–1319, 16×16 patch. 정답은 점수에만 사용.
- `evaluate.py`는 이 한 설정만 산출하며 test 결과로 임계값을 다시
  고르지 않습니다. 결과에는 per-image 세 지표도 기록합니다.
- 저장된 Slot Attention 비교값은 같은 reference 이미지의 **한 checkpoint**
  결과입니다. 이전에 이 split의 시작 baseline도 검사했으므로 완전히
  untouched test라는 주장은 하지 않습니다. 최종 목표에는 3-seed 평균과
  추가 독립 검증이 필요합니다.

| 같은 reference 이미지 320장 | FG-ARI ↑ | 전경 IoU ↑ | 물체별 IoU ↑ |
|---|---:|---:|---:|
| 저장된 Slot Attention 한 checkpoint | 0.8901 | 0.2123 | 0.2355 |
| 이번 SW_0004 + 고정 동료 분류법, seed 0 | 0.1909 | 0.2709 | 0.2298 |

**해석:** 검증에서 본 전경 IoU 개선은 reference에서도 유지됐습니다.
그러나 FG-ARI는 Slot Attention보다 크게 낮고, 물체별 IoU도 0.0057
낮습니다. 예측 물체 수는 이미지당 평균 30.14개, 예측 전경은 patch의
55.84%입니다. 3-seed 평균이나 목표 달성으로 해석할 수 없습니다.
자세한 per-image 점수와 실제 spike 비율은 `results.json`에 있습니다.

# SW_0007 — 동료 분류법의 전경 선택과 기존 spike grouping 결합

Status: fixed-split validation complete; no core training or production-code change. No all-metric gain.

## 쉽게 설명하면

동료의 연결 성분 방식은 물체가 **어디 있는지** 더 잘 찾았지만, 한 물체를
너무 많은 조각으로 나눴습니다. 이번에는 동료 방식으로 배경을 먼저
제외하고, 남은 patch만 기존의 spike 시간 패턴으로 3/5/7개 그룹으로
다시 묶습니다. 이렇게 하면 IoU의 이득을 유지하면서 FG-ARI도 개선되는지
검증할 수 있습니다. 정답 물체 개수는 사용하지 않습니다.

## 정확한 조건

- 모델: 기존 SW_0001과 같은 seed-0 40-epoch checkpoint. 재학습 없음.
- 실제 spike 256 step 중 처음 64 step 제외. 동료 `patch_v2:5a29422`의
  centered signed correlation 연결 성분으로 배경을 정함.
- 연결 임계값 0.80 또는 0.95. 가장 큰 성분을 배경으로 제외.
- 예측 전경 patch끼리 absolute centered spike correlation과 normalized
  spectral clustering으로 고정 k=3/5/7 그룹 생성. k는 정답과 무관.
- CLEVR HDF5 검증 ID 1320–1639, 16×16 patch GT, 동일한 세 평가 지표.
- 실험 구현은 `evaluate.py`뿐. 정답은 점수 계산에만 사용하고 분류에는
  사용하지 않음. 모든 후보 선택은 검증에서만 수행.

| 같은 검증 이미지 320장 | FG-ARI ↑ | 전경 IoU ↑ | 물체별 IoU ↑ |
|---|---:|---:|---:|
| 기존 SW_0001 | 0.1363 | 0.2720 | 0.1456 |
| 동료 SW_0006, 연결 성분만 | 0.0772 | 0.3287 | 0.1736 |
| 이번 결합, 임계값 0.95와 k=5 | 0.0810 | 0.3287 | 0.1341 |

**해석:** 동료 방식이 잡은 전경은 그대로라 전경 IoU 0.3287을 유지했습니다.
그러나 spike 패턴으로 다시 묶어도 FG-ARI는 기존 0.1363에 못 미쳤고,
물체별 IoU도 동료 방식의 0.1736에서 0.1341로 내려갔습니다. 임계값
0.80/0.95와 그룹 수 3/5/7의 전체 결과는 `results.json`에 있습니다.
따라서 이 조합은 채택하지 않습니다. **단순히 조각 수만 줄이는 것으로
물체별 그룹 품질이 회복되지는 않았습니다.** 다음 후보는 spike가 만들어지는
학습 신호 자체를 점검해야 합니다.

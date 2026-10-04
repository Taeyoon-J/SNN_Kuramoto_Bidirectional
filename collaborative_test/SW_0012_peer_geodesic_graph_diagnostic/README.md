# SW_0012 — 동료의 이미지 경로 기반 graph 아이디어 검토

Status: read-only diagnostic complete; no model or production-code change. No evidence to merge this graph prior yet.

## 쉽게 설명하면

동료 브랜치 `patch_v2:0dd2115`에는 두 patch 사이 거리를 직선으로 재는
대신, 이미지 속 비슷한 patch들을 따라 이동할 때의 거리로 재는 graph
후보가 올라왔습니다. 같은 색의 다른 물체가 가까이 있어도 그 사이의
배경을 건너야 하므로 연결이 약해질 수 있다는 아이디어입니다.

우리 학습된 graph에 이 거리 계산만 일시적으로 적용하고, 검증 이미지
16장에서 같은 물체 patch의 연결과 다른 물체 patch의 연결이 실제로
더 잘 벌어지는지 봅니다. 정답 mask는 이 **진단 수치**에만 쓰고 모델
학습·예측에는 넣지 않습니다. 아직 우리 core에 병합하지 않습니다.

## 정확한 조건

- 동료 코드의 soft min-plus graph distance: 3단계, 이웃 반경 1.5,
  contrast 2.0, temperature 0.5, cap 16.0.
- SW_0004 seed-0 checkpoint의 같은 graph projection/temperature/top-k
  가중치를 사용. 거리 prior만 기존 Euclidean과 비교.
- CLEVR HDF5 검증 ID 1320–1335의 16장, 16×16 patch 정답 mask.
- 결과는 same-object / different-object / foreground-background edge
  평균과 연결 비율, 계산된 거리의 범위를 기록.
- 이 graph-only 진단은 최종 spike/membrane mask 점수가 아니며, 좋게
  나와도 goal 성공으로 주장하지 않음. 학습 중인 SW_0011에는 영향 없음.

| 같은 검증 16장, 기존 가중치 | 같은 물체 edge 평균 ↑ | 다른 물체 edge 평균 ↓ | 전경-배경 edge 평균 ↓ | 같은/다른 물체 비율 ↑ |
|---|---:|---:|---:|---:|
| 기존 직선 거리 | 0.1881 | 0.0125 | 0.0132 | 15.00 |
| 동료 geodesic 거리로 일시 교체 | 0.1928 | 0.0142 | 0.0155 | 13.59 |

**해석:** 같은 물체 내부 연결은 조금 강해졌지만, 다른 물체 및 배경과의
연결도 더 강해졌습니다. 같은 물체 대 다른 물체의 구분 비율은 오히려
15.00에서 13.59로 떨어졌습니다. 계산 거리의 6.27%가 음수이고 최솟값은
-1.00입니다. 이는 soft-min 근사에서 생긴 것으로 보이므로 경로 거리로
그대로 해석하기 어렵습니다. 기존 checkpoint는 geodesic 방식으로
학습한 것이 아니고 16장만 진단했으므로, 동료 아이디어 전체가 실패했다는
뜻은 아닙니다. 다만 현재 증거로는 우리 core에 곧바로 병합하지 않습니다.
정확한 edge 수치와 설정은 `results.json`에 있습니다.

동료의 `PV2_0001` 3-seed 평균(0.4457 / 0.4115 / 0.3438)은 다른
CLEVR render·정답 분할의 점수라서 우리 SW 실험이나 Slot reference와
직접 비교하지 않았습니다. 동료 보고서에서 spike rate가 seed별로
0.043/0.224/0.324로 크게 달랐다는 점은 향후 재현성 검증에 참고합니다.

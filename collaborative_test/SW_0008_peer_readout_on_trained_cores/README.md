# SW_0008 — 동료 분류법을 다른 학습 core에도 적용

Status: fixed-split validation complete. SW_0004 plus peer readout improves all three metrics over SW_0001 on seed-0 validation, but is not a three-seed or reference-test result.

## 쉽게 설명하면

SW_0006에서는 동료의 spike 분류법을 기존 core 한 개에만 적용했습니다.
이번에는 같은 분류법을 학습 신호가 달랐던 두 core(SW_0003, SW_0004)에도
적용합니다. **학습 방법과 분류법의 조합**이 세 점수를 함께 올리는지
확인하는 것입니다.

## 정확한 조건

- 동료 출처 `patch_v2:5a29422`; SW_0006의 `evaluate.py`를 변경 없이 재사용.
- SW_0003: membrane PLV, seed 0, 40 epoch checkpoint.
- SW_0004: phase PLV + graph-to-membrane synchrony weight 0.1, seed 0,
  10 epoch pilot checkpoint. 다른 학습 길이이므로 원인 비교는 제한적.
- 각 core의 실제 spike 256 step 중 처음 64 step 제외. signed correlation
  임계값 0.50/0.70/0.80/0.90/0.95, 성분별 곱도 같이 비교.
- CLEVR HDF5 검증 ID 1320–1639, 16×16 patch, 공통 세 지표로 평가.
- 분류에 정답 mask나 물체 개수는 사용하지 않고, test split으로 설정을
  고르지 않음. 서버 실행 명령은 `run.sh`.

| 같은 검증 이미지 320장 | FG-ARI ↑ | 전경 IoU ↑ | 물체별 IoU ↑ |
|---|---:|---:|---:|
| SW_0001 기존 spike spectral | 0.1363 | 0.2720 | 0.1456 |
| SW_0006 동료 분류법 + 기존 core | 0.0772 | 0.3287 | 0.1736 |
| 동료 분류법 + SW_0003 core, 성분별 곱 0.95 | 0.0780 | 0.0940 | 0.0395 |
| 동료 분류법 + SW_0004 core, 성분별 곱 0.50 | **0.1953** | **0.2757** | **0.2388** |

**해석:** SW_0003의 막전위 PLV core와 동료 분류법은 잘 맞지 않았습니다.
반면 graph 힌트를 membrane 학습에 사용한 SW_0004 core에서, 4개 spike
성분의 동기화를 각각 비교해 곱한 뒤 0.50 이상인 patch를 연결하니 세
지표가 기존 SW_0001보다 모두 높아졌습니다. 특히 물체별 IoU가
0.1456에서 0.2388로 올랐습니다. 이는 **우리 학습 신호와 동료 분류법을
결합한 실제 개선 후보**입니다. 단, 검증 seed 0·10 epoch에 한정됩니다.

이 설정은 예측 전경 비율 0.5631, 이미지당 평균 예측 물체 약 30.99개로
물체를 너무 많이 쪼갤 우려가 있습니다. Slot Attention의 저장된 참고
점수(다른 고정 reference split)는 0.8901 / 0.2123 / 0.2355입니다.
검증과 reference 점수를 서로 직접 비교하거나 목표 달성으로 주장하지
않습니다. 다음 단계는 이 읽기 설정을 **고정**하고 reference test와
서로 다른 3 seed를 별도로 확인하는 것입니다.

각 threshold·mode의 전체 검증 결과는 `sw0003_results.json`,
`sw0004_results.json`에 있습니다. 첫 실행에서 PyTorch의 임시 파일 읽기
오류가 났고, 동일 실험 폴더 아래로 `TMPDIR`와 `TRITON_CACHE_DIR`를
설정한 뒤 1장 smoke test 및 전체 320장 평가가 완료됐습니다. 오류가
난 첫 실행 결과는 사용하지 않았습니다.

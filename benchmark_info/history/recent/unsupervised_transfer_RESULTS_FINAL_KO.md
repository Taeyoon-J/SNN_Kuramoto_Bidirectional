# unsupervised_transfer/RESULTS_FINAL_KO.md

<details>
<summary>RESULTS_FINAL_KO.md</summary>

Source SHA256: `529ea525e2fe3c22830077b46fe354d70092fdf4288270be51b56f4b76014e74`

````markdown
# SW0253 최종 결과: 비지도 SW0165의 두 데이터셋 전이 평가

두 데이터셋 각각 TEST1,000장(ID 7000–7999), seed 0·1·2의 추가 학습 없는 평가를
완료했다. 기존 TEST360 예측을 보존하고 나머지640장을 추가했다.

| 데이터셋 | Pixel FG-ARI | Foreground IoU | Matched Object IoU |
|---|---:|---:|---:|
| Multi-dSprites colored_on_colored | **0.587447** | **0.692986** | **0.649447** |
| Tetrominoes | **0.810799** | **0.603889** | **0.801700** |

| 데이터셋 | seed | FG-ARI | Foreground IoU | Object IoU | 평균 전경 그룹 수 |
|---|---:|---:|---:|---:|---:|
| Multi-dSprites | 0 | 0.579763 | 0.758025 | 0.682053 | 3.624 |
| Multi-dSprites | 1 | 0.594597 | 0.626254 | 0.641711 | 2.925 |
| Multi-dSprites | 2 | 0.587979 | 0.694680 | 0.624576 | 3.203 |
| Tetrominoes | 0 | 0.789091 | 0.707433 | 0.793876 | 9.276 |
| Tetrominoes | 1 | 0.835243 | 0.524920 | 0.812300 | 8.075 |
| Tetrominoes | 2 | 0.808064 | 0.579314 | 0.798925 | 8.736 |

## 무엇을 평가했는가

원래 CLEVR native32 고정320 평가의 비지도 최고 SW0165를 그대로 전이했다.
원래 source checkpoint,224-parameter reconstruction encoder, feature 통계,
SW0135 strict16→32 mapping, SW0143 T 연산자 및 SW0165 frozen signed partition을
유지했다. 최대component를 배경으로 삼는 원래 support 규칙도 유지했다.
252번의 마스크 지도학습 head를 사용하거나 배경 삭제 규칙을 새로 바꾼 실험이 아니다.

해당 데이터셋의 optimizer update와 마스크 지도학습은 모두0회다. Source core의
원래 학습 manifest/audit에서도 정답 mask를 학습에 사용하지 않았음을 확인했다.
이 실험은 새로운 native32 70,000장 학습이 아니며, 기존 checkpoint의 전이 평가다.

## 해상도와 평가 조건

- 모델 패치: native32×32 그대로 유지.
- 입력: 원래 benchmark의 nearest128 RGB, 원래 CLEVR encoder 및 정규화 통계.
- 평가: native32 예측을 보호된 `cell_index`로 Multi-dSprites64×64,
  Tetrominoes35×35의 원본 픽셀에 확장. 정답 픽셀 mask는 변경하지 않음.
- 데이터: 올바른 `multi_dsprites` 계열의 colored_on_colored만 사용.
  잘못된 초기 `mdsprites` 파일은 사용하지 않음.
- 단일 물체 이미지: Multi-dSprites230/1000, Tetrominoes0/1000. 모두 원래 규약대로 포함.
- 여섯 조건 모두 유효 FG-ARI1,000개. 배경만 정답 전경 계산에서 제외하며
  단일 물체 이미지를 임의로 제외하거나 일괄0점으로 처리하지 않음.

같은 TEST1,000장의 정답을 보호된 modal grid32 builder로 변환·확장한
FG-ARI 참조값은 Multi-dSprites **0.613455**, Tetrominoes **0.938016**이다.
이는 benchmark의 양자화 참조값이며 임의의 영역 배정에 대한 수학적 최대값을
증명한 것은 아니다. Multi-dSprites 모델 자체를64×64 패치로 평가한 것으로
표기하면 안 된다. Native32 해상도 제약을 고려해야 한다.

## 검증

실행 전 세 seed 모두 원래 CLEVR 첫 mask를 정확히 재현했고 두 새 데이터셋의
TRAIN 이미지에서 실제 CUDA forward를 통과했다. 평가 도중 원래 모델은 변경하지 않았다.

최종 독립 audit에서 다음을 확인했다.

- 6,000개 예측의 pixel FG-ARI·Foreground IoU·Hungarian Object IoU 독립 재계산;
  원래 점수와 최대 오차0.
- 원래360장과 전체1,000장 예측 prefix 동일성.
- 모든 입력 RGB, 각 예측 row, 최종 예측·protocol 및 checkpoint SHA.
- 원래 source 구현 의존성25개와 source checkpoint·encoder 통계 불변성.
- 두 데이터셋×세 seed의 실제 frozen 모델로 마지막 TEST 이미지7999 예측을 재현;
  여섯 건 모두 정확히 일치, 모델 state digest 유지.
- 원래 benchmark 보호 파일 및 정답 픽셀 mask 유지.

실행 증거: 비공개 SW0251_background_mask 기록의 `253_results_1000.json`,
`253_audit_1000.json`, `SW0253_launch.json`; 서버 `SW0253_unsupervised_mod`.
초기360장 결과는 [RESULTS_360_KO.md](RESULTS_360_KO.md)에 보존했다.

이 비지도 전이 평가 요청은 완료됐으며, 별도252번의 전체1,000장 평가와 독립 검증도 완료했다.
252번 결과는 [해당 최종 보고서](../neutral_readout/RESULTS_FINAL_KO.md)에 기록했다.

````

</details>

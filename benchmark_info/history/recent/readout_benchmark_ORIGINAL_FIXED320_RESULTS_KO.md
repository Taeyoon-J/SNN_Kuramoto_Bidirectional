# readout_benchmark/ORIGINAL_FIXED320_RESULTS_KO.md

<details>
<summary>ORIGINAL_FIXED320_RESULTS_KO.md</summary>

Source SHA256: `cc82f0dbb8e4d26062d67e19f4bfbbebce0617dc5f68da9b5282c377eec54657`

````markdown
# SW0251: 기존 240 성능 유지 검증 통과

64×64 패치, 기존에 사용한 고정 평가 320장(ID 1320–1639), seed 0·1·2 평균이다.
이 평가 집합은 재사용한 validation이며, 새로운 미사용 test 결과가 아니다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원본 240 | 0.889187 | 0.754775 | 0.686438 |
| 최종 mask 감독 residual readout | 0.889629 | 0.784008 | 0.707530 |
| 차이 | +0.000442 | +0.029233 | +0.021092 |

사전 조건인 평균 FG-ARI 절대 감소 0.005 이내를 통과했다. FG-ARI 차이의
paired-image 95% CI는 [−0.000673, +0.001409]이므로 확실한 FG-ARI 향상은
주장하지 않는다. Foreground IoU 차이 CI는 [+0.022518, +0.036138],
Object IoU 차이 CI는 [+0.016573, +0.026087]이다.

| seed | 원본 FG-ARI | 수정 FG-ARI |
|---|---:|---:|
| 0 | 0.891040 | 0.893264 |
| 1 | 0.897328 | 0.896829 |
| 2 | 0.879194 | 0.878795 |

후보 선택에는 별도 TRAIN256/VAL64만 사용했다. frozen 240의 실제 spike·membrane
활동을 입력으로 받는 readout head에 Hungarian mask CE·Dice 및 보조 loss를 적용했다.
encoder·core의 end-to-end mask 학습 결과가 아니다. 기존 VAL64의 기전 분석에서는
물체 간 재배정보다 전경/배경 support 변경이 개선의 주된 원인이었다.

원시 예측 1,920건(원본·수정 각각 320×3)을 독립 NumPy contingency ARI 및
Hungarian IoU 계산으로 재검증했다. 원본 mask는 기존 생산 예측과 일치하고,
240 checkpoint 3개 SHA와 보호된 benchmark 파일의 불변성을 확인했다.

증거는 비공개 실행 기록 SW0251_background_mask의
`collected/original_confirmation/result.json` 및
`original_gate_independent_audit.json`에 있다.

Multi-dSprites grid64와 Tetrominoes grid32의 세 seed zero-shot/추가학습 평가는
후속 단계이다. 앞선 seed0 TEST8 미리보기에서는 Multi-dSprites FG-ARI가 하락했으므로,
이 기존 데이터 검증만으로 배경 문제가 데이터셋 전반에서 해결됐다고 주장하지 않는다.

````

</details>

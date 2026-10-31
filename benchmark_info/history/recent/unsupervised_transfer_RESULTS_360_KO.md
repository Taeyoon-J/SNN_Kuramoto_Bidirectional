# unsupervised_transfer/RESULTS_360_KO.md

<details>
<summary>RESULTS_360_KO.md</summary>

Source SHA256: `e2bc1b45974b407903dff0e5f528d47de7d0c67d6822a2894fab74578fd0c04c`

````markdown
# SW0253: 비지도 최고 SW0165의 두 데이터셋 전이 평가

각 데이터셋 TEST360(ID 7000–7359), seed 0·1·2 평균이다.
원래 165번 checkpoint·224-parameter reconstruction encoder·전처리·readout을
그대로 사용했다. 해당 데이터셋 추가 학습과 마스크 지도학습은 모두 0회이다.
240·251·252의 지도학습 encoder/head를 사용하지 않았다.

| 데이터셋 | Pixel FG-ARI | Foreground IoU | Matched Object IoU |
|---|---:|---:|---:|
| Multi-dSprites colored_on_colored | **0.586447** | **0.686493** | **0.647252** |
| Tetrominoes | **0.806953** | **0.601668** | **0.801354** |

| 데이터셋 | seed | FG-ARI | Foreground IoU | Object IoU | 평균 전경 그룹 수 |
|---|---:|---:|---:|---:|---:|
| Multi-dSprites | 0 | 0.580298 | 0.749152 | 0.676752 | 3.3639 |
| Multi-dSprites | 1 | 0.595926 | 0.617432 | 0.639676 | 2.8222 |
| Multi-dSprites | 2 | 0.583117 | 0.692895 | 0.625329 | 3.1833 |
| Tetrominoes | 0 | 0.786525 | 0.708147 | 0.795048 | 9.5722 |
| Tetrominoes | 1 | 0.833262 | 0.523188 | 0.813553 | 7.9028 |
| Tetrominoes | 2 | 0.801071 | 0.573668 | 0.795459 | 8.8083 |

## 해상도 및 평가 규약

원래 모델은 native32×32 패치 구조다. 구조를 변경하지 않고 출력32×32를 보호된
`cell_index`로 원래 Multi-dSprites64×64 / Tetrominoes35×35 픽셀에 확장했다.
정답 픽셀 mask는 변경하지 않았다. Multi-dSprites 모델 자체를64×64 패치로
변경한 실험은 아니다.

같은360장 정답을 보호된 modal grid32 builder로 변환하고 확장한 기준FG-ARI는
Multi-dSprites **0.604509**, Tetrominoes **0.936241**이다. 이는benchmark에서
정의한 양자화 참조값이며, 임의의 영역 배정에 대한 수학적 최대FG-ARI를
증명한 값은 아니다. Multi-dSprites 결과에는native32 해상도 제약을 고려해야 한다.

원래SW0165의 최대component를 배경으로 삼는support 규칙과signed foreground partition을
유지했다. 252번의 배경 역할 비고정head로 교체한 결과가 아니다. 따라서 이 표의
Foreground IoU는 원래benchmark의 통상적인 전경 판정을 측정하며, 모든 영역을
양수ID로 남기는252번 출력의raw Foreground IoU와 의미가 다르다.

## 비지도 출처 및 검증

- SW0097 세 source checkpoint의 hash와 원래 학습 manifest/audit를 검증했다.
  core 학습의 `ground_truth_used_for_training`은 false이다.
- 초기 16-grid source를 원래 SW0135 strict mapping으로 native32에 옮기고,
  SW0143 T 연산자 및 SW0165 frozen readout을 그대로 사용했다.
  이 실험에서 새로운 native32 70,000장 학습은 하지 않았다.
- 실행 전 세 seed 모두 원래 CLEVR 첫 mask를 정확히 재현하고, 두 새 데이터셋의
  TRAIN 이미지로 실제 CUDA forward를 통과했다. 원래 가중치는 변경되지 않았다.
- 여섯 조건의 예측을 고정한 뒤에만 평가 정답을 열었다. Multi-dSprites 단일 물체
  86장은 모두 포함했고, Tetrominoes 단일 물체 이미지는 0장이다.
- 2,160개 pixel 지표를 독립 NumPy contingency ARI/Hungarian IoU로 재계산했다.
  원래 기록과 차이는 0이며, 여섯 실제 모델 예측 재현도 통과했다.
- 25개 원래 구현 의존성, checkpoint/encoder 통계, 각 입력 RGB 및 예측 SHA,
  보호된 benchmark 불변성을 확인했다.

후속1,000장 평가와 최종 독립 audit도 완료했다. 최종 결과는
[RESULTS_FINAL_KO.md](RESULTS_FINAL_KO.md)에 기록했다. 별도252번의1,000장 평가와 독립 검증도 완료했다.

실행 증거: 비공개 SW0251_background_mask 기록의 `253_results_360.json`,
`253_audit_360.json`, `SW0253_launch.json`; 서버 `SW0253_unsupervised_mod`.

````

</details>

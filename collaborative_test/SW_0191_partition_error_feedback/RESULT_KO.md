# SW0191 완료: partition-error feedback recipe 미채택

고정320장, native32 직접 예측, seed0/1/2 평균, standalone 기준:

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0191 QCE + actual-partition feedback | .799773 | .478037 | .659912 |
| SW0186 기존 최고 | .834681 | .569040 | .689802 |
| SW0188 QCE 추가 학습 | .831401 | .559170 | .681908 |
| comparable Slot70k | .781652 | .204360 | .203165 |

SW0191 seed별 점수:

| seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 0 | .796667 | .504761 | .652858 |
| 1 | .815104 | .498972 | .688736 |
| 2 | .787547 | .430379 | .638141 |

SW0186 대비 차이와 paired-image bootstrap95%CI (세 seed를 이미지별 평균 후 재표집):

- FG-ARI: -.034908, CI[-.041982,-.027848]
- Foreground IoU: -.091002, CI[-.104698,-.077117]
- Object IoU: -.029891, CI[-.036441,-.023436]

Conditional endpoint 평균은 .778284 / .609975 / .643924다. 새 loss가 실제
emitted spike와 trainable parameter에 gradient를 전달하고 CNN/Adam을 갱신한
것은 세 seed에서 검증했다. 그러나 이 구현과 weight1 추가 학습은 기존 최고보다
세 지표를 악화시켰다. Gradient 연결 증명이 일반화 성능이나 gating 인과적 기여를
증명하지는 않는다. 원인을 이 결과만으로 특정하지 않는다.

모든 seed는 같은 seed의 완료된 SW0186 전체 weights에서 시작했고 Adam은
다섯 family 모두 초기화했다. TRAIN512를 두 번 노출한64 proposals 중49/50/38을
엄격한 같은-RNG full-forward loss 감소 기준으로 수용했다. 추가 TRAIN512 예산은
기존70k foundation 학습 이력과 별개인 supervised native32 adaptation이다.
새 native32 joint70k 학습이나 unsupervised 성과로 해석하지 않는다.

세 seed 두 endpoint 예측을 전부 동결한 후 GT를 열었다. 독립 검증은 실제
후보1920 endpoint/image rows와 Slot960 rows를 다시 계산하고 means/CI/gates 및
학습 manifest를 확인했다. 원본 이미지별 점수는 비공개로 보관하고 집계와 검증만
공유한다. 정확한 검증 오차와 해시는 score_v1/completion_audit.json을 따른다.

최종 score SHA256:
`dec5e299a64f09949a32e40f3f0909d2f432afde0f30ffd71392871360c2a8b7`

독립 audit SHA256:
`4d8276c56a8cd174f746981e2d69f4bf4b40166e79b12ccfa9648e5ac349def6`

이 정확한 recipe는 종료하고 작은 weight/LR/margin 재탐색을 하지 않는다.
현재 최고 모델은 SW0186으로 유지한다. 사용자의 새64x64 FG-ARI>=.90 방향에
따라 SW0192는 SW0186 encoder/graph/gates/QCE를 유지한 native64 추가 학습의
메모리·시간·gradient 검증부터 진행한다.

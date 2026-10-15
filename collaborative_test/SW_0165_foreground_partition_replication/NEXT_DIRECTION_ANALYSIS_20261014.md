# SW0164 이후의 제한된 판별 진단

2026-10-14 prospective analysis. 이 문서는 SW0165 재현 결과를 선택하거나 변경하지 않으며, SW0164/165의 저장된 예측을 모두 고정한 뒤 한 번 수행할 GT-last 진단만 등록한다.

## 현재 증거

SW0164 seed1 foreground-induced 결과는 `.770932/.614993/.630523`, SourceT는 `.755273/.614993/.579514`였다. SourceT 대비 FG-ARI 차이는 `+.015659`, paired 95% CI `[.009116,.022883]`; object IoU 차이는 `+.051010`, CI `[.041976,.060664]`이다. Foreground IoU와 foreground membership은 이미지마다 SourceT와 정확히 같다. 따라서 SW0162의 foreground 손실 없이도 signed relation이 instance partition을 개선할 수 있다는 점은 입증됐다.

그러나 Slot seed1 FG-ARI보다 `-.070887`, 95% CI `[-.094692,-.046903]`이고 `.87`보다 `-.099068` 낮다. 320장 중 200장이 FG-ARI `.87` 미만이다. SW0164와 SW0162의 이미지별 SourceT 대비 FG-ARI 변화는 상관 `r=.7994`이므로 두 readout은 대체로 같은 쉬운 장면에서 개선된다. 고정 support가 SW0162보다 FG IoU를 `+.016608` 회복했지만 FG-ARI는 오히려 `-.004453`이므로, support 보존만으로 남은 grouping 격차가 해결되지는 않는다.

다만 현재 score만으로 남은 격차를 foreground 누락 탓 또는 covered-foreground 내부 merge/split 탓이라고 단정할 수 없다. production FG-ARI는 GT foreground만 평가하지만 그 안의 predicted-background patch를 하나의 cluster로 남긴다. SourceT의 false-negative foreground는 여러 GT object를 background cluster로 합쳐 FG-ARI 상한을 낮출 수 있다.

## SW0165 뒤 한 번만 수행할 GT-last 진단

SW0164와, 재현에 성공하면 SW0165 seed0/2의 **이미 저장되고 hash가 고정된 prediction**에 대해서만 canonical modal32 GT를 연다. 학습, 재예측, coefficient/threshold 선택은 하지 않는다. 각 이미지와 seed에 다음을 기록한다.

1. `fixed-support GT-induced FG-ARI upper bound`: SourceT foreground 안의 patch에는 GT object ID를 주고, SourceT background 안의 GT-foreground patch에는 모두 label 0을 준다. [`FIXED_SUPPORT_ARI_BOUND_20261014.md`](FIXED_SUPPORT_ARI_BOUND_20261014.md)의 증명처럼, unlimited/disconnected positive group을 허용한 relaxation에서의 인증 최대값이다. SW0164 실제 solver는 4-connectivity를 요구하므로 이 값은 실제 solver의 인증 상한이기도 하다. 실제 solver가 disconnected group을 허용한다는 뜻은 아니다.
2. `support-idealized attachment FG-ARI`: candidate의 covered-patch group은 그대로 두고, missed GT-foreground patch를 같은 GT object와 overlap이 가장 큰 기존 group에 붙인다. overlap이 없으면 그 GT object용 새 label을 준다. 이것도 GT-assisted reference이며 실제 predictor 성능이나 support만의 인과 효과로 해석하지 않는다.
3. GT-foreground patch를 `covered`와 `missed`로 나누고, missed patch 수와 missed GT-object 수를 기록한다.
4. covered patch 쌍에서 (a) 같은 GT object인데 서로 다른 candidate group인 split pair와 (b) 다른 GT object인데 같은 candidate group인 merge pair를 각각 전체 가능한 same/different-object pair로 정규화한다. predicted-background 안의 GT-foreground 쌍은 별도 `background-collapse merge`로 기록한다.
5. seed별 평균과 320-image paired bootstrap CI를 보고한다. per-image 상관은 설명용이며 원인 추론이나 선택에 쓰지 않는다.

판정은 하나다. 세 seed에서 GT-induced reference가 각 seed의 Slot FG-ARI와 `.87` 중 더 높은 기준에 못 미치고 forced background-collapse/cross-support pair 오류가 남은 deficit의 대부분이면, 다음 모델 시험은 foreground estimator를 함께 바꾸되 SW0164 partition을 고정한 단일 실험이어야 한다. Reference가 기준을 넘고 covered-patch split/merge가 deficit의 대부분이면 foreground 변경은 보류하고, 둘 중 더 큰 정규화 오류 하나를 줄이는 learned assignment 설계만 주요 검토 대상으로 올린다. 어느 쪽도 지배적이지 않으면 새 모델 시험을 열지 않고 이 이분법이 불충분하다고 보고한다. 어느 경우에도 이 진단값으로 threshold, group count, merge weight를 고르지 않는다.

SW0165가 FG-ARI 개선을 세 seed에서 재현하지 못하면 이 readout의 새 학습 실험은 열지 않고 종료한다. 이 진단은 seed1의 성공 이유를 설명하는 사후 분석으로만 남긴다.

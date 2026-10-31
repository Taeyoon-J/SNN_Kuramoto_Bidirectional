# SW_0195_native64_generalization_gap

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `768ff82ea73de7fc461dc8ddfbfb427fa1162d3d088088b1bcc5663bc560ab17`

````markdown
# SW0195: native64 학습/검증 변화 분리

상태: 진단 코드 준비 중. 학습 또는 GPU 진단을 실행했다는 뜻이 아니다.

SW0194를 변경하지 않는다. 완료된 TRAIN8 및 TRAIN64 모델의 같은 production
spike/Q readout을 사용한다. 추가 학습, loss 변경, threshold 탐색은 하지 않는다.

TRAIN64의 각 B4 proposal에서 첫 이미지 하나씩, 인덱스 0,4,...,60 총16장을
미리 선택한다. 두 모델에서 같은 이미지와 RNG(195000+offset)를 사용한다.
TRAIN8 모델에는 이 중 두 장만 native64 추가 학습에 사용되었다. 따라서 이 비교는
전체 모델의 최초 학습 전후 비교가 아니라 56장 추가 adaptation 전후 비교이다.
기존 SW0194의 32장 development 결과를 재사용한다. 두 cohort의 image 난이도가
같다고 가정하지 않고, 각 cohort 내부의 paired 변화 및 그 차이를 보고한다.

세 지표: FG-ARI, foreground IoU, matched object IoU. TRAIN 및 development 모두
같은 64x64 modal GT와 production mask 계산을 사용한다. 모든 TRAIN prediction을
먼저 고정한 후 GT로 점수를 계산한다. imagewise 점수/GT는 공개하지 않는다.

각 cohort에서 paired bootstrap 10,000회, 두 cohort의 resampling은 독립이다.
TRAIN의 FG 변화 CI 하한>0이면서 development 변화 CI 상한<0일 때만
이 adaptation에서 과적합과 일치하는 신호라고 표시한다. 원인 규명으로 해석하지
않는다. raw TRAIN>development 점수차나 minibatch loss 감소만으로 판정하지 않는다.
이 작은 cohort는 최종 held-out test나 세 seed 검증을 대체하지 않는다.

실행 전 요건: SW0194 완료 report/checkpoint/validation protocol 검증,
두 모델 digest 확인, CPU 검사 통과, 독점 idle GPU 배정. 진행 중인194 소스와
fingerprint는 변경하지 않는다. 단순한 대기 timeout으로 재학습하지 않는다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `6d44266eb631173fea563c2eacbfea2c8eb1e160f03ea3f3a25b0162e4a6ff6e`

````markdown
# SW0195 native64 generalization diagnostic

Read-only comparison of the completed TRAIN8 and TRAIN64 models on16 fixed TRAIN
images, reusing and independently rechecking the32 development predictions.
No optimizer updates, architecture changes, or threshold tuning.

Completed and independently rescored actual TRAIN16 and development32 masks.
The source and final model hashes, cohort bindings, paired means and confidence
intervals were checked; maximum production/independent metric error2.22e-16.
No optimizer updates occurred during this diagnostic.

| Cohort | TRAIN8 FG-ARI / FIoU / object IoU | TRAIN64 FG-ARI / FIoU / object IoU |
| --- | --- | --- |
| TRAIN16 | .915137 / .838085 / .752900 | .913519 / .827964 / .738010 |
| Development32 | .918586 / .832867 / .733686 | .900166 / .830834 / .731658 |

Both mean FG scores decreased. TRAIN delta95%CI[-.005617,.001131]; development
delta95%CI[-.047440,.001273]. The registered positive-TRAIN/negative-development
signal was false. This does not prove overfitting absent; neither FG change is
resolved from zero, and only seed0/small cohorts were measured. It also does not
establish that a fixed computation structure caused the changes. This unchanged
QCE continuation showed no gain and is not selected as an improved recipe.

Public audit and aggregate summary: results_archive/score_v1/completion_audit.json
and summary.json. Actual imagewise vectors remain local and ignored.

Known TRAIN masks are already present in the model factory. Prediction does not
use them; it only freezes the same production spike/Q masks before scoring.
Only aggregate results may be shared; imagewise GT-derived scores remain private.

See DESIGN_KO.md for fixed cohorts, uncertainty estimates, and interpretation.

````

</details>

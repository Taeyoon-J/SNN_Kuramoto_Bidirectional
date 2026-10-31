# SW_0215_relation_loss_decision_alignment

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `8f83254d572149559669244a916f4d8515dfa323ff5a013ced1c40a61fd65839`

````markdown
# SW0215: TRAIN 통계로 관계 loss와 merge 판정 기준을 일치

214 primary .720607은 원래 .788414보다 낮았다. Frozen geometry control의
.814008은 FG CI가0을 포함하고 ObjectIoU가 낮아 확정 개선으로 보지 않는다.
이후 점검에서214 loss는 positive/negative soft mass를 각각 normalize하지만
추론은 raw logit>0을 사용했다는 기준 차이를 확인했다.

Base image/pair weight w, soft target t에 대해 P=sum(w*t), N=sum(w*(1-t))다.
이상적 weighted BCE 해는 원래 target probability p의 logit에 log(N/P)를
더한다. 따라서 log(N/P)를 raw logit에서 빼고 기존 strict>0 complete-link
판정을 적용한다. 214의 초기 기준은 원래 p=.5와 일치하지 않았다.
이상적 loss 해의 보정이며 유한 NN의 확률 calibration을 보장하지 않는다.

값은 원래 TRAIN32 pair table에서만 계산한다. DEV score를 이용한 threshold
sweep이나 재학습은 없다. Frozen214 primary/geometry control을 CPU로 먼저
bitwise replay해야 새 예측을 평가한다. 모든 보정 mask를 고정한 뒤 GT를 읽고
원래 core, uncorrected214, corrected215 세 지표와 CI를 비교한다.

214 실패 판정을 유지하는 별도 posthoc-motivated 비교다. Core forward0,
optimizer update0, native64/고정16/seed2. Foreground와 원래 component 내부는
보존한다. 전체320/세 seed .90, 새70k 또는 모든 gating 인과 증거가 아니다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `aa292123e35b749f6520a56adef30cf04e7dc9aa7bc37d46a2f200bfed8b7467`

````markdown
# SW0215 TRAIN-derived loss/decision alignment

Four local/four Linux tests passed. Frozen214 CPU replay was exact before new
mask scoring. No optimization or core forward. TRAIN-only log-odds shift3.263823
corrects the ideal positive/negative normalized BCE objective's decision scale.

Corrected primary .781651/.445474/.664934 versus original
.788414/.445474/.674335: no improvement established. Corrected geometry merges
nothing and equals the original. No DEV threshold sweep. See RESULT_KO.md and
the completed_v1/ 80-endpoint independent audit. Full320 .848615 mean is unchanged.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `3fb40b290cfb5a8a9ed983b1c981d7d0856fd951e15cb3e3495aba7613fa01a6`

````markdown
# SW0215: loss/판정 보정은 하락을 완화했지만 baseline 개선 없음

Native64·고정16·seed2, 기존214 head를 재학습 없이 사용했다. Local4/Linux4
테스트 후 CPU에서214의 두 원래 endpoint를 모두 bitwise 재현하고 새 mask를
고정한 뒤 GT를 읽었다. 모든80 mask endpoint, 세 지표·paired CI와 source
component/전경 보존을 독립 재검증했다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 core | .788414 | .445474 | .674335 |
| 214 raw primary | .720607 | .445474 | .580461 |
| **보정 primary** | **.781651** | **.445474** | **.664934** |
| 214 raw frozen geometry | .814008 | .445474 | .632410 |
| 보정 frozen geometry | .788414 | .445474 | .674335 |

TRAIN의 image-balanced positive mass1.178666, negative mass30.821334,
positive prior .0368333에서 log(N/P)=3.263823475를 계산해 raw logit에서 뺐다.
Validation 점수를 이용해 threshold를 선택하지 않았다. 이상적 weighted BCE
해의 log-odds shift를 맞추는 것으로, 유한 classifier 확률의 calibration을
보장하지 않는다.

Primary는 raw214보다 나아졌지만 원래 core 대비 FG 차이-.006763,
95% CI[-.018548,+.000049]로 개선은 확인되지 않았다. 평균 merge는9.3125에서
1.9375로 줄었다. 보정 geometry control은 merge0으로 원래 mask와 같다.
.8140 결과를 유지하기 위해 임의의 중간 threshold를 찾지 않는다.

다음에 점검할 중요한 target 차이도 있다.214의 object histogram은 component
전체 patch 수(정답 background 포함)로 나누지만 FG-ARI는 정답 foreground
patch에서 계산한다. 따라서 background가 섞인 같은 물체 component의 학습
target이 낮아질 수 있다. 후속 시험은 같은 데이터·평가 계약을 유지하면서
TRAIN 관계 target을 foreground-conditional로 정의하는 방안을 먼저 검토한다.
Anchor 순서 불변 특징은 별도 가설이며 이 결과로 원인이라고 단정하지 않는다.

새 최고 성능은 없다. 전체320·세 seed 기준점 .848615/.592533/.672047을
보존한다. Core forward0/추가 optimizer update0이며 전체70k 실험이 아니다.
증빙: results_archive/completed_v1/score_v1/summary.json,
predictions/protocol.json 및 completion_audit.json.

````

</details>

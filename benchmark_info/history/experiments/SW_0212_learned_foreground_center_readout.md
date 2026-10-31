# SW_0212_learned_foreground_center_readout

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `e6b5afc5452998964af5b0f93d65bd5d5c24ae0051cd4c4823f58aaa41c29d73`

````markdown
# SW0212: 학습된 전경과 비국소 center assignment 결합

211 결과를 아직 읽기 전에 등록하는 후속 readout 비교다. 211의 원래 spike
전경 보존 조건을 바꾸지 않는다. 별도212에서만 이미 학습된 foreground
출력을 사용한다. Core/encoder/graph/gates와211 head 모두 재학습하지 않는다.

207의 전경 제거는 공간 연결을 끊어 object split을 증가시켰다. 211은
offset으로 떨어진 patch도 같은 center에 배정한다. 따라서 이 비국소 배정과
학습 전경을 결합하면 foreground 오류를 고치면서 split을 피할 수 있는지
확인한다. 결과를 미리 성공으로 가정하지 않는다.

- 211 primary128 head 고정. 32/64 snapshot 중 선택하지 않는다.
- 동일16 DEV/seed2/native64, 같은268-channel cache만 사용. RGB나 GT count
  입력 없음. 211 학습 mask32장 외에 추가 학습/GT supervision 없음.
- foreground logit>=0 (sigmoid>=.5)로 고정한다. DEV threshold sweep 없음.
- heat/offset/center threshold .3, radius1, capacity10은211과 동일하다.
- CPU에서 frozen head를 실행한다. 모든16장의 원래 foreground를 사용한
  replay가 저장된211 primary128 mask와 bitwise 동일해야 비교를 허용한다.
  수치 차이로 replay가 다르면 점수를 내지 않고 관찰 결과를 보존한다.
- 새 foreground를 사용한 모든16 mask를 먼저 저장·검증하고 GT를 읽는다.
  원래 baseline, 211 primary128, 새212를 같은 evaluator로 비교한다.
- FG-ARI 우선/두 IoU/count 및 fallback을 모두 보고한다. CI는211과 같은
  paired16 bootstrap이다. 유용한 한 지표 개선도 보존하며, 전체320/세seed
  .90 또는 데이터 scaling/gating 인과 증거로 해석하지 않는다.
- 211 supervisor가 실제 살아 있는지 확인하며 대기한다.211 실패/종료에
  결과가 없으면 재시작하지 않는다. GPU 작업을 추가하지 않는다.

이 실험은 frozen classifier readout 비교이며 새 training test가 아니다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `d50ba7414796be087510619ef42e35145b2d183a99ee4526ce0d4b447a6abe2e`

````markdown
# SW0212 learned foreground with nonlocal center assignment

Prospective follow-up to211, registered without reading its scores. Reuses only
the frozen primary128 head. Its learned foreground (logit>=0) replaces the old
spike foreground; the center/offset decoder is unchanged. No optimizer updates,
extra GPU lease, threshold sweep or new training data.

All16 original-support CPU predictions must exactly replay211 saved masks before
new masks may be scored. GT is read only after all new predictions are frozen.
Both IoUs, FG-ARI, counts and fallback are reported, including negative results.
See DESIGN_KO.md. Registration and test receipts will be under results_archive.

Registered before211 scores existed: five local and five Linux tests passed,
plus actual268-channel native64 CPU head/decoder admission. CPU dependency
supervisor2285616/start448185348 is waiting for211; no new training has started.
The first deployment bootstrap encountered system Python3.6's unsupported
capture_output argument before tests or model execution. Using the established
research Python3.10 fixed the deployment helper; scientific source was unchanged.
Original bootstrap logs are retained locally. This is not a failed GPU trial.

Completed and independently audited: .000000/.292532/.022362 versus original
.788414/.445474/.674335. Parent128 CPU replay was bitwise exact. Center fallback
remains100%, predicted count1/image. No threshold rescue or expansion of this
endpoint; no incumbent change. See RESULT_KO.md and completed_v1 evidence.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `b52848cbef64735daec11ce3806a20f1358d9e8decccedf2f0953f95c7417419`

````markdown
# SW0212: 학습 전경 결합도 개선 실패

추가 학습 없이211 primary128 head를 사용했다. CPU에서 원래 전경을 사용한
모든16 예측이211 GPU endpoint와 정확히 같음을 확인하고, 새 mask를 모두
고정한 뒤 GT를 읽었다. Head SHA,48개 mask의 독립 metric과 paired CI를
재검증했다. seed2·고정16·native64의 supervised readout 비교다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 warm205 | .788414 | .445474 | .674335 |
| 211 원래 전경+center head | .034678 | .445474 | .044414 |
| 212 학습 전경+같은 head | .000000 | .292532 | .022362 |

추가 전경 판별도 세 지표를 개선하지 못했다. 모든 이미지에서 center
fallback과 예측1개 상태가 유지됐고 전경 변경 비율은39.68%였다.
같은 head의 TRAIN 진단에서도 center 검출이 작동하지 않았으므로 이 결과로
학습 전경과 비국소 assignment의 일반적인 가능성을 부정하지 않는다.
이128-step checkpoint에 대한 고정 .5 전경 readout의 확대는 중단하며
threshold를 validation에 맞춰 바꾸지 않는다.

새 최고 결과나 전체320 성능 개선은 없다. 기존 기준점은 유지한다.
증빙: results_archive/completed_v1/score_v1/summary.json,
predictions/protocol.json, completion_audit.json.

````

</details>

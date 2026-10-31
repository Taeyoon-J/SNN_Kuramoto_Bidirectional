# readout_benchmark/VALIDATION_RESULTS_KO.md

<details>
<summary>VALIDATION_RESULTS_KO.md</summary>

Source SHA256: `f82f4e87be91ca93eb042d0e4a9ee720e71319df6b25e348b42e674bdc2c57ca`

````markdown
# SW0251 배경 처리·최종 mask loss 통제 비교

아래는 **TRAIN256, 별도 validation64, seed0·1·2 평균**이다. 고정320 보존 검증이나 새 데이터셋 결과가 아니다. 동일한 실제 SW240 activity를 재사용하고, head마다20epoch/5120 Adam updates를 수행했다. 원래 encoder/core/graph/readout 파라미터와 source checkpoint는 보존했다. 학습한 새 head는 실제 spike·membrane의144채널 특징을 입력으로 사용한다.

| 설정 | FG-ARI | Foreground IoU | Matched Object IoU |
|---|---:|---:|---:|
| 원래240 | .881432 | .793802 | .695344 |
| 기존 큰 component도 유지 | .885387 | .198595 | .702056 |
| 전체 영역 partition | .885710 | .200268 | .732342 |
| 배경 분류만 학습 | .874705 | .751942 | .669335 |
| 직접 최종 mask loss를 학습한 residual head | **.885416** | **.815043** | **.712711** |

큰 component 유지/전체 영역 설정은 전경을 매우 넓게 예측한다. FG-ARI만 높다고 배경 인식이 좋아졌다고 해석하면 안 된다. 배경-only head는 이 validation에서 원래240 대비 FG-ARI가 .006727 낮다. 두 학습 설정 중 FG-ARI가 높은 residual_mask를 기존 사전 정의된 기준으로 선정했다.

| seed | 원래240 FG-ARI | residual_mask FG-ARI |
|---|---:|---:|
| 0 | .880402 | .882063 |
| 1 | .891204 | .901498 |
| 2 | .872689 | .872686 |

Residual head의 평균 변화는 FG-ARI +.003984, foreground IoU +.021240, object IoU +.017367이다. 직접 Hungarian mask CE·Dice, 배경 BCE, 전경 pair 및 partition anchor loss의 gradient는 **새 readout head에만** 전달한다. Encoder/core까지 end-to-end mask loss로 학습한 실험이 아니다.

## 실제 변경 원인 확인

선정된 checkpoint를 CPU에서 validation64 전체에 다시 적용해 위 점수를 재현하고, 기존 물체 prototype의 ID 대응을 보정한 뒤 변경 patch를 집계했다. 고정320 정답은 열지 않았다.

| seed | 배경에서 전경으로 | 전경에서 배경으로 | 기존 전경의 물체 prototype 변경 |
|---|---:|---:|---:|
| 0 | 1777 | 3 | 1 |
| 1 | 1539 | 0 | 0 |
| 2 | 331 | 10652 | 7 |

배경/전경 변경만 적용하고 기존 전경의 물체 assignment를 유지한 counterfactual의 FG-ARI는 .882068/.901498/.872686으로, 실제 선정 모델과 사실상 같다. 따라서 이번 개선은 주로 전경 support 보정이며, 물체 간 재분류 또는 gating의 인과적 기여가 향상됐다는 증거로 사용하지 않는다. 원본 근거는 selected_mask_change_audit.json이며 selected checkpoint SHA도 검증했다.

## 후속 검증 완료

- 251번의 고정320 원래 mask 재현과 성능 유지 검증을 통과했다. 평균FG-ARI는 원래240의 .889187에서 .889629로 유지됐다. 이320장은 재사용한validation으로 untouched test가 아니다.
- 이후 사용자의 요청에 따라 배경 역할 고정과 영역 삭제를 없앤252번을 별도로 구현했다. 기존 데이터 고정320 평균FG-ARI는 .892326이며 원래240보다 절대 .005 이상 감소하지 않는 조건을 통과했다.
- 252번의 Multi-dSprites grid64 및 Tetrominoes grid32 zero-shot/추가 학습 후 세 seed TEST360 및 동일TEST1000 평가와 독립 검증을 모두 완료했다.
- 최종252 결과와 Foreground IoU 해석상의 차이는 [최종 보고서](../neutral_readout/RESULTS_FINAL_KO.md)에 기록했다. 별도 비지도 최고165번의 두 데이터셋 전이 평가도 완료했다.

최초251 후보 선정은 validation64만 사용했고 고정320 activity는 정답을 열지 않고 미리 준비했다. 작은 TRAIN64/validation16 실험과 평가 이미지 수가 다르므로, 두 표의 평균 차이를 학습 데이터 증가 효과로 해석하지 않는다. 이 문서의 비교 표는 당시251 validation 단계의 기록이며,252의 배경 비고정 모델과 구분한다.

원본 근거: SW0251_background_mask/collected/validation_TRAIN256.json, validation_selection.json, h_cv_z_seed*_*.json 및 collected_manifest.json의 서버 SHA 검증 기록. Benchmark Python 파일과 source240 checkpoint는 변경하지 않았다.

````

</details>

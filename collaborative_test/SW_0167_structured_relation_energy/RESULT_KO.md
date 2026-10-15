# SW0167 결과: 분할 loss 학습은 성능 개선으로 이어지지 않음

2026-10-15. 고정320장, native32, seed1 평가를 완료했다. 두 학습은 동일한512 TRAIN 이미지·초기값·순서·385-parameter pair MLP를 사용하고 loss만 다르다. 각256 Adam 업데이트와4096 노출, 모든 parameter의 실제 optimizer step256, 최종 head와 durable boundary 일치를 검증했다. 두 학습의 binding은 arm 이름 외에 정확히 같았다. Source encoder·graph·core는 학습하지 않은 **supervised head-only 실험**이다.

| 방식 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW164 / SW165의 동일 seed1 설정 | .770932099 | .614993430 | **.630523482** |
| Structured partition hinge | .770932717 | .614993430 | .627593144 |
| Matched local margin | **.771162710** | .614993430 | .629836618 |
| 동일 rollout의 Source-T QCC | .755273083 | .614993430 | .579513786 |
| matched Slot70k seed1 | .841819292 | .203416709 | .222115156 |

Structured FG-ARI의 기존 partition 대비 차이는+.000000618, paired95%CI[-.000540090,+.000509511]이다. Local margin은+.000230612, CI[-.000119001,+.000637749]다. Structured-minus-local은-.000229994, CI[-.000622098,+.000042253]다. CI는 고정320 이미지의 common paired bootstrap10000회/RNG167이며 seed 모집단의 CI가 아니다. Foreground IoU는 모든 이미지에서 Source-T와 정확히 같았지만 두 새 head의 Object IoU는 기존 partition보다 낮았다. **등록한 expansion gate 실패: 추가 seed·70k 확장이나 이 recipe의 loss/width/계수 sweep을 하지 않는다.**

학습 자체가 멈춘 결과는 아니다. Epoch1→8 평균 TRAIN loss는 structured .293436→.267798, local .551978→.490236으로 감소했고 모든256 업데이트에서 gradient norm이 양수였다. 서로 다른 loss 값의 크기를 성능 점수로 비교하지 않는다.

실제 분할도 일부 변했다. Label 번호에 무관한 pair-membership 비교에서 structured는54/320 이미지, local은43/320 이미지에서 기존 partition과 달랐다. 평균 unordered-pair disagreement는 각각 .001828970/.000423163이다. 다만 FG-ARI는 각각292/298 이미지에서 같았다. 이것은 관측된 출력 변화량이며 encoder가 원인이라는 증명이나 단순 학습 부족의 증거는 아니다.

두 head와 reference를 먼저 고정한 뒤에만 VAL GT를 읽었다. 모든320 이미지에서 fresh Source-T QCC와 별도의 zero-residual SW164 partition 재현을 확인했고, source state는 전후 동일한 SW161 digest family로 검증했다. Prediction/score 큐는 모두 정상 완료했다. 공개 `validate_score()`가 전체 metric·paired CI·gate를 독립 재계산해 통과했고, root가 다시 실행한 검증도 같은 score SHA를 재현했다: `714deb2ab8a0c944d5b5fbedfe09b568433de54c959b5d486414e0629dcb4ab4`.

**최고 모델은 SW165 유지:** 세 seed 평균 .804298915/.609975448/.651148975. 위 seed1 pilot과 이 세-seed 평균을 섞어 성능 향상으로 해석하지 않는다. .87, physical-gate 기여 입증, data/70k scaling은 아직 해결하지 못했다. 다음 후보는 이 frozen-source readout을 세부 조정하기보다 encoder/graph/native core까지 실제로 학습할 수 있는 다른 가설을 별도 설계로 검토한다.

근거: `results_archive/completed_training_20261015/`, `results_archive/completed_score_20261015/server_records.json`, `independent_verification.json`.

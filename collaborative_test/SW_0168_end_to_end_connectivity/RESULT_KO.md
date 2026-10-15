# SW0168 결과: maximin은 local보다 낫지만 기존 모델에는 패배

2026-10-15. 고정320장·native32·seed1, TRAIN512장·두 pass·64 Adam logical updates의 supervised joint-training 비교를 완료했다. Encoder·graph·native core·integration을 학습했다. 두 arm의 최종 checkpoint/optimizer/history/boundary와 최종 prediction/score를 독립 검증했고, root가 공개 `validate_score()`로 실제 GT에서 전체 지표와 paired bootstrap을 다시 계산해 통과했다. 평가 복구 동안 학습을 반복하지 않았다.

| 방식 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 SW164 / SW165의 동일 seed1 | **.770932099** | .614993430 | **.630523482** |
| Maximin, 기존 SourceT foreground 유지 | .757128250 | .614993430 | .624364865 |
| Maximin, learned rollout의 standalone mask | .751409368 | **.618290863** | .617541401 |
| Local control, 기존 SourceT foreground 유지 | .753386670 | .614993430 | .622111225 |
| Local control, standalone mask | .745430517 | .610882881 | .611957898 |
| 기존 SourceT QCC | .755273083 | .614993430 | .579513786 |

Conditional maximin은 local 대비 FG-ARI+.003741580, paired95%CI[+.000728168,+.006925432]였다. Standalone도 local 대비+.005978851, CI[+.001482236,+.010460219]였다. 따라서 이번 비교에서 structured connectivity credit은 matched local credit보다 좋았다.

그러나 기존 SW164 대비 conditional FG-ARI는−.013803849, CI[−.019752503,−.008323134]이고 standalone은−.019522730, CI[−.027443721,−.012086738]다. 두 Object IoU도 기존보다 낮았다. Conditional foreground는 모든 이미지에서 원본과 정확히 같았으므로 그 FG-ARI 감소를 foreground 범위 감소로 설명할 수 없다. Standalone extent IoU 상승만으로 instance 구분 성공을 주장하지 않는다.

Bootstrap은 등록한 공통320 이미지·10000회·RNG168이다. 이는 이미지별 paired 차이이며 세 seed 모집단 신뢰구간이 아니다. **두 co-primary를 모두 요구한 expansion gate 실패: 같은 recipe의 seed0/2·70k 확장, LR/passes/loss mix/threshold/support 선택 sweep을 하지 않는다.** Conditional만 골라 성과로 바꾸지 않는다.

고정 TRAIN8에서 initial/final·두 arm 모두 settled binary E-on1, transition0이라는 진단도 확인했다. 그8장에서는 S=g이고 membrane 발화 선택의 기여는 미증명이다. Nonzero surrogate gradients와64회 parameter update는 이 한계를 해소하지 않는다. 다만 이 포화 진단 자체가168의 FG-ARI 감소 원인을 확정하는 것은 아니다. `dendric_layer` raw norm의 이름 필터 누락도 별도로 기록했다.

최고 모델은 SW165 세 seed 평균 **.804298915/.609975448/.651148975**로 유지한다. 이번 결과는 seed1 pilot이며 그 평균과 섞지 않는다. FG-ARI 평균.87·functional gate contribution·fresh native32 70k/data scaling은 여전히 미달이다. 다음 모델 변경은 별도 설계로 등록하며168의 실패를 재학습이나 평가 계약 변경으로 지우지 않는다.

최종 score SHA256: `0a197c6292df1ccd9ebdeff6ea04563bcb7f87f903faa705f5aa75a77072fe70`.

증거: `results_archive/completed_score_20261015/server_records.json`, `independent_verification.json`; 학습·운영 이력은 `TRAINING_RESULT_KO.md`, gating 해석은 `POST_TRAIN_DIAGNOSIS_KO.md`. 실제 생성 파일은 `collaborative_testing/generated/` 아래에 보관한다.

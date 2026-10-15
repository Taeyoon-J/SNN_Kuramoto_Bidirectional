# SW0168 본 학습 완료: 성능 평가는 별도 진행

2026-10-15. 두 arm 모두512장·두 pass·64 logical B16 업데이트를 완료했다. Worker의 완료 로그에 더해 root가 각 최종 artifact의 공개 validator를 독립 실행해 checkpoint·optimizer·history·durable boundary 일치 검증을 통과했다. 재학습하지 않았다.

| Arm | Epoch1 TRAIN loss 평균 | Epoch2 TRAIN loss 평균 | 최종 manifest SHA256 |
|---|---:|---:|---|
| Maximin | .350470750 | .340214532 | `14fe4b3055155f15889ae293fd12c5b41cbd3abc95b39637c6662a07b4b57b71` |
| Local edge | .248665896 | .231140953 | `d075eab288cb6f2b393f81e845178b7980ff6a5800a695bf6da43001d38cf3a3` |

서로 다른 목적함수의 크기를 성능 점수로 비교하지 않는다. Maximin/local 양쪽에서 Kuramoto·membrane·integration parameter delta가64회 모두 양수였다. Native core의10개 parameter에는 실제 Adam step64가 기록됐으며 `dendric_layer.tau_n`, oscillator dense weight/bias도 포함한다. 다만 raw gradient 진단의 `dendrit` 문자열 필터가 실제 이름 `dendric_layer`를 놓쳤으므로, **dendritic raw gradient norm이 개별 측정됐다고 주장하지 않는다.**

더 중요한 제한은 고정 TRAIN8의 실제 동역학이다. Initial/final·양쪽 arm 모두 settled event-on fraction=1, event transition fraction=0이었다. 이 기록에서는 `S=E*g`의 E가 항상1이므로 settled S는 g와 같다. Membrane과 integration의 nonzero surrogate gradient 및 parameter update는 유효하지만, 이8장에 대한 실제 발화 선택의 기여를 입증하지 않는다. 다른 TRAIN 이미지나 전체70k까지 동일하다고 일반화하지 않는다.

Maximin supervisor는 worker가 `training_complete`를 출력하고 최종 모델까지 게시한 뒤, 정상 종료 시점의 `/proc`/Popen race를 identity 오류로 판정했다. 이 때문에 기존 prediction/score 큐가 의존성 실패로 종료됐다. 원본 상태와 로그를 수정하지 않고 보존했다. 완료 모델은 위 독립 validator를 통과했으므로 **학습을 다시 돌릴 이유가 없다.**

새 CPU recovery receipt는 두 모델·64회 이력·optimizer·checkpoint·boundary·원본 terminal state/완료 로그의 SHA를 결합한다. Receipt SHA는 `ceb3f2c06386c4b616cbbfd9c158656dfebe21268be29dd7429d8e99d0ad27e3`다. 첫 v2 prediction 재개는 GPU owner helper에 필수 callback을 빠뜨린 호출 오류로 GPU 실행 전에 종료돼 별도로 보존했다. v3는 실제 `_stable_idle` signature에 대한 autospec 검사를 추가하고 callback을 전달한다. 또한 active worker의 자기 GPU memory를 idle admission threshold와 혼동하지 않도록 검사하고, 정상 종료 시 실제 Popen을 기다려 재확인한다. 이는 모델·loss·평가 계약 변경이 아니다.

아직 held-out FG-ARI 결과가 없으므로 incumbent는 SW165이며 성능 개선을 주장하지 않는다. Source support를 쓰는 conditional과 같은 learned rollout에서 support를 구하는 standalone을 모두 고정320에서 평가해야 한다.

증거: `results_archive/completed_training_20261015/independent_verification.json`, `results_archive/evaluation_failed_20261015/server_records.json`. 실제 생성 파일은 `collaborative_testing/generated/`에 저장한다.

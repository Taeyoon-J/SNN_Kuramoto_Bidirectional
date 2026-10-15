# SW0171 — matched prefix negative

두 조건 모두 GPU1/2에서 에러 없이 실제 8 Adam updates를 완료했다. 동일 source 초기화, 동일 TRAIN64 두 pass와 B16/micro4, 각 128 exposures를 검증했다. 첫 두 업데이트의 Q→S 및 graph/core/context credit과 실제 가중치 변화가 확인됐고, frozen base/preprocessing/T contrast는 유지됐다. 학습 후 original source 전체 digest가 바뀌는 것은 trainable core가 실제 변경됐기 때문이다.

| 고정 TRAIN8 진단 (낮을수록 좋음) | 초기 | Q-only update8 | Q+instance update8 |
|---|---:|---:|---:|
| actual-spike Q maximin loss | 0.374244 | 0.376598 | 0.380756 |
| instance prototype loss | 0.693389 | 0.697329 | 0.673850 |

Q+instance는 embedding의 물체별 분류 loss를 낮췄지만, 실제 spike에서 계산한 연결 loss는 초기 및 Q-only보다 모두 나빠졌다. 따라서 등록한 prefix gate를 통과하지 못했으며, 이 recipe를 64 updates/추가 seed/70,000장으로 확대하지 않는다. own-QCC nonempty 통과 여부 및 모든 소수점 원본은 첨부 독립 검증에 기록했다.

이 결과는 짧은 TRAIN 진단에서의 확대 중단 근거다. VAL FG-ARI를 측정하지 않았으므로 최종 FG-ARI 하락이나 장시간 학습의 불가능성을 주장하지 않는다. 또한 encoder gradient/가중치 변화만으로 gate의 물체 구분 기여가 입증된 것은 아니다. 현재까지의 증거는 embedding 개선이 actual-spike binding 개선을 보장하지 않는다는 것이다.

`results_archive/completed_prefix_20261015/`에 두 최종 manifest/history/초기·최종 진단, checkpoint·Adam·boundary SHA, terminal PID와 실제 배포 테스트 기록 및 독립 재계산을 보관했다. 데이터·checkpoint 파일은 GitHub에 올리지 않는다. 최고 모델은 여전히 SW165이며 goal은 유지한다.

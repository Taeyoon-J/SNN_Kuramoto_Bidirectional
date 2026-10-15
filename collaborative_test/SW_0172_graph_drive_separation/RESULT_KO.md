# SW0172 — gamma graph/drive separation prefix negative

유효한 full1024-step v4 초기 검증 후, GPU2의 shared_gamma와 GPU1의 separated_gamma가 모두 에러 없이 실제8 Adam updates를 완료했다. Original SourceT의 동일 초기 상태와 TRAIN64×2passes/B16/micro4/고정lambda를 사용한 fresh matched 비교다. Invalid v1–v3 검증 시도는 모델 실패 결과에서 제외하며 ADMISSION_KO.md에 구분했다.

| 고정 TRAIN8 진단 (낮을수록 좋음) | 초기 | shared gamma update8 | separated gamma update8 |
|---|---:|---:|---:|
| 실제 spike Q maximin loss | 0.374244 | 0.380800 | 0.380393 |
| instance prototype loss | 0.693389 | 0.673965 | 0.673843 |

경로 분리의 Q loss는 shared control보다 조금 낮지만 자기 초기보다 높다. Prototype loss 감소와 own-QCC nonempty만으로 확대 조건을 충족하지 않는다. 등록된 prefix gate가 미통과했으므로 이 recipe를64updates/추가seed/70,000장으로 확대하지 않는다. Source base/normalization/T contrast는 그대로 유지됐고 context 및 actual native parameter 변화가 확인됐다. 첫 두 실제 업데이트에서 L_Q→context gamma, Q→S 및 graph/core/context credit 검사도 통과했다.

이 결과는 짧은 TRAIN 진단의 확대 중단 근거다. VAL FG-ARI를 측정하지 않았으므로 최종 FG-ARI 하락이나 장시간 학습의 불가능성을 주장하지 않는다. Gamma 경로 분리만으로 actual spike binding을 개선하기에 충분하다는 근거를 얻지 못했다. 기존171 결과와 함께, embedding loss 감소가 task Q 개선을 보장하지 않는다는 관찰을 유지한다.

원본 final manifest/history/초기·최종 진단, checkpoint·Adam·boundary SHA, terminal PID 및 실제 배포 CPU 검증은 results_archive/completed_prefix_20261015/에 있다. 데이터·checkpoint는 GitHub에 올리지 않는다. 최고 모델은 SW165이며 전체 연구 goal은 유지한다.

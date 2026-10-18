# SW0210: 전체320·64×64·독립 세 seed 평가 완료

동일 canonical128 RGB/no crop, IDs1320–1639, 2px modal GT와 기존 고정
readout에서 비교했다. 우리 모델 960개와 Slot960개 mask를 모두 검증한 뒤
GT를 읽었다. 별도의 읽기 전용 재검증도 모든 mask·평균·paired CI를 재계산해
일치했으며, 독립 metric 구현 간 최대 차이는2.22e-16이다.

| 모델 | seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|---:|
| 우리 warm192/205 | 0 | .877709 | .791224 | .696284 |
| 우리 warm192/205 | 1 | .825583 | .517168 | .660796 |
| 우리 warm192/205 | 2 | .842552 | .469206 | .659062 |
| **우리 평균** | **0/1/2** | **.848615** | **.592533** | **.672047** |
| 비교 Slot | 0 | .636265 | .205508 | .184986 |
| 비교 Slot | 1 | .846161 | .197193 | .212411 |
| 비교 Slot | 2 | .878278 | .191901 | .188211 |
| **Slot 평균** | **0/1/2** | **.786902** | **.198201** | **.195203** |

세 평균 모두 Slot보다 높다. 차이는 FG+.061713 / FIoU+.394332 /
ObjectIoU+.476845이다. Paired image bootstrap95% CI는 각각
[.044403,.078783], [.377264,.412516], [.461945,.491677]이다.
이는 세 고정 모델의 이미지별 평균을 resample한 구간이며 seed 모집단에
대한 불확실성 구간은 아니다.

이번 결과는 16장 mini의 .90과 달리 전체320장의 세 seed 결과다. 그러나
**FG-ARI 평균 .90 목표는 아직 미달**이다. seed0가 현재 세 모델 중 가장
높지만 각 seed의 최고 설정을 따로 골라 평균한 결과는 아니다. 공통 TRAIN8
적응 recipe를 사용했고 seed0는 완료된208 결과를 재사용했다.

이 모델들은 기존70k core 학습과32-grid 학습을 물려받고, mask GT를 사용한
supervised 적응을 포함한다. Native64 추가 고유 TRAIN은 seed당8장이다.
210에서는 학습 update0이며 새 native64 TRAIN70,000 학습을 수행하지 않았다.
비교 Slot은 기존154/208에 고정된 reference이며 공개 checkpoint의 과거
.8946 수치와 평가 조건을 섞지 않는다. 같은 평가 계약에서의 결과 비교이며
동일한 supervision과 학습 비용을 맞춘 우월성 실험이라고 주장하지 않는다.

320장은 반복 사용한 validation이다. 데이터 scaling, untouched test 일반화,
모든 gating의 인과 기여를 이번 평가로 입증하지 않는다. 기존205의 작은
적응 전후 비교가 실패했다는 결론도 바꾸지 않는다. 이번210은 선택된 공통
recipe의 절대 성능을 더 넓은 평가에서 확인한 것이다.

이 결과를 native64 세 seed 기준점으로 보존한다.211은 spike/membrane에서
center·offset을 학습해 grouping을 개선하며,212는 동일 head의 학습 전경을
결합하는 별도 비교다. .90 목표를 위해 goal은 활성 상태로 유지한다.

증빙: results_archive/completed_v1/score_v1/summary.json,
completion_audit.json,8개 shard protocol과 queue 기록.
Checkpoint 식별자는 results_archive/selected_reference_v1.json에 기록한다.

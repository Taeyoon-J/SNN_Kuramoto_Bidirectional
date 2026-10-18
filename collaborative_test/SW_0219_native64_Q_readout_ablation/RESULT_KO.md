# SW0219 native64 actual-Q readout ablation 결과

실제 best-reference seed0/1/2, 동일 고정16장/64x64 patch. Graph가 spike를 만드는
dynamics는 그대로이고, decoder의 추가 A 가중만0으로 두었다. GPU/core forward/학습0.
원래 production helper로 baseline을 bitwise 재현한 뒤 같은 helper를 사용했다.

| 세 seed 평균 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| baseline | 0.842891 | 0.600692 | 0.702161 |
| Q_only | 0.833809 | 0.600692 | 0.701250 |

Q_only FG 차이 -0.009081, paired image95% CI[-0.022494,-0.000757].
사전 확대 gate 통과: False.
CI는 세 고정 모델의 image별 평균을 재표집했으며 seed 모집단 CI가 아니다.

| seed | baseline FG | Q_only FG |
|---|---:|---:|
| 0 | 0.924401 | 0.904659 |
| 1 | 0.815856 | 0.811901 |
| 2 | 0.788414 | 0.784867 |

Local4/server4 tests와 실제48개 Q/A cache 검증 및 real native64 CPU smoke를
통과했다. 모든48개 actual Q/A에서 두 readout을 다시 실행해96개 mask endpoint를
bitwise 재현하고 독립 지표/평균/paired CI를 검증했다. 최대 metric 차이 2.22e-16.
원래 전경, singleton,4-connectivity, Q threshold, NATIVE_NODES1024와 모든 모델이 유지됐다.

217의 pair AUC 차이만으로 실제 partition 개선을 결론 내리지 않고 확인한 실험이다.
이 비교는 decoder A weighting의 조건부 ablation이며 물리적 graph/gating 전체의
인과 기여를 증명하지 않는다. 반복 사용한16장은 untouched test가 아니다.
중간 alpha/threshold sweep은 하지 않는다. FG-only 유용 후보는 별도 보존하되
mini 결과를 full320/세 seed .90/신규 native64 TRAIN70k로 표기하지 않는다.
기존 full320 최고 평균 .848615/.592533/.672047은 그대로 유지한다.

근거: results_archive/completed_v1/score_v1/summary.json 및 completion_audit.json.

Q_only FG는 세 seed에서 모두 낮아졌고 paired image CI도 음수다.
따라서 현재 fixed16/세 고정 모델에서는 decoder의 A weighting을 유지한다.
Pair AUC가 더 높다는 사실이 discrete grouping 성능 향상을 보장하지 않았다.
이는 해당 readout 가중의 조건부 기여 근거이며 physical gate 전체의 증명은 아니다.

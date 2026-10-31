# SW_0219_native64_Q_readout_ablation

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `6e60e46ea766b343c4365c1078656a13e83dec4ca3d19df00e0c6965dea31504`

````markdown
# SW0219: 실제 Q만 사용하는 readout 가중 ablation

217의 actual Q pair AUC는 native graph-weighted energy보다 높았다. Graph가
spike를 생성하는 dynamics는 유지하고, decoder에서 A가 energy 크기에 주는
추가 가중만 제거하면 어떠한지 확인한다. 물리적 graph/gate 전체 제거 실험이
아니며 gating 인과 증명이라고 하지 않는다.

실제 best-reference seed0/1/2, 같은16 offsets0,2,..30와217의 실제 private
Q_F/A_F/gain을 재사용한다. 새 rollout, encoder/core update, GPU 사용이 없다.
Q_F 밖은 neutral Q=.5/A=0으로 채워 full-N 연산을 호출한다. F 내부 actual
Q/A는 그대로이고, source 전경 밖은 merge에 참여하지 않으므로 정규화를
바꾸지 않는다. 원래 SW0164 helper를 호출해 baseline mask를 먼저 bitwise
재현해야 한다. 이후 같은 helper에서 A만0으로 두어 logit(Q)의 signed
agglomeration을 실행한다. Threshold .5, 4-connectivity, positive cross-sum,
tie 순서, singleton 보존, 원래 전경과 NATIVE_NODES1024는 유지한다.

198의1024->4096 normalizer와 다르게 이번에는 readout A weighting의0/1
비교다. 중간 alpha/threshold sweep을 하지 않는다. 163의 learned context
assignment Q_only head와도 다르다. 단일 사전 고정 후보이며217 진단 이후의
exploratory validation이다. 반복 사용한16장은 untouched test가 아니다.

모든96 endpoint를 먼저 고정/검증하고 GT를 읽어 세 지표, seed별 결과,
paired-image bootstrap10000 및 세 고정 모델 평균을 계산한다. 확대 gate는
평균 FG+.01 이상,95% CI lower>0, ObjectIoU 감소 없음. FG-only 유용 후보는
따로 보존한다. Mini .90을 전체320 세 seed .90 또는70k로 바꾸어 말하지 않는다.
원래 source baseline/CNN/core/학습 이력과 최고 full320 .848615는 그대로다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `392da4a0233c85a612a51c19634c4924f5791232824bcfe2a8166db9a6926171`

````markdown
# SW0219 native64 actual-Q readout ablation

CPU reuse of all48 private native rows from217. Bitwise replay the original
Q/A baseline, then remove only A weighting at readout using the same production
agglomerator. No new core forward, GPU work, fitted threshold or model update.
The native graph remains inside the actual spike dynamics. This is a readout
ablation, not removal of a physical gate. See DESIGN_KO.md.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `a638c822da17c3a4f232d617087fb7dc8aef538155095f1312945b2d5cfd6874`

````markdown
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

````

</details>

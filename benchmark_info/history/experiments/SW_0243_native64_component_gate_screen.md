# SW_0243_native64_component_gate_screen

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `925ab8dd913efdbca1a2844f1d8a5af40b274a1d7e0035480988b385f889b25d`

````markdown
# SW0243: component-specific sinusoidal gate viability screen

Native64 traces showed a shared gate across four components and99.3–100% observed event occupancy in the six TRAIN cases. The controlled241 replay showed event information changing two masks, so it is not absent; however most emitted signal still follows the shared gate. Test whether preserving each component's own delayed phase through the sinusoidal layer improves grouping.

Replace the shared raw mask0.5*(1+sin(mean(delayed_theta))) with four masks0.5*(1+sin(delayed_theta_component)). Each component's carrier is sin(current_theta_component) times its own delayed mask. Use those gates in the existing dendritic integration, membrane integration/freeze and emitted event multiplication. The rollout is an explicit fork with only the gate import and folded-gate construction changed. Kuramoto, graph, delays, parameters, threshold, recurrence, T1024/settle512 and production readout remain the same. No training is added.

Use the accepted232 checkpoint confirmed by237. The targeted coarse screen is seed1, the seed whose two238 TRAIN cases had fully saturated events. Predetermine the first four original-mini validation inputs: offsets0,2,4,6 in the fixed320 contract. This is a viability screen, not a three-seed performance claim or new incumbent. Compare against the exact cached237 masks for those inputs, and show the matching Slot seed only as a scoped reference.

Before scoring, freeze all four candidate masks. On the first input also replay the original native model and require its mask to match the cached control bitwise. Reset the inference RNG and require the candidate theta trajectory and captured adjacency to match that replay bitwise, isolating the downstream gate change. CPU tests compare the actual small-core rollouts, parameter preservation, aligned-component parity, phase-lift/permutation properties and queue routing. Actual checkpoint restoration precedes CUDA allocation.

After all prediction protocols and Slot artifacts validate, read evaluation GT. A promising screen requires candidate FG-ARI at least0.01 above its matched control and neither IoU mean decreasing. Otherwise close this untrained gate-change branch without coefficient/threshold/epoch sweeps. A positive screen justifies a separate16-image three-seed confirmation; it cannot establish the full320 FG0.90 target, gating necessity, large-data scaling or unsupervised eligibility. Existing mask-supervised weights and the running70k job stay fixed.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `312a81095dc1a6845caef5196b7fad2636321f87b766c6e6bd858f5288fd41e7`

````markdown
# SW0243 closed: untrained component-gate substitution

This is seed1 on four preregistered validation images, not the full320 contract or a three-seed mean.

| Same four images | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| Cached237 control | 0.969270 | 0.647514 | 0.754141 |
| Component-specific gates | 0.914802 | 0.826210 | 0.697210 |

FG-ARI decreased0.054468 and Object IoU decreased0.056931; foreground coverage improved0.178696. The preregistered expansion gate failed, so close this frozen-weight substitution without coefficient or threshold sweeps. The candidate's0.9148 on this small easy prefix does not establish the full320 FG0.90 target; the matched parent already scored0.9693 on exactly these inputs.

The intended gate change executed: component gates differed by almost1.0, while the original mask replay and the complete upstream theta/adjacency trajectory matched bitwise on the control image. No parameter update occurred. However observed events remained fully on wherever gates were positive in all four cases. Preserving separate phase gates therefore did not correct the membrane's saturated operating regime for these inputs, and did not improve grouping under the fixed accepted weights.

Independent rescoring of actual masks reproduced the metrics with maximum error2.78e-17. Four component-wise gate/spike/membrane mean/std arrays per case were finite and had the expected native64 shape. Seven local and seven Linux tests and actual CPU restoration passed. These results do not rule out a separately trained component-gate architecture, establish membrane necessity or demonstrate large-data scaling.

Retain the shared-gate237 checkpoint as the current full320 reference:0.872680 /0.626110 /0.676331. The separate70k training remains unchanged. The collaborator branch was rechecked at a1df1c3c2d43693050ff9a55e1895abd83461de1 with no new updates.

````

</details>

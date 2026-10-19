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

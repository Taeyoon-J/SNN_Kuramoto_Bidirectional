# SW_0039_per_region_transduction

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `fa1124516d66397fa5d913e59dff5ee9fd594454c842e69415b5ed3ada617151`

````markdown
# SW_0039: Per-region dendritic transduction

Status: on hold pending resolution of checkpoint/data pairing. Do not launch
this architecture experiment until the root agent explicitly resumes it.

This tests whether giving each region its own dendritic projection improves
binding while keeping the Kuramoto drive and the rest of the SNN fixed. The
default `shared` mode preserves the prior module names, parameter shapes, RNG
order, and checkpoint loading behavior. `per_region` gives each region its own
weight `[N, branch, input_vector_dim + 1]` while retaining one shared bias
`[branch]`. Its initial weights are copies of the same seeded shared Linear,
so the two modes begin with exactly equal forward outputs; training can then
specialize the regional weights.

The seed-0 factorial uses the SW_0038 spike-PLV weight-5, vth-2 settings and
graph spatial decay .55. `branch=4` is held fixed in every cell. The varied
factor is `plv_bimodality_weight`: existing SW_0038 `spike5_seed0` is the
shared-weight-1 control. Train the three remaining cells with
`run.sh GPU_ID VARIANT 0`:

| Evaluation cell | Projection | PLV bimodality weight | Checkpoint |
|---|---|---:|---|
| `control_b1` | shared | 1 | Existing SW_0038 `spike5_seed0` |
| `shared_b6` | shared | 6 | New SW_0039 run |
| `regional_b1` | per-region | 1 | New SW_0039 run |
| `regional_b6` | per-region | 6 | New SW_0039 run |

After each run, call `evaluate.sh GPU_ID VARIANT 0`. It explicitly constructs
the evaluator in the checkpoint's projection mode and runs SW_0036's stage
diagnostic, SW_0035's selected classifier suite, and a same-checkpoint,
label-free phase-versus-spike affinity correlation diagnostic. That diagnostic
uses the existing `return_theta=True` API and the same gamma images for both
affinities; it does not use labels to form predictions.

For component mode (`spike_per_component=True`), expected dendritic plus
membrane parameter counts are 1292 for shared branch-4 and 3332 for
per-region branch-4. The focused local test checks this count, exact seeded
forward equivalence and regional gradients, strict state-dict roundtrips, and
default/shared key compatibility. Regional checkpoints require the explicit
`--dendritic-projection per_region` evaluator option; strict loading will reject
them under the default shared mode.

All cells use branch-4 dendrites. The existing control checkpoint is reused
exactly, with no retraining. Do not pool these results with peer scores or
change the fixed split/evaluator contract.

````

</details>

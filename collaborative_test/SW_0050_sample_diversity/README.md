# SW0050 - Cross-sample activity diversity for seed2 collapse

This is a model-training intervention on the SW0042 aligned BIM6 recipe, not
an inference-only diagnostic. The trainer already implements
`sample_activity_diversity_loss`; SW0050 makes its CLI name explicit and
applies it to the selected training activity. The recipe sets
`--loss-signal sigmoid_membrane`, so the criterion's legacy `spikes` argument
contains `sigmoid(core_out)`. No labels enter the loss.

We cannot measure real-batch loss/gradient scale in this local Windows
workspace because the server gamma/checkpoint assets are not mounted. Instead,
`probe.sh GPU_ID` measures the unweighted objective and gradient norms on four
real 16-example gamma batches using the trained SW0042 seed2 checkpoint. It
reports norms for graph-generator, upstream/Kuramoto, dendrite/membrane, and
all parameters, then suggests the weight giving a diversity gradient norm
about 0.1x the full BIM6 baseline gradient. A second suggestion reaches 1x,
but that high level is deliberately held: centering activity makes constant
patterns degenerate for this objective, and large weights could favor noisy
variation over image-specific object structure.

The first seed2 comparison is a pair of full 40-epoch runs with identical
seed/data order and SW0042 architecture:

| Arm | Sample-diversity weight | Learning rate | Role |
|---|---:|---:|---|
| A | Measured for 0.1x baseline gradient | 1e-3 | Rescue candidate |
| B | 0 | 3e-4 | Learning-rate harm/control arm |

Both runs save intermediate state_dicts at epochs 20, 25, 30, 35, and 40.
After choosing A on seed2 validation, run the same selected A weight at seed0
as a harm-control only if seed2 shows a material signal. Two-epoch `pilot`
mode is available for script plumbing but is not used to screen the late
collapse, which emerges around epochs 29-40.

Each variant uses train gamma IDs 0-999, BIM6 phase/spike losses, sigmoid
membrane activity, shared projection, factorized backend, vth .06, graph decay
.35, geodesic steps 3/radius 1.5/contrast 2/temperature .5/cap 16, 64 rollout
steps and PLV settle 32. Evaluation reuses SW0040 on HDF5-aligned IDs 1320-1639
at both T256/settle64 and T1024/settle512, with the usual threshold sweep and
same-window phase diagnostic.

Queue the seed2 diversity rescue for a measured 0.1x weight and the matched
lower-learning-rate control on separate GPUs. Both poll every 20 seconds and
start only after their assigned GPU is idle:

```bash
bash collaborative_test/SW_0050_sample_diversity/queue_seed2_full.sh GPU_A 0.1
bash collaborative_test/SW_0050_sample_diversity/queue_seed2_lr_control.sh GPU_B
```

The diversity queue first runs the gradient probe if its JSON is absent, then
launches the full seed2 run and watcher. Outputs are variant-specific and
overwrite is refused. To run seed0's selected-weight harm-control after arm A
has completed both validations:

```bash
bash collaborative_test/SW_0050_sample_diversity/launch_seed0_harm_control.sh GPU_ID SELECTED_WEIGHT
```

Local loss behavior/gradient test:

```bash
python -m pytest collaborative_test/SW_0050_sample_diversity/test_loss_behavior.py -q
```

Before any future SW0050 variant, `preflight.sh GPU_ID` now requires an idle
GPU, checks the absolute SNN Python and required CLI option, validates or
measures the real seed2-checkpoint gradient probe on real training gamma, then
performs exactly one optimizer update from the first real gamma row in a unique
temporary directory. It verifies both `core.pt` and the epoch-1 checkpoint
before writing `trained_models/SW0050_PREFLIGHT_V1.txt`. The marker binds the
checkpoint, gamma, trainer, loss, probe, and run-script hashes; code or asset
changes force a new preflight. Queue scripts call this only after their GPU
idle wait, and `run.sh` refuses a stale/missing marker or a newly occupied GPU.
This gate does not restart existing SW0050 arms.

Remote execution started on 2026-10-04. The no-diversity, LR `3e-4` seed2
arm completed all 40 epochs and both evaluations. At the common long threshold
`.35` it scores `.347156/.167773/.135250`, compared with the original seed2
`.185651/.135165/.036969`. This is a material rescue on all three metrics, but
still below seeds 0 and 1. Substituting this arm for only the collapsed seed2
would raise the three-seed common-threshold mean from
`.415879/.418226/.296118` to `.469714/.429096/.328878`; this is a diagnostic
counterfactual until the same selected recipe is checked for seed0/1 harm.
The measured-`0.1x` diversity arm remains in its 40-epoch training/evaluation
pipeline on GPU2.

The low-LR outputs are versioned under `results/seed2_low_lr/`.
The first LR-arm launch exposed a pre-environment bare-`python` validation
call and exited before creating an output directory; `run.sh` now uses the
absolute SNN-environment Python path, validates both weight and LR, and the
failed launcher log is retained on the server.

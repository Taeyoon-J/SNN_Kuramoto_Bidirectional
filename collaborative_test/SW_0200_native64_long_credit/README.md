# SW0200 native64 full settled-window credit

QCE and production inference stay fixed. Direct temporal credit grows from the
last64 to the last512 of1024 steps, covering the entire settled scoring window.
The first512 recurrent steps remain detached. This tests a longer learning
window, not a new gate or a claim of full-horizon exact gradients.

Matched reference is the actual completed199 control, with identical TRAIN8
model/Adam/RNG start and two new B4 proposals. Before training, a one-image
bounded-memory full512 gradient admission must pass and restore all state.
Read-only admission exposure is reported separately. See DESIGN_KO.md.

No performance result or full320/three-seed goal achievement is claimed here.
Raw imagewise GT-derived vectors, prediction arrays and checkpoints stay private.

2026-10-17 registration:14 local and14 Linux tests, all Python ASTs and2 local/
Linux CLI entrypoints passed. Actual CPU model/Adam/RNG matches the199 control's
start, Adam steps2, frozen control masks/protocol checked without opening GT.
Supervisor2000695 is registered for all four GPUs. Actual native64 GPU full512
memory/gradient admission is still pending; CPU tests do not establish GPU fit.

Actual GPU admission subsequently PASSED onGPU1: full512/no-grad forward losses
matched exactly, actual emitted-S and all five parameter-family gradients were
positive. Peak allocation6,431,305,216bytes (~5.99GiB), below the .8 allocator
budget. The read-only probe restored model/Adam/RNG before training. Worker2003084
then started the registered two-proposal training. This is learning admission,
not a segmentation improvement result or a gate-contribution ablation.
Public validation facts are in gradient_admission/public_receipt.json; the
single-TRAIN-image loss/gradient values remain private.

Completed and independently audited actual final model/Adam/history plus48 masks
(candidate/control/warm on16 fixed images), maximum metric error2.22e-16.

| Arm | FG-ARI | Foreground IoU | Object IoU |
| --- | ---: | ---: | ---: |
| Full512 credit | .899738 | .857661 | .723357 |
| Matched64 control / warm | .924401 | .859896 | .727638 |

FG difference-.024664,95%CI[-.074037,.001665]. One new update committed (actual
Adam step3), model and CNN changed; healthy credit/committed learning did not
produce a mask improvement on this mini. The registered extension gate failed.
Close this512 recipe, retain the incumbent/live64 setting, and do not extend it
with small tail/LR sweeps. A population degradation or general claim that longer
credit can never help is not established by this one seed/16-image screen.

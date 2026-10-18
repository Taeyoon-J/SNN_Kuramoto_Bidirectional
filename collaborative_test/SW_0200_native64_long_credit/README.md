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

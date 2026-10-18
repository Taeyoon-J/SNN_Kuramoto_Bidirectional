# SW0218 native64 energy-ranking training

Compare against completed SW0204 from the exact same warm192 model, Adam/RNG
and eight new TRAIN images. Add a loss that ranks actual native Q/A/gain energy
of the mined production partition below a TRAIN-GT connected reference. Keep
the source support, frozen before-step context, all gates, joint encoder/core
updates and the production spike readout. No decoder threshold or head fitting.

See DESIGN_KO.md for formulas, parity/credit checks, exact control, fixed16
screen and the full320 three-seed .90 goal. No result is claimed before execution.

Actual start2026-10-18: 22 local and22 Linux tests passed; actual CPU model/Adam/RNG admission passed. Supervisor2368483, worker2368524 on exclusive GPU1. Registration and actual NVIDIA/process evidence are archived. Peer patch_v2 was rechecked at7a21632; see PEER_REVIEW_KO.md. No new score yet.

Independent native64 TRAIN8 adaptation for weak seeds1/2. Pending preflight, no result. See DESIGN_KO.md.

2026-10-18:six local and six Linux checks, AST validation, queue CLI check
and actual independent-model CPU admission passed. The tests cover accepted
updates, rejected full transactions, wrong seed/GPU and inference resumption
without repeating completed training. Supervisors2054790/2054791 registered
for seeds1/2, respectively. Both scan all physical GPUs0..3 with shared exclusive
leases. Actual GPU training has not started at registration.

A CPU-only post-completion aggregate is registered after both jobs finish. It
combines the two new independent seeds with the already completed192 seed0
TRAIN8 adaptation on the same sixteen IDs. Exact batch_loss/run_probe source
equality and the original seed0 initial tensor transaction were verified.
Tensor transaction hashes do not cover RNG; no RNG proof is claimed from them.
Two extra local/Linux aggregation tests passed. GPU training source hashes were
unchanged by this addition. Aggregate audits actual model/Adam/history/masks
and compares against same-image native64 Slot outputs; no fresh Slot inference.

Seed2 worker2066038 actually obtainedGPU1 after204 completed. NVIDIA PID and
committed proposal0 checkpoint verified: original independent seed2 model
b2c1a56f... and empty Adam, not seed0 weights. Seed1 remains queued.

Seed2 completed and independently audited on fixed16 development images:

| Arm | FG-ARI | Foreground IoU | Object IoU |
| --- | ---: | ---: | ---: |
| baseline | .8276711245 | .4448827161 | .6892385263 |
| TRAIN8 candidate | .7884142488 | .4454744949 | .6743347722 |

One accepted update, actual Adam step1, changed CNN. Initial CPU/GPU tensor
transaction, actual boundary and32 masks verified. FG difference -.0392568757,
CI[-.1393297488,.0190920790]; this is a negative mini mean, not a population-level
proof of degradation. No adoption. Raw mask analysis shows fewer within-object
splits (.1226->.1115) but more cross-object merges (.0567->.1024), also on common
retained GT foreground. BG false positives hardly change (.3212->.3207).
Raw positive-label count nevertheless rises20.19->21.81 vs true5.375; counting
predicted groups alone does not diagnose true-object merging. Rates have different
denominators and are not additive causal shares. Existing masks only, no GPU
forward/update. Nine local/Linux operator tests passed. This does not prove
which gating causes the issue. Seed1 worker2095873 subsequently startedGPU1,
with its own initial model6c48389f... and empty Adam verified; final mean pending.

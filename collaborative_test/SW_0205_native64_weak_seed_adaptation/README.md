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

All three TRAIN8 seed lineages now independently audited on the same16 IDs.
Seed0 reuses completed192; source/data/update-code equality and actual boundary
were checked. New seed1 accepts2 updates, FG .7984->.8159, while seed2 accepts1
and FG .8277->.7884. This variation is not a uniform adaptation benefit.

| Fixed16 three-seed mean | FG-ARI | Foreground IoU | Object IoU |
| --- | ---: | ---: | ---: |
| before native64 adaptation | .8486554367 | .5850936464 | .6909369327 |
| after TRAIN8 adaptation | .8428907017 | .6006915810 | .7021610527 |
| comparable Slot | .8034094297 | .1641841075 | .2083960247 |

FG change -.0057647350, CI[-.0439420276,.0234490905]. IoUs improve in mean,
primary FG does not. Extension gate failed; close the shared recipe without an
update/LR micro-rescue. Slot FG difference CI also crosses zero. This remains a
16-image development result, not full320 or a three-seed .90 achievement.

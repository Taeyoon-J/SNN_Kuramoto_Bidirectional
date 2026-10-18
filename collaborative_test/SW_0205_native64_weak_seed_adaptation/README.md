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

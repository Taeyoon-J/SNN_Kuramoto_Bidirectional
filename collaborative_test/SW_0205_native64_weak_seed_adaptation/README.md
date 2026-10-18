Independent native64 TRAIN8 adaptation for weak seeds1/2. Pending preflight, no result. See DESIGN_KO.md.

2026-10-18:six local and six Linux checks, AST validation, queue CLI check
and actual independent-model CPU admission passed. The tests cover accepted
updates, rejected full transactions, wrong seed/GPU and inference resumption
without repeating completed training. Supervisors2054790/2054791 registered
for seeds1/2, respectively. Both scan all physical GPUs0..3 with shared exclusive
leases. Actual GPU training has not started at registration.

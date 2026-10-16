# SW0183 — completed shared-prefix unique-data comparison

All training, six arm predictions, GT-last scoring and independent audit are complete.
The 1024-image candidate scores **.810934 / .608932 / .661713**; the matched
512-repeat control scores **.813732 / .577707 / .665824**. Both exceed Slot,
but the registered unique-data gate fails. SW181 remains the supervised
three-seed FG incumbent; no native70k or gate-causality claim is made.

See [full results and limits](RESULT_KO.md) and [preregistered design](DESIGN_KO.md).
Actual server validation passed 29 tests, actual executable CLI checks,
first-two GPU spike/gradient/Adam admission, and independent final metrics.

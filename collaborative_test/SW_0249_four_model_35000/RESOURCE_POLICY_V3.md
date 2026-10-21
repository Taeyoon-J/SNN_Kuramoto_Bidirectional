# Shared GPU admission based on the running worker's observed footprint

The initial admission rule reserved the worker's full80% allocation cap plus
512MiB, approximately20GiB on24GiB hardware. This excluded GPU3 despite its
approximately17GiB free memory. Two workers had already passed actual native64
spike preflight and training, with observed resident memory2,234MiB each.

The resource-only v3 wrapper requires at least8GiB free before admission,
exceeding three times that observed residency plus headroom. This threshold is
not a claimed upper bound on allocation. Each unchanged worker still runs its
actual GPU gradient/spike preflight, retains its original80% PyTorch allocation
cap, and preserves the existing OOM retry/checkpoint policy. No foreign-process
arrival causes yielding; no other user's process is signalled.

Only ten waiting supervisors were replaced. Their exact identities and start
times were checked. Their old state/admission logs were archived, and replacement
queues resumed the same job paths under the same GPU/queue leases. The two active
training workers were untouched. The owned coordinator was briefly suspended
only to make this waiting-queue handoff atomic to its observer, then resumed.

The nine scientific-source files and their registered fingerprint were verified
unchanged before/after handoff. Batch, data, loss, optimizer, runtime math flags,
architecture, checkpoint state and readout are unchanged. The resource wrapper
and tests have their own SHA manifest retained privately. Both local and Linux
resource tests passed. This document supersedes the initial admission paragraph
of DESIGN_KO.md without editing that immutable scientific-source fingerprint.

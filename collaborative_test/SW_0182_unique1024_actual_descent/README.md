# SW0182 — unique TRAIN1024 at the SW0181 proposal budget

This tests whether replacing SW0181's repeated second TRAIN512 pass with the next
512 canonical TRAIN images improves fixed320 masks. Source initialization, loss,
four optimizer families, learning rates, RNG law, actual-loss acceptance and
64-proposal budget remain fixed. See [the preregistered design](DESIGN_KO.md).

The independent server data reference verified 1,024 unique IDs, canonical RGB
alignment, exact preservation of the old 512 RGB/modal rows, and no validation
or reserve reads. It recorded the expected full-array hashes before the candidate
cache was generated: [data reference proof](results_archive/cpu_v1/reference_data_proof.json).

The actual candidate cache was built on Frontier and matched the independent
RGB/modal reference hashes exactly. Its second load reused identical immutable
files. Five data/schedule tests passed locally and on the server:
[cache admission receipt](results_archive/cpu_v1/data_cpu_receipt.json).

The trainer, exclusive GPU supervisors, GT-free predictions and GT-last scoring
are implemented. All 32 local tests passed, including first-32-proposal control
matching, the actual fixed320 aggregation-to-gate connection, and a final GPU
owner/memory check immediately before prediction launch. Four separate module
`--help` subprocesses also passed:
[local CLI admission](results_archive/cpu_v2/local_evaluation_admission.json).

Server CPU training admission was submitted, but the SSH observer eventually
exited 1 with `Broken pipe` and `banner exchange timeout`. The previously live
local master also disappeared. This establishes a terminal observation failure,
not the state of the submitted remote command or a scientific trial failure:
[transport record](results_archive/cpu_v2/ssh_transport_terminal.json).
Inspect existing server receipts/processes through a restored connection before
retrying deployment or launching anything. No GPU training or fixed320 result
has been confirmed.

After actual server admission, the prepared queues run seeds 0/1/2, predict
each completed checkpoint, then score only after all three prediction archives
and protocols freeze. SW0181 remains unchanged. This is supervised fine-tuning
from a pretrained source, not fresh native32/70k training.

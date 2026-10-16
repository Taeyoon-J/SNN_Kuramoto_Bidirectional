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

Trainer validation is in progress. GPU training and fixed320 evaluation have
not started. SW0181 remains unchanged. This is supervised
fine-tuning from a pretrained source, not fresh native32/70k training.

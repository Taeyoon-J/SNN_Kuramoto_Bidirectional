# SW0170 CPU verification before GPU admission

2026-10-15. Leaf recurrence and Stage1 preflight are implemented and frozen.
Root and the independent reviewer each ran the 13 CPU contracts successfully;
the real Frontier interpreter also passed all 13. Actual fixed first-eight TRAIN
source/cache/RGB inputs were revalidated on Frontier without GT or CUDA.
Root compared the complete runtime fingerprint, including nested native
calibration dependencies, against local bytes and verified the deployed manifest.

There is no SW170 GPU admission or segmentation performance result in this
receipt. The queued GPU experiment must still test optimizer-zero loss/separation,
standalone support and only then disposable-update gradient feasibility. SW165
remains the incumbent. This admission uses eight TRAIN masks in a separate later
handoff; it never changes the fixed validation labels or the dataset.

The same-cycle peer fetch remains cab6c676e24a149a45ea960586c04e579f78d384.
Evidence: results_archive/server_cpu_admission_20261015/server_records.json and
independent_verification.json. Generated files physically live in the centralized
collaborative_testing/generated tree via the experiment archive junction.

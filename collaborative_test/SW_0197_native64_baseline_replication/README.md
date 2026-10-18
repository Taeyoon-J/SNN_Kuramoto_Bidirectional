# SW0197 native64 baseline seed replication

Two independent jobs completed: seed1 and seed2. Each scanned all four GPUs for an
exclusive idle lease and resumed frozen image rows after yielding.
This is read-only evaluation of the native32 incumbents transferred to64;
it does not repeat training or assume the pending194 adaptation succeeds.

The32-image development cohort, RNG, production masks, GT and metrics match193.
The three-seed baseline mean is assembled from actual frozen masks, including
193's verified seed0 baseline and comparable Slot3mean. This remains a mini
development result, not full320 validation or a new incumbent.

Public outputs are aggregate scores and audit/queue metadata. Imagewise scores
remain private. Admission/actual job handles are recorded under results_archive.

All96 baseline and96 Slot masks were independently rescored, max metric error
2.22e-16. Ours three-seed mean .8514281207/.5780720892/.6940038040;
Slot .8215711846/.1733153073/.2039881566 on the same32 images. Seed FG scores
are .9095369755/.8175115352/.8272358513. FG paired image-bootstrap CI for the
mean difference[-.026523,.082492] includes zero. No full320 or FG.90 goal claim.
All three are mapped native32 incumbents without extra native64 learning;
seed0 is the frozen BASELINE, not the TRAIN8 candidate from193.

Frozen-mask error attribution found substantially more background false positives
and object fragmentation in seeds1/2. Nine existing error/paired-aggregation tests
passed locally and on Linux. No new GPU forward or optimizer update was used.
Aggregate reports are in results_archive/aggregate_v1 and error_diagnosis_v1.
See RESULT_KO.md for scope, values and interpretation.

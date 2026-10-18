# SW0196 geodesic matrix products

Read-only numerical/throughput experiment; not a new segmentation result.
CPU reference-value and gradient tests precede deployment. GPU admission and
actual four-image results are recorded separately under results_archive.

The queue depends on SW0195 completion to preserve its priority. It then admits
the immutable completed TRAIN8 model on CPU and requests an exclusively idle GPU.
Native64 gradient comparison and four full production mask pairs run without any
optimizer updates. Completed image pairs resume after a foreign owner joins.

Float64 exponential/matrix/log products preserve source edges above cap16;
they are not clipped early. Final masks must match exactly and values must pass
the registered tolerance. Whole-model optimizer equivalence remains a separate
requirement before any production training can use the candidate implementation.

The current194/195 models, sources and checkpoints are unchanged.

2026-10-17 user scheduling update: the original unstarted waiter was stopped
gracefully and preserved. `queue --ready-now` uses `queue_all_gpus` and removes
the optional195 dependency. It scans all four GPUs for an exclusive idle lease.
The operator, model, numerical gates and benchmark cohort remain unchanged.

Completed four-image GPU comparison, independently checked against actual saved
mask pairs. Whole predictions133.789s ->80.800s, speedup1.65581; geodesic portion
72.573s ->3.434s, speedup21.1328. Dense z-gradient relative error2.063e-6,
cosine .999999999998 passed the registered local gradient gate.

Production gate FAILED: all four Q arrays failed the registered forward tolerance
(max errors .8748,.2788,.0332,.5916), although distance/adjacency passed and graph
support was identical. Two of four final masks differed; the mismatches also
changed foreground support, so they are not merely different instance IDs.
This is evidence that local operator/gradient agreement was insufficient to
preserve the full recurrent spike/readout computation. No optimizer-equivalence
claim, no GT performance improvement claim, and no production training adoption.
The original operator remains in199 matched training.

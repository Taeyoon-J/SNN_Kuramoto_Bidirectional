Native64 paired readout calibration; three independent seed queues scanning GPUs0..3. Pending actual execution. See DESIGN_KO.md.

Completed: fixed8 development images per seed, independent seeds0/1/2,
4096 patches, no optimizer steps. All48 old/new masks and CPU model admissions
independently audited; maximum metric discrepancy2.22e-16.

| Three-seed mean, same eight IDs | FG-ARI | Foreground IoU | Object IoU |
| --- | ---: | ---: | ---: |
| original product Q | .8946055391 | .5522615349 | .7391562767 |
| geometric consensus units | .8952023021 | .5522615349 | .7389540016 |
| comparable Slot | .7978265641 | .1805270567 | .2059092105 |

FG change+.0005967631, paired image CI[.0001360351,.0011160475].
Despite a small positive difference, the registered+.01 gain criterion fails,
and object IoU mean declines. No adoption or exponent/threshold micro-rescue.
This is a readout diagnostic, not newly learned weights. Our source models are
mask-supervised; this is not an unsupervised benchmark win. Eight images are not
the full320 evaluation, nor the previous32-image mini. Full320/three-seed .90
remains unverified. See aggregate_v1/summary.json under results_archive.

# SW0214 source-component relation correction

Keep the original core partition and learn which of its components to merge.
No patch reassignment within a source component, foreground change, center
detector or GT object count at inference. Cached actual spike/membrane features
from211 are reused; failed211–213 classifier weights are not used.

One full-batch128 Adam endpoint, symmetric548-dimensional pair features,
soft TRAIN object-overlap targets and complete-link merging. Zero output is
exactly the old mask. Geometry-only is a frozen input control. See DESIGN_KO.md.
Native64/seed2/TRAIN32/DEV16 supervised pilot; full320 .848615 reference preserved.

Eight local/eight Linux tests and actual6648-pair/native64 CPU admission passed.
Supervisor2350264, actual GPU1 Adam128 completed. Independent48-endpoint audit
passed. Primary .720607/.445474/.580461 versus original .788414/.445474/.674335:
gate failed. Frozen geometry .814008/.445474/.632410 has FG CI crossing0 and lower
ObjectIoU, so is an unconfirmed FG candidate, not a new best. See RESULT_KO.md.

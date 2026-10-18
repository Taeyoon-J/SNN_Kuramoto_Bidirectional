# SW0216 foreground-conditional relation supervision

Same source core/cache/TRAIN32/DEV16, same548 features/weights/MLP/seed/128 Adam
as214. Only TRAIN relation target is conditioned on true foreground. The215
TRAIN-derived odds correction is applied to the new target statistics. Evaluation
contract is unchanged. See DESIGN_KO.md.

Component interiors and source foreground remain fixed. Compare with frozen215
all-patch target endpoint. Posthoc GT-assisted merger references are explicitly
not predictions or certified ceilings. No new core forward or core/encoder update.
Five target tests plus inherited merger/alignment tests cover background dilution,
mixed objects, identity, complete-link/ID invariance, actual Adam resume and GT gate.
Registration and verified outcomes will be under results_archive.

Completed after17 local/17 Linux tests and actual6648-pair/native64 admission.
Primary .781651/.445474/.665117 versus original .788414/.445474/.674335:
no improvement. Matched215 FG unchanged. Exact input/weight/initialization match,
actual CUDA Adam128 and64 endpoints independently audited. GT-assisted merge
references .847321/.849997 are not predictors or certified upper bounds.
No further small target/threshold sweep. See RESULT_KO.md. Full320 reference stays.

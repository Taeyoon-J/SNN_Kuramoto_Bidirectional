# SW0222

Prospective three-seed native64 teacher-denoising model test. Existing source
object-frequency means and gamma are frozen TRAIN teachers. No arbitrary
inter-object frequency margin; original core/gates/readout retained.
See DESIGN_KO.md. No segmentation result claimed before execution.

Actual start2026-10-18:9 local/9 Linux tests,3 local CLI checks and all3 actual source/TRAIN/teacher CPU admissions passed. Frozen teacher initial loss matches source within-object variance; original gamma anchor is zero. Three exclusive idle GPU queues registered; START_EVIDENCE.json records actual NVIDIA-bound workers. No segmentation improvement claimed before all three final endpoints freeze and score. Full320 incumbent .848615/.592533/.672047 unchanged.

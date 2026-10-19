# SW0230: source-relative class-center separation

Prepared matched TRAIN64/128-update encoder+projection comparison against SW0223.
Original center weight1 retained; add a squared one-sided source-distance deficit,
normalized by source distance. No new absolute margin or center16 anchoring.
No model performance result yet. Frozen16/three-seed admission and candidate gates
are specified in DESIGN_KO.md. Source targets use TRAIN masks; not unsupervised.

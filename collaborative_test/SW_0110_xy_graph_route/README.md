# SW0110: absolute XY in graph node features

This preregistered matched test adds a zero-initialized 16x2 XY projection before graph feature normalization. The existing distance prior remains fixed. Native image features, oscillator drive, actual spike classifier and evaluation contract remain unchanged. Candidate and control start from each whole SW0097 checkpoint and train for 256 matched updates; all three seeds receive full320 evaluation.

`protocol.json` is the frozen scientific contract. Run only after implementation review and real first-four-TRAIN-batch preflight. A zero coordinate projection must reproduce the legacy forward, and coordinate gradients must be finite and nonzero. No GT-selected coordinate strength or alternate readout is allowed.

Promotion requires the registered FG-ARI margin against both the newly continued matched control and the original SW0097, with IoU floors and paired-image uncertainty checks. A failed candidate does not block unrelated queued tests.

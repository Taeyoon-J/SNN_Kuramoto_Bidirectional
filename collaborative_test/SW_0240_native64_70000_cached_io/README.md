# SW0240: exact input-cache successor to SW0239

This continues the requested best-model training on 70,000 unique TRAIN images, followed by the fixed 320-image native64 evaluation and predetermined mask figures. It is an I/O correction, not a new model or loss hypothesis.

The original gzip HDF5 chunks span 3,125 images and small spatial regions. Random batches repeatedly decompress large chunks. The complete canonical TRAIN cache preserves RGB and modal labels, split, seed shuffle, teacher, optimizer, update budget and readout. Reserved evaluation masks are excluded from cache construction. Two actual original/cache batches match bitwise. The measured 3,424–4,666-fold improvement applies to input reads only.

Seven local and seven Linux tests passed. A real Adam interruption/rebind test matches uninterrupted model, encoder and history hashes. After Linux tests passed, only the verified owned239 supervisor and worker were stopped. Their final committed seed0 boundary at update128 was preserved, including model, Adam, RNG and complete TRAIN history. Seeds1/2 have not started. All three actual CPU admissions must pass before a single successor GPU worker can start.

Each seed still completes 43,750 logical updates and ten full passes through the same 70,000 unique images; inherited updates count toward that total. Seeds run sequentially on one GPU. Other experiment queues remain independent. The migration manifest and checkpoint bytes are private generated artifacts; aggregate verification is recorded in `MIGRATION_VALIDATION_EVIDENCE.json`.

All three actual CPU admissions subsequently passed. On the assigned GPU, inherited Adam128 resumed and committed Adam192, confirming 64 new successor updates. The final CPU raw-checkpoint/mask audit is attached. No new performance result is available yet. SW0227 remains the audited full320 incumbent. This training uses mask supervision and should not be presented as unsupervised learning.

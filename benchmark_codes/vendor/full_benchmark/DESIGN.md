# SW0255 random-initialized full-training reconstruction

No learned240/165/252/254 weights, feature statistics or old activity cache may
initialize the new scientific training. Dataset-specific statistics must be fitted
on the registered35,000 TRAIN images using that run's own initial encoder.
The actual full data order and10 complete passes are shared within each dataset/seed.

All variants activate the existing94/95 unsupervised core objective:
phase PLV objective +5 times the positive component-spike objective, with
bimodality6, balance10, coherence0.5, collapse1, density0.867. This gives graph,
Kuramoto, dendrite, membrane and integration parameters an actual training path.
The supervised variants additionally use the existing240 frequency denoising and
gamma anchor objective. Its detached teacher is a shadow of the run's OWN random
initialization, not a previously trained CLEVR model. This is a from-scratch
full-training recipe; previous partial-training scores are not its results.

| Variant | TRAIN mask frequency/anchor | Final CE+Dice | Final-mask activity gradient |
|---|---|---|---|
|240|yes|none|none|
|252|yes|yes, all regions including background exchangeable|detached at head input|
|253/165 unsupervised family|no masks/teacher|none|none; core learns unsupervised|
|254|yes|yes, all regions exchangeable|live through the whole active body|

252/254 share forward architecture but NOT backward routes. Body and head use
separate Adam/clipping, preventing head gradients from silently rescaling body
updates in252. Potential252/240 common body training must be proven identical
before sharing; no sharing is yet certified.253 retains its native32 architecture;
the other models are native64. Report this resolution difference in pixel results.

All variants use the same data-independent zero initialization for the shared
dendritic projection bias; all projection weights retain their fresh random
draws. Default random biases silenced all spikes in the initial native64 audit.
The zero-bias policy is fixed for every model/seed, not selected using validation
or test scores. No new auxiliary loss was added to hide the inactive path.

The original fixed8 squared-geodesic T operator remains a fixed architectural
constant; its legacy unused contrast Parameter is explicitly excluded, not called
learned. All active encoder, graph, Kuramoto, dendrite, membrane, sensory projection,
integration and existing learned head groups must show finite nonzero gradients
and actual weight updates. Sinusoidal gating itself has no independent Parameter;
its gradient route and associated integration weights are audited.

The graph's dense soft-min-plus relaxation is evaluated using the SAME algebra:
`-tau*log(exp(-D/tau) @ exp(-D/tau))`, with row/column shifts for numerical stability.
All intermediate nodes remain; this is not a sparse/local replacement. Value AND
derivative agreement against the original logsumexp operator is checked. Float32
TF32 acceleration is disabled for numeric consistency. Small rounding differences
are not described as bitwise identity with old checkpoints.

Actual forward is1024steps and settled features512..1023, with truncated BPTT over
the last64steps and live graph construction derivatives. This is full active
parameter learning, NOT full1024-step temporal BPTT. Hard partition/Hungarian/argmax
remain discrete; no gradient through those decisions is claimed.

The failed first audit used diagnostic identity normalization. Subsequent
admissions fit the exact35,000 TRAIN initial-encoder statistics without masks.
Two-step checkpoints remain diagnostic ONLY, never reused as a35000 training run.
Actual CUDA data/loss/backward/save/resume/evaluation
admission and input/split validation must finish before scientific jobs launch.

Priority phase1:252,253,254,240 each on CLEVR/Multi-dSprites/Tetrominoes, with
evaluation complete. Phase2:ObjectsRoom/CLEVRTex. Preserve protected benchmark and
all prior model checkpoints. COCO has been stored but is outside this matrix.

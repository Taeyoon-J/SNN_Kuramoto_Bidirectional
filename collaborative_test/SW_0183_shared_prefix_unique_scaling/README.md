# SW0183 — shared-prefix unique-data comparison

Preregistered successor to the SW0182 technical reproduction stop. Each seed
executes one 32-proposal common prefix, then forks the exact model/Adam/RNG state
into 32-proposal repeat512 and new512 branches. This yields two logical
64-proposal arms with 96 proposals of actual work per seed.

See [the registered design](DESIGN_KO.md). Implementation is in progress;
no SW0183 GPU training or evaluation has started. The SW0181 incumbent is intact.
This is supervised continuation, not a fresh native32/70k or unsupervised result.

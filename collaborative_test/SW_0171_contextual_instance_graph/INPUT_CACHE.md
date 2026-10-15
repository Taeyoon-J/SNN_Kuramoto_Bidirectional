# SW0171 verified reusable TRAIN inputs

2026-10-15. The CPU-only producer3736173 completed normally in374.99 seconds.
It materialized the same original-source TRAIN512 RGB, source foreground support,
feasible labels and canonical full modal32 labels once for both comparison arms.
It did not initialize a candidate, run CUDA or update parameters. No images or
labels were generated or changed, and validation/reserve rows are absent.

Root independently re-read all512 canonical HDF5 RGB/mask rows, recomputed their
modal32 and feasible targets, and compared every cached row. All data hashes,
schema, original source order, producer code/dependencies and support bindings
matched. The actual arrays remain exclusively on Frontier. Git contains only
the producer text and validation metadata under input_cache_ready_20261015.

Consumers can validate immutable bytes/source/HDF identity and reuse these exact
inputs; they should not recompute the full CPU QCC cache separately for every arm.
The producer's SW169 metadata describes data provenance only. SW171 uses original
SourceT model dynamics, not SW169 learned parameters, slow membrane or hard reset.

Manifest SHA256:
`388e8748f5ebf7e8155e95b0651ddc59f239a467ea9351f26d2f6b16cebb8a67`.
Arrays SHA256:
`4006f191baff3b9e1c9effc8ec67421c011726a5b02b05fdb8c9a432499819ba`.
This is verified input preparation, not a SW171 segmentation score or GPU test.

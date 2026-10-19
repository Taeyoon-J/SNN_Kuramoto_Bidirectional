# Matched-layout control completed

All six original actual masks and all six original contiguous gate-only masks replayed bitwise. The gate-only substitution was then recomputed in the exact native component-spike strides. Both seed1 negative controls, whose actual post-settle spikes equal their gates, retained bitwise-identical Q and masks.

The matched-layout substitution produced the same masks as the original substitution in all six cases. Four cases retained their actual masks; the same two cases changed. Seed0/case0 changed three foreground-support patches and seed2/case1 merged two groups while retaining foreground support. Therefore layout did not explain the observed changes in these six cases. Four cases had changed Q after event removal, but only two changed the final partition.

This is controlled evidence that the emitted event information can affect downstream grouping on these sampled TRAIN inputs. It does not show that either changed mask is more accurate, prove the necessity of membrane recurrence, or establish contributions for all gating layers. No GT, new trajectory forward or optimizer update was used. Full membrane/dendritic recurrence remained unchanged upstream.

The original241 experiment and limitation record remain preserved. This auxiliary control resolves that layout concern for the six observed cases. Saved paired masks and source bindings were revalidated at collection. The current full320 model reference remains SW0237 at0.872680 /0.626110 /0.676331; the requested70k training continues independently.

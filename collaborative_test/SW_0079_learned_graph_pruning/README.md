# SW0079 - prune ambiguous edges in the learned graph

SW0065 showed that the trained image-conditioned graph is causal for seed2's
grouping failure. SW0078 tests replacing part of that graph with a fixed RGB
bilateral graph. SW0079 asks a narrower question: does the learned seed1 graph
benefit from fewer low-confidence edges, without adding any RGB or changing
downstream dynamics?

The pilot runs the frozen SW0072 seed1 checkpoint on aligned validation IDs
1320-1351, T256/settle64, graph decay `.35`, geodesic settings 3/1.5/2/.5/16,
factorized rollout, vth `.06`, raw gate, and the same per-component spike CC
readout at threshold `.35`, min group size2, largest component as background.
The three rows are the exact checkpoint default top_k32 path and graph-generator
recomputations at top_k16 and top_k8.

Only the graph generator's plain `top_k` attribute changes during each
adjacency recomputation; a `finally` block restores it to 32 before the core
rollout. Projection, temperature, selected-edge logits, coupling gain,
softmax normalization, symmetrization, gamma, checkpoint tensors, all dynamics,
and classifier remain fixed. Reducing top-k also changes which logits survive
and how softmax mass is distributed among them; this is the intended pruning
intervention. The graph generator's parameters/state dict are not updated. No
masks, counts, or validation labels enter graph creation or parameter choice.

The top-k values are coarse structural settings chosen before scoring. The
test can fail if useful object edges are among the removed edges, if removing
edges disconnects objects, or if re-normalization increases the weight of
remaining misleading edges. Group count and all three mask metrics are
reported; the pilot alone does not warrant promotion.

Run on an idle GPU0 or GPU1:

```bash
bash collaborative_test/SW_0079_learned_graph_pruning/evaluate.sh 0
```

The script refuses existing result/log files. No server evaluation has been
launched; shared INDEX and STATUS files are intentionally unchanged.

## Result

| seed1 32-image pilot | FG-ARI | foreground IoU | matched-object IoU |
|---|---:|---:|---:|
| learned top-k 32 | 0.708757 | 0.468105 | 0.509881 |
| learned top-k 16 | 0.635492 | 0.460102 | 0.461217 |
| learned top-k 8 | 0.614964 | 0.427776 | 0.445628 |

Both pruning interventions lower all three metrics, and the stronger pruning is
worse. Predicted group count stays near `5.7`, so this is not primarily a count
shift; the same number of components have poorer object membership. Apparently
weak learned edges help Kuramoto synchronize complete objects. Stop inference
graph surgery: subsequent graph work must change how edge quality is learned.

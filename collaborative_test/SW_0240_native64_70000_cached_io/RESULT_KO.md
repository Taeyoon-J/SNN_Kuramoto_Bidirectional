# SW0240: native64 supervised full70k result

All three seeds completed 70,000 distinct TRAIN images, ten passes and43,750
new optimizer updates per seed. Evaluation uses the same fixed320 validation
images and native64 hard masks as the previous reference.

| Model | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| SW0240 three-seed mean | **0.888130** | **0.743444** | **0.684382** |
| Previous SW0237 mean | 0.872680 | 0.626110 | 0.676331 |
| Comparable Slot Attention | 0.786902 | 0.198201 | 0.195203 |

| Seed | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| 0 | 0.895956 | 0.828796 | 0.700791 |
| 1 | 0.894247 | 0.746766 | 0.688654 |
| 2 | 0.874186 | 0.654772 | 0.663702 |

All three means exceed Slot Attention; seed2 individually remains below Slot
seed2 in FG-ARI. The three-seed FG-ARI0.90 target is not met.

This uses TRAIN mask supervision; Slot uses reconstruction training. It is not
an unsupervised comparison. The source selected at registration was SW0227;
the subsequently completed SW0237 did not silently replace the initialization.
Only the encoder and sensory projection are adapted; other core/readout remain
unchanged. Neither all-gating causality nor a controlled data-size effect is
established: data amount and optimization budget both changed. These repeatedly
used validation images are not an untouched final test.

The user-requested shared GPU successor changed resource admission only; the
training objective, source fingerprint, Adam/history and masks were preserved.
Raw score and independent audit receipts are retained privately; public
aggregate evidence is in PUBLIC_RESULT.json once the independent audit passes.

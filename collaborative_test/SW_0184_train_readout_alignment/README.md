# SW0184: TRAIN-only loss/readout alignment diagnostic

Started after the negative unique-data comparison in SW0183. Both frozen183
models are replayed on sixteen preregistered TRAIN images per seed: eight old,
eight new. Actual spikes generate Q/QCC and SW165 masks, while TRAIN labels are
used only for diagnostic measurements. No optimizer steps or validation reads.

Eight tests, the actual queue CLI and six actual checkpoint CPU loads passed.
Three seed supervisors are registered; an actual GPU worker was observed live.
See [design](DESIGN_KO.md) and [CPU/launch evidence](results_archive/start_v1/).
No new fixed320 score or incumbent promotion is implied.

# SW_0003 — matched-duration membrane PLV

Status: training pending.

The SW_0002 pilot changed PLV source and used only 10 epochs, while the starting
checkpoint used 40. This follow-up repeats SW_0002 at seed 0 for 40 epochs to
isolate the source change at matched training duration. All other settings are
copied verbatim from the seed-0 baseline. See `run.sh`.

Compare the original spatial-components classifier and the SW_0001 fixed-k=8
spike-synchrony classifier on validation IDs 1320–1639, with the same patch
metrics. No test split is used for selection. Training loss alone is not a
success criterion.

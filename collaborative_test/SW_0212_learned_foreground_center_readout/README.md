# SW0212 learned foreground with nonlocal center assignment

Prospective follow-up to211, registered without reading its scores. Reuses only
the frozen primary128 head. Its learned foreground (logit>=0) replaces the old
spike foreground; the center/offset decoder is unchanged. No optimizer updates,
extra GPU lease, threshold sweep or new training data.

All16 original-support CPU predictions must exactly replay211 saved masks before
new masks may be scored. GT is read only after all new predictions are frozen.
Both IoUs, FG-ARI, counts and fallback are reported, including negative results.
See DESIGN_KO.md. Registration and test receipts will be under results_archive.

Registered before211 scores existed: five local and five Linux tests passed,
plus actual268-channel native64 CPU head/decoder admission. CPU dependency
supervisor2285616/start448185348 is waiting for211; no new training has started.
The first deployment bootstrap encountered system Python3.6's unsupported
capture_output argument before tests or model execution. Using the established
research Python3.10 fixed the deployment helper; scientific source was unchanged.
Original bootstrap logs are retained locally. This is not a failed GPU trial.

# SW0241 posthoc matched-layout control

Keep the original registered241 code and results unchanged. Reuse all six retained spike/gate/adjacency arrays. Rebuild actual component-spike strides using the exact stack-time-first, permute and reshape operations from the registered native rollout; gates and adjacency retain their canonical contiguous storage.

Before interpreting a new substitution, the identical production readout must reproduce both the original actual masks and the original contiguous gate-only masks bitwise. Copy gate values into storage with the actual spike strides, rebuild mean spikes and component correlations/Q, and run the production readout again. When the actual post-settle spikes equal the gates, require exact Q and mask equality as a negative control.

No optimizer update, GT input or native trajectory forward is added. The same three accepted232 models, six241 TRAIN cases, software/hardware math flags and downstream readout apply. Each seed has a fresh CPU restore admission and runs under the existing exclusive-GPU lease. The requested70k training is independent. All new source files are auxiliary and outside the frozen original241 fingerprint; a separate fingerprint binds this control to the original experiment and the rollout storage implementation.

Retain paired masks, exact replay checks and actual/legacy/matched strides. Aggregate only after all three control protocols validate. This can support a statement about event information at readout for these six cases. It does not remove membrane recurrence, prove population-level necessity, produce held-out performance scores or replace the current237 full320 reference.

# SW0125 actual foundation result

All3 original same-seed SW0097 B16 TRAIN probes passed native1024 forward parity on2026-10-09. Every phase, component/mean spike/membrane trace and registered512-band phase/Q/analyticRGB forward loss matches exactly. All6 parameter families receive finite nonzero RGB credit through the last64 live steps. No optimizer updates, masks, or evaluation scores were used.

The firstGPUattempt failed only the averaged-membrane exact guard while everycomponent trace matched. It is preserved in original_foundation_failure_20261009.json and server results_archive/foundation_attempt_original_20261009/. The corrected helper uses native stack/permutation/reshape strides and unchanged strict equality guards. Corrected attempts and all3supervisor/childPIDs are terminal rc0; full reports and hashes are in completed_foundation_proofs_20261009.json.

Membrane gradient norms are6.38e-7/2.78e-7/1.40e-6; encoder norms .0367/.1378/.0417. Nonzero credit alone does not establish useful gate contribution, expected parameter changes, model improvement, or data scaling. Adam update and complete256-update training/evaluation remain pending; existing97 incumbent unchanged. Do not infer metric gains or full1024-step BPTT from this probe.

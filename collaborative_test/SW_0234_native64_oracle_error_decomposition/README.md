# SW0234: GT-assisted error decomposition, not a model score

Peer review at origin/patch_v2 commit087cd82 found PV2_0104 three-seed native32
contract-v2 validation300 FG .7109; oracle merge .8242, coverage .8111,
both .9329. These use ground truth and are not trained-model improvements.
Their Slot contract and dataset differ from ours. No scores are imported.

Transfer the diagnostic question to our independently audited SW0227 native64
full320 masks: apply prescribed merge-only, coverage-only and combined GT-assisted
operations on CPU. Preserve pure-background groups in merge-only so foreground
support and foreground IoU remain identical. Coverage fills only GT foreground
currently predicted background. It can invent a group for entirely missing objects.

These operations do not prove a mathematical upper bound or that a learned readout
can recover the improvements. Existing straddling groups are not split. No mask,
checkpoint, loss, GPU queue or best-model record changes. No model forwards.
Prior SW0214–216 learned coarse-merger failures remain evidence; a high oracle
result alone does not justify repeating that approach. Use this diagnostic to
prioritize remaining error types after SW0232's convergence comparison.

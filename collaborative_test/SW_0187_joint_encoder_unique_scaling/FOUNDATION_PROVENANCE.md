# Foundation training budget clarification

Source097 inherits a core that had already trained on the complete 70,000-image
RGB pool. Its final 4,096-image stage is additional training, not its complete
training history. The server receipt verifies all three actual Source097
checkpoint hashes, their Source095 parents and the Source090 epoch-1 parents.
Source095 manifests record 4,375 batches of 16 and 70,000 distinct TRAIN IDs;
Source097 manifests record another 256 batches of 16. Both stages record no
ground-truth masks used for training.

Consequently, SW0186's TRAIN512 twice and SW0187's repeated512 versus unique1024
refer only to the additional supervised native32 adaptation stage. SW0187 tests
the benefit of distinct adaptation images and labels at equal proposal budgets.
It does not test whether the inherited model benefits from seeing previously
unseen RGB images, nor whether total foundation data increased from 512 to 1,024.

The documented downstream lineage is Source090 epoch1 (70,000 exposures),
Source095 (another 70,000), then Source097 (4,096). Earlier encoder/graph
pretraining has a separate budget. Later Source090 epochs are not included in
this inherited checkpoint chain. A new native32 joint-encoder 70,000-image
training pass remains outstanding.

Evidence: [three-seed server receipt](results_archive/foundation_provenance_v1/foundation_provenance_receipt.json),
SHA256 `87b812f2d4256c9805640c2f5cdcbb0d7b28b32b2a2f2a5562d312c65d0d0e10`.
No validation masks were read for this provenance check. This clarification
changes no registered model, training, prediction or scoring source.

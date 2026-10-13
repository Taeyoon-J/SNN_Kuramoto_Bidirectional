# Peer incorporation: c493a0d (2026-10-13)

Reviewed origin/patch_v2 commit c493a0dfbbce7595966efe5dd7eeab13f2b46556, REVIEW_0005, PV2_0070_crossview, PV2_0058_chromaticity, and PV2_0088_grid32_enablers.

PV2_0070 reports cross-view PLV consistency becoming stronger while FG-ARI declines: baseline .7239, best jitter arm .6757. This is adverse evidence against treating invariance or a lower consistency loss as proof of object grouping. SW0152 differs: direct emitted-spike partition JS, a jointly learned spatial encoder/graph/core, source mass retention, horizontal-flip alignment, native32 evaluation, and a two-view matched mass-only control. These differences justify finishing the registered pilot, not presuming it succeeds. Do not broaden an unsuccessful pilot into coefficient sweeps or 70k training.

PV2_0058 also reports FG-ARI improvements accompanied by severe foreground overprediction and worse foreground IoU. Retain all three fixed metrics and predicted foreground diagnostics. Neither loss, feature margin, nor increased predicted foreground alone is a promotion result.

PV2_0088 adds row-chunked checkpointed geodesic relaxation with reported exact forward parity and matching printed gradient norms. Its contracts use original 240x320 mask geometry and a different split; ours remains 128x128/native32/fixed320. Do not import its target generation or gamma standardization into the frozen evaluation contract. Assess the graph memory optimization separately against our selective-geodesic implementation before any adoption. Complexity counts are theoretical, not measured wall-clock speedups.

REVIEW_0005 sees our remote branch at 94de778 and thus misses SW0140-SW0152 local records. This is a publication lag, not absence of local experiments. Expanded GitHub publication remains pending approval; local commits continue to preserve evidence. No message was sent to the peer and no remote push was performed in this review.

Current pilot remains unchanged. No new performance result or incumbent promotion follows from this review.

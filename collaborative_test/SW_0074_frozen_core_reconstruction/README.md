# SW0074 - frozen-core reconstruction feature tuning

SW0073 shows that reconstruction-held features contain useful new grouping
information, but joint core updates erase the gain. Its weight-1 encoder in the
unchanged SW0072 core improves FG-ARI and matched-object IoU while losing
foreground IoU. SW0074 freezes the complete SW0072 seed1 core and updates only
the native RGB encoder through the same PLV/spike/reconstruction path.

Two coarse gamma anchors, `10` and `100`, test whether limiting feature drift
recovers foreground localization without losing the grouping improvements.
Both use reconstruction weight `1`, seven slots, encoder LR `3e-5`, five
epochs, the same 2,500 scenes, and no masks or counts. Core parameters must be
bitwise unchanged. An arm advances only if all three fixed seed1 metrics exceed
SW0072.


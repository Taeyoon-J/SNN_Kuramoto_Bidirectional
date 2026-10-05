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

## Result

No arm advances. Anchor10 scores `.67210/.44354/.50297`; anchor100 scores
`.68271/.41791/.50016`, both below SW0072 seed1
`.70876/.46810/.50988`. Core and graph changes are exactly zero. Anchor100
reduces validation gamma RMS drift to `.01004` from anchor10's `.02499`, but
foreground IoU is worse, so total gamma proximity is not a useful proxy for
preserving the foreground classifier. Stop anchor-weight refinement. The next
loss must preserve the baseline's patchwise downstream activation directly.


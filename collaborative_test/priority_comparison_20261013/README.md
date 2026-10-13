# Verified comparison, 2026-10-13

All scores are FG-ARI / foreground IoU / matched-object IoU. Rows with measured three-seed means are distinguished from the single released Slot checkpoint.

| Training / evaluation | Grid | Ours | Slot Attention | Scope |
|---|---|---|---|---|
| ours / ours | 16x16 | 0.7814 / 0.4780 / 0.6167 | 0.7749 / 0.2036 / 0.2069 | SW0072; 2500 training budget |
| ours / ours | 16x16 | 0.7763 / 0.5771 / 0.6015 | 0.7749 / 0.2036 / 0.2069 | SW0097; 70000 lineage plus4096 continuation |
| ours / ours | 32x32 | 0.7830 / 0.6100 / 0.5962 | 0.7817 / 0.2044 / 0.2032 | SW143T inference graph; SW97 native16-trained source transferred to32, not fresh32 training |
| official70k additional continuation / ours | 16x16 | 0.7694 / 0.5324 / 0.5919 | 0.8946 / 0.2235 / 0.2460 | SW107; inherited ours-trained SW97, not exclusive official training |
| official / ours | 32x32 | Not verified | Not verified | not verified |
| official / official | 16x16 | Not verified | Not verified | not verified under our matched evaluation contract |
| official / official | 32x32 | Not verified | Not verified | not verified under our matched evaluation contract |
| ours / ours | 64x64 | Not verified | 0.7869 / 0.1982 / 0.1952 | Slot baseline completed; own native64 pending |

The .894641 released Slot result is evaluation on our images, not official-domain evaluation. SW107 continues an already ours-trained model. Neither SW107 nor SW143T is an exclusive official-domain or fresh native32 experiment. No contract/split/gamma recipe changes were used to fill missing cells.

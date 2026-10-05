# SW0062 - DINOv2 patch features with core adaptation

Peer feature-curve diagnostics attribute `.170` FG-ARI to reducing
within-object feature variation and rank label-free self-supervised ViT patch
features as the strongest next intervention. SW0061 showed that direct feature
substitution into a core trained on another channel basis fails. SW0062
therefore freezes a public DINOv2 encoder and trains a fresh SNN/Kuramoto core
on its patch features.

The feature asset uses `vit_small_patch14_dinov2.lvd142m` at 224x224, yielding
a 16x16 grid of 384-dimensional patch tokens. PCA is fitted only on training
IDs0-999 and reduces tokens to eight normalized channels so the downstream
architecture remains unchanged. Validation IDs1320-1639 are transformed with
that frozen training-only basis. No masks, object counts, or labels enter
feature extraction or training.

The first arm is a seed0 five-epoch direction pilot on the same 1,000 training
images and original mixed phase plus component-spike loss. A one-update real
asset preflight and a 32-image fixed short evaluation precede any longer or
multi-seed run. If the direction is promising, it advances to the matched
25-epoch three-seed contract.

The direction pilot completed at `.612146/.182948/.363269`, below the native
epoch25 reference `.668520/.692026/.489879`. A post-hoc feature diagnostic
explained the failure: normalized DINO-PCA8 features had within/between object
centroid ratio `1.018`, much worse than native gamma `0.225`. Full 384D output
tokens measured `0.774` on 32 images, while last-layer attention keys measured
`0.919`; neither provides the required CLEVR object consistency. The generic
DINO direction is stopped before longer training. Reports are in `results/`.

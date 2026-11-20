"""Three additive edits to the ctcae clone, each with a stated reason. Backups are written.

1. data.py: CLEVR resolution 96 -> 64.
   Its 96 does not divide our grid 64 (96/64 = 1.5), so a prediction at 96 could not reach the
   grid by the contract's exact lift or its patch majority vote. At 64 the prediction *is* the
   grid and nothing is resampled.

2. data.py: drop the CLEVR crop.
   `crop((64,29,256,221))` assumes the 320x240 EMORL render. Our frames are already the 128x128
   square render this whole project uses, so that crop would cut most of the image away and pad
   the rest with black.

3. train.py: add 64x64 profiles for CLEVR and Tetrominoes.
   The repo ships a `*_dsprites_64x64` architecture (one extra stride-1 and stride-2 layer, and
   a patched decoder) for running at 64 instead of 32. We reuse that same architecture for the
   other two datasets rather than inventing one, so the only thing that differs between our
   three columns is the data.
"""
import re, shutil, pathlib, sys

R = pathlib.Path('/work/USERS/tkim1/baselines/_clone/ctcae')

# ---------------------------------------------------------------- data.py
p = R/'data.py'
if not (R/'data.py.bak').exists():
    shutil.copy(p, R/'data.py.bak')
s = p.read_text()

old = '''DATASET_IMG_RESOLUTION = {
    "clevr": (96, 96),  # Note(astanic): (96, 96) is the original clevr resolution in EMORL
    "multi_dsprites": (64, 64),  # Note(astanic): (64, 64) is the original dsprites resolution in EMORL
    "tetrominoes": (32, 32),
}'''
new = '''DATASET_IMG_RESOLUTION = {
    # patch_v2: every dataset runs at 64x64, which is our evaluation grid, so a prediction never
    # has to be resampled to be scored. CLEVR's 96 does not divide 64; tetrominoes' 32 would be
    # an exact x2 lift but we keep one resolution across the three columns.
    "clevr": (64, 64),
    "multi_dsprites": (64, 64),
    "tetrominoes": (64, 64),
}'''
assert s.count(old) == 1, 'DATASET_IMG_RESOLUTION not found verbatim'
s = s.replace(old, new, 1)

# the image crop
old = """        # square center crop of 192 x 192
        elif self.dataset_name == 'clevr':
            PIL_img = PIL_img.crop((64,29,256,221))
            PIL_img = PIL_img.resize(self.img_resolution)"""
new = """        # patch_v2: no crop. crop((64,29,256,221)) assumes the 320x240 EMORL render; our frames
        # are already the 128x128 square render used everywhere else in this project.
        elif self.dataset_name == 'clevr':
            PIL_img = PIL_img.resize(self.img_resolution)"""
assert s.count(old) == 1, 'clevr image crop not found verbatim'
s = s.replace(old, new, 1)

# the mask crop
old = """            elif self.dataset_name == "clevr":
                # square center crop of 192 x 192
                PIL_mask = PIL_mask.crop((64,29,256,221))"""
new = """            elif self.dataset_name == "clevr":
                # patch_v2: no crop, for the reason given above. Our masks are written at 64x64
                # already, so this resize is an identity and label planes are never interpolated.
                pass"""
assert s.count(old) == 1, 'clevr mask crop not found verbatim'
s = s.replace(old, new, 1)
# the two resize lines that followed the mask crop must still run
old = """                pass
                # This resize was added such that we don't have to upsample predicted image in the evaluation"""
new = """                pass
                # This resize was added such that we don't have to upsample predicted image in the evaluation"""
assert s.count(old) == 1
p.write_text(s)
print('data.py patched')

# ---------------------------------------------------------------- train.py
p = R/'train.py'
if not (R/'train.py.bak').exists():
    shutil.copy(p, R/'train.py.bak')
s = p.read_text()

ARCH = """        'use_32x32_res': 0,
        'enc_n_out_channels': '128,128,256,256,256,256,256',
        'enc_strides': '2,1,2,1,2,1,2',
        'enc_kernel_sizes': '3,3,3,3,3,3,3',
        'dec_use_patches': 1,
        'dec_patch_size': 2,"""

block = """    ################################################################
    # patch_v2: 64x64 profiles for CLEVR and Tetrominoes, reusing the repo's own
    # *_dsprites_64x64 architecture so only the data differs between our three columns.
    ################################################################

    parser.Profile('ctcae_clevr_64x64', {
%(arch)s
    }, include='ctcae_clevr'),
    parser.Profile('caepp_clevr_64x64', {
        'loss_coeff_contrastive': 0,
    }, include='ctcae_clevr_64x64'),
    parser.Profile('cae_clevr_64x64', {
        'loss_coeff_contrastive': 0,
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'decoder_type': 'conv_transpose',
    }, include='ctcae_clevr_64x64'),

    parser.Profile('ctcae_tetrominoes_64x64', {
%(arch)s
    }, include='ctcae_tetrominoes'),
    parser.Profile('caepp_tetrominoes_64x64', {
        'loss_coeff_contrastive': 0,
    }, include='ctcae_tetrominoes_64x64'),
    parser.Profile('cae_tetrominoes_64x64', {
        'loss_coeff_contrastive': 0,
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'decoder_type': 'conv_transpose',
    }, include='ctcae_tetrominoes_64x64'),

""" % {'arch': ARCH}

anchor = """    ################################################################
    # CLEVR full resolution (96, 96) profiles
    ################################################################"""
assert s.count(anchor) == 1, 'CLEVR 96x96 section header not found'
s = s.replace(anchor, block + anchor, 1)
p.write_text(s)
print('train.py patched: added 6 profiles')

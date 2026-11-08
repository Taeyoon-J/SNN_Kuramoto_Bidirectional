#!/usr/bin/env bash
# PV2_0150 -- AKOrN trained on CLEVR only, evaluated on all three. Host frontier, GPU index 3.
# Everything below ran in this order on 2026-11-08. Base commit 17d0fde.
set -euo pipefail

A=/work/USERS/tkim1/baselines/_clone/akorn
C=/work/USERS/tkim1/sw0255b/benchmark_codes
B=/work/USERS/tkim1/baselines
P=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
G=GPU-6801a3dc-cde3-86c6-d665-6d9ecf063a1e        # TITAN RTX, index 3 -- ours
RUN=/work/USERS/tkim1/sw0255b/run
CFG=$C/configs/frontier.json

# ---------------------------------------------------------------- 0. environment
$P -m pip install --quiet scikit-learn tensorboard

# ---------------------------------------------------------------- 1. data
# Our splits in AKOrN's NumpyDataset format. `labels` is OUR grid-64 truth lifted x2 to 128,
# so AKOrN scores itself against the identical tensor evaluate_patch_masks uses.
$P -u $B/akorn_clevr_data.py        # clevr_{train,val,eval320}.npz
$P -u $B/akorn_data128.py           # tetrominoes/dsprites {val128,eval320_128}.npz
$P -u $B/akorn_val_data.py          # grid-64-label variants, for the dSprites-trained row

# AKOrN hardcodes its test paths, so they are symlinks we repoint per split.
cd $A

# ---------------------------------------------------------------- 2. train (CLEVR only)
# Published config; only --epochs changed, 300 -> 10, to match 100,000 exposures.
setsid nohup env CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 \
  $P -u train_obj.py --exp_name=ours_clevr_akorn_attn --data=clevr \
  --model=akorn --J=attn --L=1 --ch=256 --psize=8 --epochs=10 --c_norm=none \
  > $A/akorn_clevr.log 2>&1 < /dev/null &
# ~7 min. 40 batches x 250 x 10 epochs = 100,000 exposures.

# the target-trained comparison row, published config, --epochs 50 -> 10
setsid nohup env CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 \
  $P -u train_obj.py --exp_name=ours_dsprites_akorn_attn --data=dsprites \
  --model=akorn --J=attn --L=1 --ch=128 --psize=4 --epochs=10 --c_norm=none \
  > $A/akorn_dsprites.log 2>&1 < /dev/null &

# ---------------------------------------------------------------- 3. wrap the checkpoint
# train_obj.py writes the final ema as a bare state_dict; eval_obj.py wants it wrapped.
for R in ours_clevr_akorn_attn ours_dsprites_akorn_attn; do
  $P -c "
import torch
d = torch.load('runs/$R/ema_model.pth', weights_only=True, map_location='cpu')
torch.save({'model_state_dict': d}, 'runs/$R/ema_wrapped.pth')"
done

# ---------------------------------------------------------------- 4. predictions
# eval_dump.py is a copy of eval_obj.py with shuffle=False in the generic eval loader (so the
# dumped order matches the id order) and a prediction dump appended. eval_obj.py is untouched.
M="--model=akorn --model_path=runs/ours_clevr_akorn_attn/ema_wrapped.pth --J=attn --L=1 --T=8 \
   --ch=256 --psize=8 --c_norm=none --model_imsize=128"

for S in val eval320; do
  ln -sf $PWD/data/clevr_with_masks/clevr_$S.npz data/clevr_with_masks/clevr_test.npz
  DUMP=$B/akornC_clevr_$S.npz CUDA_VISIBLE_DEVICES=$G $P -u eval_dump.py \
    --data=clevr --data_imsize=128 $M
done

for D in dsprites tetrominoes; do
  case $D in
    dsprites)    LINK=data/multi_dsprites/colored_on_grayscale_test.npz ;;
    tetrominoes) LINK=data/tetrominoes/tetrominoes_test.npz ;;
  esac
  for S in val128 eval320_128; do
    ln -sf $PWD/data/ours/${D}_${S}.npz $LINK
    DUMP=$B/akornC_${D}_${S}.npz CUDA_VISIBLE_DEVICES=$G $P -u eval_dump.py \
      --data=$D --data_imsize=128 $M
  done
done

# the dSprites-trained row runs at its own published resolution (64)
for S in val eval320; do
  ln -sf $PWD/data/ours/dsprites_$S.npz data/multi_dsprites/colored_on_grayscale_test.npz
  DUMP=$B/akorn_dsprites_$S.npz CUDA_VISIBLE_DEVICES=$G $P -u eval_dump.py \
    --model=akorn --data=dsprites --model_path=runs/ours_dsprites_akorn_attn/ema_wrapped.pth \
    --model_imsize=64 --J=attn --L=1 --T=8 --ch=128 --psize=4 --c_norm=none
done

# ---------------------------------------------------------------- 5. reduce to the grid
for F in akornC_clevr_val akornC_clevr_eval320 \
         akornC_dsprites_val128 akornC_dsprites_eval320_128 \
         akornC_tetrominoes_val128 akornC_tetrominoes_eval320_128; do
  $P -u $C/to_grid64.py --inp $B/$F.npz --out $B/${F}_g64.npz
done

# ---------------------------------------------------------------- 6. our evaluation
O=$A/data/ours; CL=$A/data/clevr_with_masks
score () {  # name val eval val-ids eval-ids dataset
  $P -u $C/bg_any.py --val "$2" --eval "$3" --val-ids "$4" --eval-ids "$5" \
     --dataset "$6" --code $C --config $CFG --run $RUN --out $B/akornC_$1_bg.json
}
score clevr          $B/akornC_clevr_val_g64.npz              $B/akornC_clevr_eval320_g64.npz \
                     $CL/clevr_val.npz                        $CL/clevr_eval320.npz            clevr
score tetrominoes    $B/akornC_tetrominoes_val128_g64.npz     $B/akornC_tetrominoes_eval320_128_g64.npz \
                     $O/tetrominoes_val128.npz                $O/tetrominoes_eval320_128.npz   tetrominoes
score multi_dsprites $B/akornC_dsprites_val128_g64.npz        $B/akornC_dsprites_eval320_128_g64.npz \
                     $O/dsprites_val128.npz                   $O/dsprites_eval320_128.npz      multi_dsprites

$P -u $C/bg_any.py --val $B/akorn_dsprites_val.npz --eval $B/akorn_dsprites_eval320.npz \
   --val-ids $O/dsprites_val.npz --eval-ids $O/dsprites_eval320.npz \
   --dataset multi_dsprites --code $C --config $CFG --run $RUN --out $B/akorn_dsprites_bg.json

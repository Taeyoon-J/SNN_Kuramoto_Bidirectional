"""Frozen user-requested four-model final comparison; no model search."""
import hashlib
from functools import lru_cache
import numpy as np

MODELS=('240','237','244','245');SEEDS=(0,1,2)
COUNT=35000;EPOCHS=10;BATCH=16;BATCHES=(COUNT+BATCH-1)//BATCH;UPDATES=BATCHES*EPOCHS
GRID=64;EVAL_IDS=tuple(range(1320,1640))

def require(model,seed):
    if model not in MODELS or type(seed) is not int or seed not in SEEDS:raise ValueError('registered model and seed required')
    return model,seed

@lru_cache(maxsize=1)
def train_ids():
    pool=np.r_[np.arange(1000),np.arange(1640,70640)].astype(np.int64)
    return np.sort(np.random.RandomState(24935000).choice(pool,COUNT,replace=False))

@lru_cache(maxsize=30)
def epoch_ids(seed,epoch):
    if seed not in SEEDS or not 0<=epoch<EPOCHS:raise ValueError('invalid seed/epoch')
    return train_ids()[np.random.RandomState(249000+seed+10000*epoch).permutation(COUNT)]

def batch_ids(seed,step):
    if type(step) is not int or not 0<=step<UPDATES:raise ValueError('invalid update')
    epoch,index=divmod(step,BATCHES)
    return epoch_ids(seed,epoch)[index*BATCH:(index+1)*BATCH]

def specification():
    return dict(models=list(MODELS),seeds=list(SEEDS),TRAIN_count=COUNT,TRAIN_ids_sha256=hashlib.sha256(train_ids().tobytes()).hexdigest(),
        epochs=EPOCHS,batch=BATCH,last_batch=COUNT%BATCH,updates=UPDATES,exposures=COUNT*EPOCHS,grid=[GRID,GRID],
        evaluation_ids=list(EVAL_IDS),learning_rates=[3e-5,3e-4,3e-4],optimizer='fresh Adam',gradient_clip=1.,
        shared_trainable_components='whole encoder and actual sensory projection',
        loss_coefficients={'240':1.,'237':1.,'244':16.,'245':16.},
        architecture_and_readout_unchanged=True,source_model_histories_differ=True,
        reproduce_missing_244_seeds=[0,2],intermediate_checkpoints_retained=True,
        final_checkpoint_selection='fixed terminal10-pass endpoint; no validation-based early selection')

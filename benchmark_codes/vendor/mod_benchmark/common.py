import hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
import torch

CODE=Path(__file__).resolve().parents[1]
DATA=Path('/work/USERS/tkim1/mod')
EXPORTS=Path('/Data0/kevinswk/patch_v2_sw/collaborative_testing/generated/patch_final_exports')
DATASETS={'multi_dsprites':64,'tetrominoes':32}
EPOCHS=10;TRAIN=6000;BATCH=16;UPDATES=3750;TEST_IDS=list(range(7000,8000))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)

def immutable():
    expected=json.loads((CODE/'mod_benchmark/BENCHMARK_LOCK.json').read_bytes())
    actual={p.relative_to(CODE).as_posix():sha(p) for p in [CODE/'benchmark.md',*sorted((CODE/'benchmark').rglob('*'))]
            if p.is_file() and '__pycache__' not in p.parts}
    if actual!=expected:raise RuntimeError('protected benchmark files differ from frozen f9301ed snapshot')
    return expected

def benchmark_module(name):
    immutable();old=sys.path.copy()
    try:
        spec=importlib.util.spec_from_file_location('_sw0250_benchmark_'+name,CODE/'benchmark'/f'{name}.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    finally:sys.path[:]=old

def source_record(seed):
    manifest=json.loads((CODE/'checkpoints/manifest.json').read_bytes())
    return next(v for v in manifest['checkpoints'] if v['seed']==seed)

def dataset_record(archive,ds):
    return json.loads((Path(archive)/'inputs'/ds/'binding.json').read_bytes())

def rgb(archive,ds,split):
    return np.load(Path(archive)/'inputs'/ds/f'{split}128.npy',mmap_mode='r',allow_pickle=False)

def training_labels(ds,indices):
    grid=DATASETS[ds]
    blob=torch.load(DATA/f'targets_{ds}_grid{grid}.pt',map_location='cpu',weights_only=True,mmap=True)
    y=blob['patch_labels'][torch.as_tensor(indices,dtype=torch.long)].numpy().copy()
    if np.any(np.asarray(indices)<0) or np.any(np.asarray(indices)>=TRAIN):raise ValueError('TRAIN rows only')
    # The model remains native64; Tetrominoes supervision is the declared grid32
    # partition lifted by exact 2x2 repetition, not a fabricated native35 grid64 target.
    return np.repeat(np.repeat(y,2,axis=1),2,axis=2) if grid==32 else y

def prediction_grid(labels,ds):
    labels=torch.as_tensor(labels,dtype=torch.int64)
    if labels.shape!=(64,64):raise ValueError('actual SW0240 native64 output required')
    if DATASETS[ds]==64:return labels.numpy().copy()
    # Use the protected benchmark's own modal rule, including smallest-ID ties.
    cells=benchmark_module('make_targets_grid')
    return cells.mask_cells(labels[None],32)[0][0].numpy()

def epoch_order(seed,epoch):
    return np.random.RandomState(249000+seed+10000*epoch).permutation(TRAIN)

def frozen_sha(model):
    h=hashlib.sha256()
    for key,value in sorted(model.wrapped.state_dict().items()):
        if key=='core.gamma_channel_proj.weight':continue
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def job_binding(archive,ds,seed,mode):
    from snn_kuramoto_bidirectional.sw240.model import model_sha256
    return dict(dataset=ds,seed=seed,mode=mode,source=source_record(seed),
        inputs=dataset_record(archive,ds),benchmark_sha256=immutable(),epochs=EPOCHS,
        TRAIN_count=TRAIN,batch=BATCH,optimizer_updates=UPDATES if mode=='finetune' else 0,
        native_model_grid=64,evaluation_grid=DATASETS[ds],test_ids=TEST_IDS,
        supervision='mask supervised' if mode=='finetune' else 'no new training; source was mask supervised',
        readout_adapter='identity' if DATASETS[ds]==64 else 'native64 to grid32 with protected modal rule',
        gamma_files_used_by_model=False,pretrained_encoder_replaced=False,
        scientific_code={p.relative_to(CODE).as_posix():sha(p) for parent in ('mod_benchmark','snn_kuramoto_bidirectional')
                         for p in sorted((CODE/parent).rglob('*.py'))})

def check_inputs(binding):
    immutable()
    for name,record in binding['inputs']['source_files'].items():
        if sha(DATA/name)!=record['sha256']:raise RuntimeError('benchmark input changed: '+name)

def preflight(ds,folder):
    import subprocess,time
    immutable();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    stamp=str(time.time_ns())
    command=[sys.executable,'-B',str(CODE/'benchmark/preflight.py'),'--dataset',ds,'--grid',str(DATASETS[ds]),'--src',str(DATA)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=180,
                          env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    (folder/f'preflight_{stamp}.stdout').write_text(result.stdout,encoding='utf-8')
    (folder/f'preflight_{stamp}.stderr').write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('protected preflight failed: '+result.stdout+'\n'+result.stderr)
    immutable()
    return dict(command=command,exit_code=0,stdout_sha256=sha(folder/f'preflight_{stamp}.stdout'))

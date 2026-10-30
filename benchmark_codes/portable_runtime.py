"""Path-only adapters around the unmodified actual server research source."""
from __future__ import annotations
import copy,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))

def configure(config_file):
    if os.name!='posix':raise RuntimeError('Use Linux/WSL2: the original runner uses POSIX fcntl locks.')
    config=json.loads(Path(config_file).read_text())
    from mod_benchmark import common as c
    from readout_benchmark import runtime as r
    c.DATA=Path(config['peer_mod_root']);r.CANONICAL=Path(config['clevr_cache'])
    original_record=r.canonical_record
    def canonical_record():
        manifest=json.loads((r.CANONICAL/'manifest.json').read_bytes())
        assert manifest['status']=='complete_exact_registered_TRAIN70000_input_cache'
        source=dict(manifest['binding']['dataset']);path=Path(config['clevr_hdf5'])
        assert path.stat().st_size==source['size'],'wrong CLEVR source; validate asset SHA before training'
        source.update(path=str(path),mtime_ns=path.stat().st_mtime_ns)
        return source,c.sha(r.CANONICAL/'manifest.json')
    r.canonical_record=canonical_record
    def protected_preflight(dataset,folder):
        c.immutable();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);stamp=str(time.time_ns())
        command=[sys.executable,'-B',str(c.CODE/'benchmark/preflight.py'),'--dataset',dataset,
                 '--grid',str(c.DATASETS[dataset]),'--src',str(c.DATA),'--reference',config['preflight_reference']]
        proc=subprocess.run(command,capture_output=True,text=True,timeout=180,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        (folder/f'preflight_{stamp}.stdout').write_text(proc.stdout)
        (folder/f'preflight_{stamp}.stderr').write_text(proc.stderr)
        if proc.returncode:raise RuntimeError('protected preflight failed: '+proc.stdout+'\n'+proc.stderr)
        c.immutable();return dict(command=command,exit_code=0)
    c.preflight=protected_preflight
    # The parent cache has35000 TRAIN, while the authorized subset has3500.
    # Run the original exact parent-cache checks, then validate the actual subset.
    # This changes no data, model, objective, predictions or metric rules.
    from full_benchmark import extra_preflight
    original_extra=extra_preflight.preflight
    def selected_preflight(dataset,out):
        import numpy as np
        with np.load(dataset.folder/'ids.npz',allow_pickle=False) as saved:parent=saved['train'].copy()
        assert len(dataset.train)==3500 and np.isin(dataset.train,parent).all()
        assert len(np.unique(np.r_[dataset.train,dataset.validation,dataset.test]))==5500
        proxy=copy.copy(dataset);proxy.train=parent
        result=original_extra(proxy,out)
        result.update(actual_selected_TRAIN_count=3500,actual_selected_TRAIN_subset_of_verified_parent=True)
        c.write(Path(out)/'preflight.json',result);return result
    extra_preflight.preflight=selected_preflight
    c.immutable()
    return config

def connect_inputs(config,output):
    output=Path(output).resolve();source=Path(config['prepared_inputs']).resolve()
    forbidden=Path('/work/USERS/kevinswk/experiments/SW0255_full3500')
    if output==forbidden:raise ValueError('Use a NEW output directory; do not write into the live server experiment.')
    output.mkdir(parents=True,exist_ok=True)
    inputs=output/'inputs'
    if inputs.exists():assert inputs.resolve()==source,'output already bound to other input data'
    else:inputs.symlink_to(source,target_is_directory=True)
    return output

def check_splits(config,output,datasets):
    from full_benchmark.data import Dataset
    from mod_benchmark import common as c
    result={}
    for name in datasets:
        dataset=Dataset(name,output)
        expected=json.loads((HERE/'contracts'/f'{name}_split.json').read_text())
        for key in ('train_ids','validation_ids','evaluation_ids'):
            assert dataset.binding[key]==expected[key],(name,key)
        assert dataset.rgb(dataset.train[:1]).shape==(1,128,128,3)
        assert dataset.labels(dataset.train[:1]).shape==(1,64,64)
        if name in ('multi_dsprites','tetrominoes'):c.preflight(name,output/'preflight'/name)
        elif name in ('objects_room','clevrtex_original'):
            from full_benchmark.extra_preflight import preflight
            preflight(dataset,output/'preflight'/name)
        result[name]=dict(unique_TRAIN=3500,validation=len(dataset.validation),evaluation=len(dataset.test),
                          splits_equal_actual_server=True,RGB_shape=[128,128,3])
    c.write(output/'preflight.json',result);return result

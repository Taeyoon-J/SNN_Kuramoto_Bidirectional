"""TRAIN16 oracle-count admission of a whole-graph spike-affinity readout."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from collaborative_test.SW_0186_full_encoder_qce import train, predict_evaluate as pe
from collaborative_test.SW_0180_qce_train_validation_triage import run as helpers
from . import spectral

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXPERIMENT='SW0190_spectral_count_admission'
SEEDS=(0,1,2)
POSITIONS=tuple(range(0,512,32))
BATCH=4
VARIANTS=('baseline','spectral_existing_support','spectral_all_patch')
write_report=helpers.write_report


def implementation_fingerprint():
    files={str((HERE/name).relative_to(ROOT)).replace('\\','/'):hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in ('run.py','spectral.py','test_run.py','DESIGN_KO.md')}
    for module in (helpers,helpers.native,helpers.sw136,helpers.solver):
        path=Path(module.__file__).resolve()
        files[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(experiment=EXPERIMENT,files=files,parent_prediction=pe.implementation_fingerprint(),
        positions=list(POSITIONS),seeds=list(SEEDS),variants=list(VARIANTS),batch=BATCH,
        steps=1024,settle=512,optimizer_updates=0,validation_data_read=False,
        oracle_count_scope='TRAIN modal GT visible object count only; no learned count predictor yet')


def prepare_inputs_cpu(seed_subset=SEEDS):
    if torch.cuda.is_initialized():raise RuntimeError('CPU admission must precede CUDA')
    seeds=tuple(seed_subset)
    if not seeds or len(set(seeds))!=len(seeds) or not set(seeds).issubset(SEEDS):raise ValueError('invalid seed subset')
    inputs=train.prepare_training_inputs_cpu();ids=np.asarray(inputs['ordered_train_ids'])
    if ids.shape!=(512,) or not np.all(((ids>=0)&(ids<=999))|((ids>=1640)&(ids<=70639))):
        raise ValueError('canonical TRAIN512 required')
    arrays=inputs['arrays']
    if arrays['rgb'].shape!=(512,128,128,3) or arrays['modal_gt'].shape!=(512,32,32):
        raise ValueError('canonical native32 TRAIN cache required')
    records={}
    for seed in seeds:
        artifact=train.OUTPUT_ROOT/f'product_q_ce_seed{seed}'
        manifest=train.validate_training_artifact('product_q_ce',artifact,seed=seed)
        prepared=dict(training_inputs=inputs,artifact=artifact,manifest=manifest)
        found,digest=pe._load_model(seed,prepared,'cpu')
        if digest!=manifest['final_model_state_sha256']:raise ValueError('frozen incumbent changed')
        del found
        records[seed]=prepared
    if torch.cuda.is_initialized():raise RuntimeError('CPU admission initialized CUDA')
    return dict(inputs=inputs,records=records,selected_seeds=list(seeds),cuda_initialized=False)


@torch.no_grad()
def run_diagnostic(prepared,*,seed_subset=SEEDS,device='cuda:0'):
    seeds=tuple(seed_subset)
    if prepared.get('cuda_initialized') is not False or not set(seeds).issubset(prepared['selected_seeds']):
        raise ValueError('unadmitted inputs')
    dev=torch.device(device)
    if dev.type!='cuda' or not torch.cuda.is_initialized():raise RuntimeError('exclusive GPU admission required')
    inputs=prepared['inputs'];ids=np.asarray(inputs['ordered_train_ids']);arrays=inputs['arrays']
    report=dict(experiment=EXPERIMENT,status='complete',implementation_fingerprint=implementation_fingerprint(),
        optimizer_updates=0,validation_data_read=False,oracle_object_count=True,
        ground_truth_scope='TRAIN modal object counts for oracle readout; TRAIN masks for scoring only',
        cohort_ids=ids[list(POSITIONS)].tolist(),positions=list(POSITIONS),seeds={})
    for seed in seeds:
        found,before=pe._load_model(seed,prepared['records'][seed],dev)
        outputs={key:[] for key in VARIANTS};counts=[];digests=[]
        for batch_index,start in enumerate(range(0,len(POSITIONS),BATCH)):
            indices=list(POSITIONS[start:start+BATCH])
            rgb=torch.from_numpy(np.ascontiguousarray(arrays['rgb'][indices])).to(dev)
            helpers._paired_seed_rng(seed,batch_index)
            _,trace,adj,z,q,_=helpers.native._native_forward(found,rgb,live_tail_steps=0)
            if q.shape!=(BATCH,1024,1024):raise ValueError('native affinity geometry changed')
            qcc=np.asarray(helpers.sw136._qcc_labels(trace),dtype=np.int32)
            gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().cpu())
            digests.append(dict(q=helpers._sha_array(q.cpu().numpy()),z=helpers._sha_array(z.cpu().numpy())))
            for i,index in enumerate(indices):
                target=arrays['modal_gt'][index]
                # Only the scalar TRAIN object count enters the alternative readout.
                count=int(np.unique(target[target>0]).size)
                if count<1:raise ValueError('oracle cohort has no visible object')
                qn=q[i].cpu().numpy().astype(np.float64)
                base=np.asarray(helpers.solver.foreground_induced_merge(qn,adj[i].cpu().numpy().astype(np.float64),
                    gain,qcc[i],grid_shape=(32,32))['labels'],dtype=np.int32)
                alternatives=spectral.transform(qn,base,count)
                if not np.array_equal(alternatives['spectral_existing_support']>0,base>0):
                    raise ValueError('fixed-support arm changed foreground support')
                outputs['baseline'].append(base)
                for key,value in alternatives.items():outputs[key].append(value)
                counts.append(count)
            del rgb,trace,adj,z,q
        after=train.base._state_sha(found.wrapped,found.encoder)
        if before!=after:raise ValueError('no-update admission mutated model tensors')
        target=arrays['modal_gt'][list(POSITIONS)]
        report['seeds'][str(seed)]=dict(before_sha256=before,after_sha256=after,state_unchanged=True,
            batch_digests=digests,oracle_visible_object_counts=counts,
            variants={key:helpers._metric_arrays(np.stack(value),target) for key,value in outputs.items()})
        del found
    return report

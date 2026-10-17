"""Explicit five-family SourceT factory; original tensor initialization preserved."""
from __future__ import annotations
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0179_qce_seed_replication import train as parent
from collaborative_test.SW_0175_critical_component_binding import train_v2 as native

SOURCE_UTIL_MODULE=__name__
SOURCE_BASE_LR=3e-5
FAMILIES={'context_encoder','graph','native_core','integration','source_base'}

def __getattr__(name):
    if hasattr(parent,name):return getattr(parent,name)
    return getattr(native,name)

def implementation_fingerprint():
    here=Path(__file__).resolve().parent;root=here.parents[1]
    names=('DESIGN_KO.md','source.py','test_source.py')
    return dict(experiment='SW0191_full_encoder_qce_factory',parent=parent.implementation_fingerprint(),
        files={str((here/name).relative_to(root)).replace('\\','/'):hashlib.sha256((here/name).read_bytes()).hexdigest() for name in names},
        source_base_lr=SOURCE_BASE_LR,source_base_trainable=True,loss='product_q_ce_plus_partition_error_feedback',
        initialization='exact completed SW0186 weights, reset empty Adam',
        incumbent_factory_sha256=hashlib.sha256((root/'collaborative_test/SW_0186_full_encoder_qce/source.py').read_bytes()).hexdigest())

def fingerprint_sha256():
    return hashlib.sha256(json.dumps(implementation_fingerprint(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def enable_source_encoder(found,optimizer,families):
    if set(families)!={'context_encoder','graph','native_core','integration'} or optimizer.state:
        raise ValueError('joint encoder requires the fresh four-family empty Adam factory')
    existing=[p for group in optimizer.param_groups for p in group['params']]
    entries=[('base_encoder.'+name,p) for name,p in found.encoder.base_encoder.named_parameters()]
    if not entries or any(id(p) in {id(q) for q in existing} for _,p in entries):
        raise ValueError('source CNN parameters are absent or already optimized')
    found.encoder.base_encoder.requires_grad_(True)
    optimizer.add_param_group(dict(name='source_base',params=[p for _,p in entries],
        parameter_names=[name for name,_ in entries],lr=SOURCE_BASE_LR))
    result={**families,'source_base':entries}
    parameters=[p for rows in result.values() for _,p in rows]
    if len(parameters)!=len({id(p) for p in parameters}) or any(not p.requires_grad for p in parameters):
        raise ValueError('five optimizer families overlap or contain frozen parameters')
    return result

def _frozen_source_components(found,base_encoder):
    values=native._frozen_source_components(found,base_encoder)
    if 'base_encoder' not in values:raise ValueError('missing original frozen source schema')
    return {key:value for key,value in values.items() if key!='base_encoder'}

def _base_encoder_sha(found):
    return native.preflight._module_sha(found.encoder.base_encoder)

def _base_encoder_sha_from_state(encoder_state):
    entries={name[len('base_encoder.'):]:value for name,value in encoder_state.items() if name.startswith('base_encoder.')}
    if not entries:raise ValueError('checkpoint has no original RGB encoder tensors')
    digest=hashlib.sha256()
    for name,value in sorted(entries.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def _read_incumbent(seed):
    from collaborative_test.SW_0186_full_encoder_qce import train as incumbent
    artifact=incumbent.OUTPUT_ROOT/f'product_q_ce_seed{int(seed)}'
    manifest=incumbent.validate_training_artifact('product_q_ce',artifact,seed=int(seed))
    path=artifact/'checkpoint.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise ValueError('incumbent checkpoint SHA mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (payload.get('experiment')!=incumbent.EXPERIMENT or payload.get('arm')!='product_q_ce'
            or payload.get('seed')!=int(seed) or payload.get('bindings')!=manifest['bindings']
            or parent._state_sha_from_payload(payload)!=manifest['final_model_state_sha256']):
        raise ValueError('incumbent checkpoint producer/state mismatch')
    return manifest,payload,dict(experiment=incumbent.EXPERIMENT,seed=int(seed),
        manifest_sha256=hashlib.sha256((artifact/'manifest.json').read_bytes()).hexdigest(),
        checkpoint_sha256=manifest['checkpoint_sha256'],final_model_state_sha256=manifest['final_model_state_sha256'],
        artifact=str(artifact),optimizer_policy='reset_empty_Adam_all_five_groups')


def _make_fresh(arm,seed,inputs,device):
    if arm!='product_q_ce':raise ValueError('SW0191 only registers unchanged QCE')
    found,optimizer,families,binding,setup=parent._make_fresh(arm,seed,inputs,device)
    before=parent._state_sha(found.wrapped,found.encoder)
    families=enable_source_encoder(found,optimizer,families)
    if parent._state_sha(found.wrapped,found.encoder)!=before:raise ValueError('unfreezing mutated source initialization')
    manifest,payload,lineage=_read_incumbent(seed)
    found.wrapped.load_state_dict(payload['wrapped'],strict=True)
    found.encoder.load_state_dict(payload['encoder'],strict=True)
    warm_sha=parent._state_sha(found.wrapped,found.encoder)
    if warm_sha!=manifest['final_model_state_sha256'] or optimizer.state:
        raise ValueError('warm model must equal incumbent exactly with empty Adam')
    if _base_encoder_sha(found)!=manifest['source_base_encoder_final_sha256']:
        raise ValueError('warm source CNN differs from trained incumbent')
    binding={**binding,'initial_model_state_sha256':warm_sha,
        'base_encoder_sha256':_base_encoder_sha(found),'incumbent_lineage':lineage,
        'factory_variant':'SW0191_full_encoder_qce',
        'source_base_trainable':True,'source_base_lr':SOURCE_BASE_LR,
        'source_factory_fingerprint_sha256':fingerprint_sha256(),
        'optimizer_groups':parent._optimizer_contract(optimizer)}
    setup=(setup[0],sys.modules[__name__],setup[2],_frozen_source_components(found,setup[2]),setup[4])
    return found,optimizer,families,binding,setup

def prepare_fresh_training_state_cpu(seed,inputs=None):
    if seed not in (0,1,2) or torch.cuda.is_initialized():raise RuntimeError('fresh registered CPU source required')
    inputs=parent.prepare_training_inputs_cpu() if inputs is None else inputs
    found,optimizer,families,binding,_setup=_make_fresh('product_q_ce',seed,inputs,'cpu')
    if set(families)!=FAMILIES or optimizer.state or torch.cuda.is_initialized():raise ValueError('invalid five-family CPU factory')
    proof=dict(experiment='SW0191_full_encoder_qce_factory',seed=seed,
        context_seed=parent.CONTEXT_SEED_BASE+seed,source_binding=binding['source_binding'],
        source_state_sha256=binding['source_state_sha256'],base_encoder_sha256=binding['base_encoder_sha256'],
        initial_model_state_sha256=binding['initial_model_state_sha256'],optimizer_groups=binding['optimizer_groups'],
        family_names=list(families),parameter_count=sum(len(rows) for rows in families.values()),adam_state_entries=0,
        dataset_manifest_sha256=inputs['dataset_manifest_sha256'],dataset_arrays_sha256=inputs['dataset_arrays_sha256'],
        ordered_train_ids_sha256=hashlib.sha256(np.asarray(inputs['ordered_train_ids'],dtype='<i8').tobytes()).hexdigest(),
        source_base_trainable=True,incumbent_lineage=binding['incumbent_lineage'],validation_data_read=False,cuda_initialized=False,
        training_fingerprint=implementation_fingerprint())
    return dict(proof=proof,found=found,optimizer=optimizer,families=families,bindings=binding,inputs=inputs)

def _credit_failures(row):
    groups=('family_gradient_norms','family_parameter_delta_l2')
    if any(set(row.get(key,{}))!=FAMILIES for key in groups):return ['five-family gradient/delta schema mismatch']
    original={**row,**{key:{name:value for name,value in row[key].items() if name!='source_base'} for key in groups}}
    failures=parent._credit_failures(original)
    for key in groups:
        value=row[key]['source_base']
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            failures.append('source_base has no finite-positive '+key)
    aux=row.get('feedback_loss_to_emitted_S_gradient_l2')
    if not isinstance(aux,(int,float)) or not math.isfinite(aux) or aux<=0:
        failures.append('partition-feedback-only loss has no finite-positive emitted-S gradient')
    parameter_credit=row.get('feedback_trainable_gradient_rss')
    families=row.get('feedback_family_gradient_rss')
    if (not isinstance(parameter_credit,(int,float)) or not math.isfinite(parameter_credit) or parameter_credit<=0
            or not isinstance(families,dict) or set(families)!=FAMILIES
            or any(not math.isfinite(float(v)) or v<0 for v in families.values())):
        failures.append('partition-feedback-only loss has no finite-positive trainable-parameter gradient')
    return failures

"""Train actual64 core+encoder and detached RGB readout; no TRAIN masks opened."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from portable import load,module,HERE
from training import configure,make_optimizers,step


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):digest.update(b)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)


def cpu(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def save(path,found,head,optimizers,record,epoch,cursor,updates,best,binding):
    path=Path(path);tmp=path.with_suffix('.tmp')
    device=next(found.encoder.parameters()).device
    value=dict(format='SW0261_two_path_v1',base_checkpoint=record['base_checkpoint'],base_sha256=record['base_sha256'],seed=record['seed'],
        encoder=cpu(found.encoder),wrapped=cpu(found.wrapped),head=cpu(head),feature_mean=found.feature_mean.detach().cpu(),
        feature_std=found.feature_std.detach().cpu(),feature_clip=found.feature_clip,
        body_optimizer=optimizers[0].state_dict() if optimizers[0] else None,head_optimizer=optimizers[1].state_dict(),
        epoch=epoch,cursor=cursor,updates=updates,best=best,binding=binding,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
    torch.save(value,tmp);os.replace(tmp,path)


def initialize_random(found,head,seed):
    from SW_0094_aligned_joint_pilot.run import hparams
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
    from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder
    from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
    from collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator import install_operator
    device=next(found.encoder.parameters()).device;torch.manual_seed(261000+seed)
    hp=hparams('raw');hp.num_regions=4096;hp.k=4096.;hp.graph_top_k=512
    hp.graph_spatial_grid_size=64;hp.spike_spatial_grid_size=64;hp.gamma_patch_grid_size=64
    core=S2NetCore(hp.validate(),device=device).to(device)
    torch.nn.init.zeros_(core.dendric_layer.oscillator_dense.bias)
    install_operator(core.graph_generator,'T');found.wrapped=PhaseStateIntegration(core,'phase')
    found.encoder=CNNFeatureEncoder(8,3,3,bias=True).to(device)
    original=module('train261_fresh_original',HERE.parent/'unsupervised_readout_benchmark/model.py').ReconstructionReadout(seed)
    original._seed_hint=seed;head=type(head)(original,'context_rgb').to(device)
    import types
    implementation=module('train261_fresh_native',HERE/'model.py')
    head.routing=types.MethodType(implementation.native_routing,head)
    return found,head


def fit_statistics(found,data,ids):
    total=0.;squared=0.;count=0
    with torch.no_grad():
        for i in ids:
            image=torch.from_numpy(np.asarray(data[i]).copy())[None].to(next(found.encoder.parameters()).device)
            x=found.encoder(image.permute(0,3,1,2).float()/255.).double()
            total+=float(x.sum());squared+=float(x.square().sum());count+=x.numel()
    mean=total/count;std=max(squared/count-mean*mean,1e-12)**.5
    device=next(found.encoder.parameters()).device
    found.feature_mean=torch.tensor(mean,device=device);found.feature_std=torch.tensor(std,device=device);found.feature_clip=3.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,help='HDF5, uint8 image[N,128,128,3]; mask never read')
    p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--seed',type=int,default=0,choices=[0,1,2])
    p.add_argument('--init',choices=['pretrained','random'],default='pretrained')
    p.add_argument('--train-count',type=int,default=128);p.add_argument('--train-ids',help='JSON list of TRAIN image IDs')
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--steps',type=int,default=1024);p.add_argument('--settle',type=int,default=512);p.add_argument('--tail',type=int,default=64)
    p.add_argument('--geodesic-chunk',type=int,default=8,help='exact geodesic row chunk; controls memory, not model resolution')
    p.add_argument('--decoder-chunk',type=int,default=16,help='checkpoint decoder groups for memory-bounded RGB loss')
    p.add_argument('--encoder-lr',type=float,default=3e-6);p.add_argument('--core-lr',type=float,default=3e-5);p.add_argument('--head-lr',type=float,default=.001)
    p.add_argument('--freeze-core',action='store_true');p.add_argument('--freeze-encoder',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--save-every',type=int,default=8);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.epochs<1 or args.train_count<1 or args.save_every<1 or min(args.geodesic_chunk,args.decoder_chunk)<1 or min(args.encoder_lr,args.core_lr,args.head_lr)<=0:p.error('positive budget/lr required')
    if not (0<=args.settle<args.steps and 0<args.tail<=args.steps and (args.steps-args.settle)%32==0):p.error('valid tail and settled history divisible by32 required')
    torch.set_num_threads(1);torch.manual_seed(261000+args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    import h5py
    path=Path(args.data).resolve();dataset=h5py.File(path,'r');images=dataset['image']
    if images.dtype!=np.uint8 or images.shape[1:]!=(128,128,3):raise ValueError('uint8 image[N,128,128,3] required; no implicit preprocessing')
    if args.train_ids:ids=list(map(int,json.loads(Path(args.train_ids).read_text())))
    else:
        pool=np.r_[np.arange(min(1000,len(images))),np.arange(1640,min(70640,len(images)))]
        if len(pool)<args.train_count:raise ValueError('not enough registered TRAIN images')
        ids=np.random.RandomState(2613500).permutation(pool)[:args.train_count].tolist()
    if not ids or len(set(ids))!=len(ids) or min(ids)<0 or max(ids)>=len(images) or any(1000<=i<1640 for i in ids):raise ValueError('unique TRAIN IDs, excluding1000–1639, required')
    checkpoint=Path(args.checkpoint) if args.checkpoint else HERE.parent/'readout64_optimization/checkpoints'/f'sw0260_context_seed{args.seed}.pt'
    found,head,record=load(checkpoint,args.device)
    from collaborative_test.SW_0135_native32_spike_binding import resolution
    resolution.ROW_CHUNK=args.geodesic_chunk
    if record['seed']!=args.seed:raise ValueError('checkpoint seed differs')
    record=dict(record,base_checkpoint=record.get('base_checkpoint',checkpoint.name),base_sha256=record.get('base_sha256',sha(checkpoint)))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    binding=dict(seed=args.seed,initialization=args.init,checkpoint_sha256=sha(checkpoint),train_ids=ids,
        data=dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns),
        source={p.name:sha(p) for p in [HERE/'train.py',HERE/'training.py',HERE/'portable.py',HERE/'model.py',HERE/'partition.py']},
        steps=args.steps,settle=args.settle,tail=args.tail,geodesic_chunk=args.geodesic_chunk,decoder_chunk=args.decoder_chunk,lrs=[args.encoder_lr,args.core_lr,args.head_lr],
        freeze_core=args.freeze_core,freeze_encoder=args.freeze_encoder,TRAIN_masks_used=False)
    if (out/'binding.json').exists():
        if not args.resume:raise ValueError('existing output; use --resume or a new output directory')
        if json.loads((out/'binding.json').read_bytes())!=binding:raise ValueError('resume contract differs')
    else:write(out/'binding.json',binding)
    if args.init=='random' and not args.resume:
        found,head=initialize_random(found,head,args.seed);fit_statistics(found,images,ids)
    configure(found,head,freeze_core=args.freeze_core,freeze_encoder=args.freeze_encoder)
    optimizers=make_optimizers(found,head,args.encoder_lr,args.core_lr,args.head_lr)
    epoch=cursor=updates=0;best=float('inf')
    if args.resume:
        saved=torch.load(out/'resume.pt',map_location=args.device,weights_only=True)
        if saved['binding']!=binding:raise ValueError('saved contract differs')
        found.encoder.load_state_dict(saved['encoder']);found.wrapped.load_state_dict(saved['wrapped']);head.load_state_dict(saved['head'])
        found.feature_mean=saved['feature_mean'].to(args.device);found.feature_std=saved['feature_std'].to(args.device);found.feature_clip=saved['feature_clip']
        if optimizers[0]:optimizers[0].load_state_dict(saved['body_optimizer'])
        optimizers[1].load_state_dict(saved['head_optimizer'])
        epoch,cursor,updates,best=saved['epoch'],saved['cursor'],saved['updates'],saved['best']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None and str(args.device).startswith('cuda'):torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),args.device)
        write(out/'resume_verified.json',dict(restored_updates=updates,body_Adam_restored=optimizers[0] is not None,head_Adam_restored=True,epoch=epoch,cursor=cursor))
    elif not (out/'initial.pt').exists():save(out/'initial.pt',found,head,optimizers,record,epoch,cursor,updates,best,binding)
    for e in range(epoch,args.epochs):
        order=np.random.RandomState(249000+args.seed+10000*e).permutation(len(ids))
        begin=cursor if e==epoch else 0
        for position in range(begin,len(order)):
            image_id=ids[int(order[position])];torch.manual_seed(255000+image_id+100000*e)
            rgb=torch.from_numpy(np.asarray(images[image_id]).copy())[None].to(args.device)
            started=time.monotonic()
            report=step(found,head,rgb,*optimizers,steps=args.steps,settle=args.settle,tail=args.tail,package=HERE,group_chunk=args.decoder_chunk)
            updates+=1;next_epoch,next_cursor=(e+1,0) if position+1==len(order) else (e,position+1)
            # Monitor training objective only; best.pt is NOT a validation-selected claim.
            combined=report['body_loss']+report['head_loss'];improved=combined<best
            if improved:best=combined
            report.update(epoch=e+1,position=position,image_id=image_id,updates=updates,seconds=time.monotonic()-started,updated=time.time())
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
            if updates==1 or updates%args.save_every==0 or next_cursor==0:
                save(out/'resume.pt',found,head,optimizers,record,next_epoch,next_cursor,updates,best,binding)
            write(out/'progress.json',report)
    write(out/'complete.json',dict(status='training_finished',epochs=args.epochs,updates=updates,unique_TRAIN=len(ids),core_grid=64,
        TRAIN_masks_used=False,core_trained=not args.freeze_core,encoder_trained=not args.freeze_encoder,
        head_loss_reaches_body=False,body_loss_reaches_encoder=not args.freeze_encoder,selection='training_objective_only; no validation/performance claim'))

if __name__=='__main__':main()

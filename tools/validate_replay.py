"""Replay frozen evaluation images without importing the research repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import torch
import h5py
from snn_kuramoto_bidirectional import load_model
from snn_kuramoto_bidirectional.sw240.model import model_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference-archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--count',type=int,default=3)
    args=parser.parse_args()
    if args.output.exists() or not 1<=args.count<=320:
        raise ValueError('new output file and count1..320 required')
    torch.set_num_threads(1)
    report=json.loads((args.reference_archive/'training_report.json').read_bytes())
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.8,device)
    model=load_model(args.checkpoint,device=device)
    assert model_sha256(model)==report['final_model_sha256']
    assert model.config['source_checkpoint_sha256']==report['checkpoint_sha256']
    assert model.config['seed']==report['binding']['seed']
    assert not any(name.startswith('collaborative_test') for name in sys.modules)
    for name,module in tuple(sys.modules.items()):
        if name.startswith('snn_kuramoto_bidirectional') and getattr(module,'__file__',None):
            assert Path(module.__file__).resolve().is_relative_to(ROOT)
    rows=[]
    with h5py.File(report['binding']['dataset']['path'],'r') as stream:
        for i in range(args.count):
            image_id=1320+i
            rgb=np.asarray(stream['image'][image_id:image_id+1])
            if rgb.dtype!=np.uint8:
                rgb=np.rint(np.clip(rgb.astype(np.float32),0,1)*255).astype(np.uint8)
            folder=args.reference_archive/'rows'/f'{i:04d}'
            meta=json.loads((folder/'row.json').read_bytes())
            assert meta['RGB_sha256']==hashlib.sha256(rgb[0].tobytes()).hexdigest()
            raw=(folder/'prediction.npz').read_bytes()
            assert meta['data_sha256']==hashlib.sha256(raw).hexdigest()
            with np.load(folder/'prediction.npz',allow_pickle=False) as archive:
                expected=archive['labels'].copy()
            started=time.perf_counter()
            labels=model.predict(rgb,inference_seeds=[meta['inference_seed']])[0]
            difference=int(np.count_nonzero(labels!=expected))
            row=dict(image_id=image_id,exact_mask_match=difference==0,different_patches=difference,
                     seconds=time.perf_counter()-started)
            rows.append(row)
            print(json.dumps(row),flush=True)
            if difference:
                raise AssertionError('portable model differs from frozen native64 prediction')
    assert model_sha256(model)==report['final_model_sha256']
    result=dict(status='passed_exact_SW0240_native64_replay',seed=model.config['seed'],
        model_sha256=report['final_model_sha256'],source_checkpoint_sha256=report['checkpoint_sha256'],
        count=args.count,rows=rows,torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        original_research_imports=False,training_updates=0,GT_read=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()

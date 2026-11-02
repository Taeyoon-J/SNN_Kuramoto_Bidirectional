"""Load all SW0259 weights offline; use the exact exported legacy32 implementation."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent

def load(checkpoint,device='cpu'):
    snapshot=HERE/'snapshot/python'
    if not snapshot.is_dir():raise FileNotFoundError('SW0259 snapshot/python is required')
    sys.path[:0]=[str(snapshot),str(snapshot/'collaborative_test'),str(snapshot/'snn_kuramoto_bidirectional')]
    from collaborative_test.SW_0135_native32_spike_binding.foundation import wrap_mapped_source,Native32Foundation
    from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator
    from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder
    from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
    blob=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if blob['format']!='SW0259_portable_v1':raise ValueError('not a SW0259 portable checkpoint')
    wrapped=wrap_mapped_source(blob['core16_state'],device=device)
    with torch.no_grad():
        for key in ('a_d','a_m','b'):getattr(wrapped,key).copy_(blob['integration'][key].to(device))
    geodesic_operator.install_operator(wrapped.core.graph_generator,'T')
    encoder=CNNFeatureEncoder(8,3,3,bias=True).to(device);encoder.load_state_dict(blob['encoder_state'],strict=True)
    found=Native32Foundation(blob['seed'],wrapped,encoder,FeaturePatchGammaInitializer(grid_size=32).to(device),
        blob['feature_mean'].to(device),blob['feature_std'].to(device),blob['feature_clip'],np.array([],dtype=np.int64),[],blob['source_provenance'])
    found.wrapped.eval().requires_grad_(False);found.encoder.eval().requires_grad_(False)
    spec=importlib.util.spec_from_file_location('sw0259_portable_architecture',HERE/'model.py')
    architecture=importlib.util.module_from_spec(spec);spec.loader.exec_module(architecture)
    readout=architecture.ReconstructionReadout(blob['seed']).to(device)
    readout.load_state_dict(blob['readout_state'],strict=True);readout.eval().requires_grad_(False)
    return found,readout,blob


@torch.no_grad()
def predict(found,readout,rgb):
    from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
    from collaborative_test.SW_0162_connected_signed_partition import preflight as source
    from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as solver
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.shape!=(128,128,3):raise ValueError('exact uint8 RGB128x128 required; no implicit resize')
    device=next(found.encoder.parameters()).device
    gamma=found.encode(torch.from_numpy(rgb[None].copy()).to(device))
    trace,a=source.capture_source_coupling(found.wrapped.core.kuramoto,
        lambda:late_rollout(found.wrapped,gamma,total_steps=1024,live_tail_steps=0))
    q=source._actual_q(trace)[0].cpu().numpy().astype(np.float64)
    gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
    regions=solver.foreground_induced_merge(q,a[0].cpu().numpy().astype(np.float64),gain,
        np.ones((32,32),dtype=np.int64),grid_shape=(32,32))['labels']
    spec=importlib.util.spec_from_file_location('sw0259_portable_activity',HERE.parent/'vendor/readout_benchmark/head.py')
    activity=importlib.util.module_from_spec(spec);spec.loader.exec_module(activity)
    return readout.predict(activity.activity_features(trace)[0],regions)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    from PIL import Image
    rgb=np.asarray(Image.open(a.image).convert('RGB'))
    found,head,_=load(a.checkpoint,a.device);labels=predict(found,head,rgb)
    np.save(a.output,labels);print(json.dumps(dict(output=a.output,grid=list(labels.shape))))

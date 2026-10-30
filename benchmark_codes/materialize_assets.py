#!/usr/bin/env python3
"""Fetch or verify the exact data assets; no other project checkout is needed."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--ssh',help='e.g. frontier; omit when --verify-only')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--config-out',type=Path,default=HERE/'configs/local.json')
    a=p.parse_args();root=a.data_root.resolve();inventory=json.loads((HERE/'contracts/assets.json').read_text(encoding='utf-8-sig'))
    for asset in inventory['assets']:
        relative=Path(asset['destination']);assert not relative.is_absolute() and '..' not in relative.parts
        destination=root/relative
        valid=destination.is_file() and destination.stat().st_size==asset['size'] and sha(destination)==asset['sha256']
        if not valid:
            if a.verify_only:raise RuntimeError('missing or changed asset: '+str(destination))
            if not a.ssh:raise ValueError('--ssh is required to fetch assets')
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_name(destination.name+'.partial')
            subprocess.run(['scp','-p',a.ssh+':'+shlex.quote(asset['server_path']),str(temporary)],check=True)
            assert temporary.stat().st_size==asset['size'] and sha(temporary)==asset['sha256'],destination
            temporary.replace(destination)
        print('verified',asset['destination'],flush=True)
    config=dict(prepared_inputs=str(root/'inputs'),clevr_cache=str(root/'clevr_cache'),
                clevr_hdf5=str(root/'clevr/clevr_10-full.hdf5'),peer_mod_root=str(root/'peer_mod'),
                preflight_reference=str(root/'preflight_reference/targets_v1.pt'))
    a.config_out.parent.mkdir(parents=True,exist_ok=True);a.config_out.write_text(json.dumps(config,indent=2)+'\n')
    print('config:',a.config_out)
if __name__=='__main__':main()

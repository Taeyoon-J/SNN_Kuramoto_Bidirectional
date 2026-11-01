"""Fetch the15 existing SW0256 initialization assets by SSH, verifying exact SHA."""
import argparse,hashlib,json,re,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent


def valid(path,record):
    if path.stat().st_size!=record['bytes']:return False
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()==record['sha256']


def main():
    p=argparse.ArgumentParser();p.add_argument('--ssh',default='frontier')
    p.add_argument('--destination',required=True);p.add_argument('--sources-out',required=True)
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',a.ssh):raise ValueError('use a host or SSH alias')
    inventory=json.loads((HERE/'frozen_core_benchmark/sources.frontier.json').read_bytes())
    destination=Path(a.destination).expanduser().resolve();destination.mkdir(parents=True,exist_ok=True)
    for seed,roles in inventory['seeds'].items():
        for role,record in roles.items():
            extension='.json' if role=='manifest' else '.pt'
            target=destination/f'seed{seed}_{role}{extension}'
            if target.exists():
                if not valid(target,record):raise RuntimeError('existing destination differs; preserve and inspect: '+str(target))
            else:
                part=target.with_suffix(target.suffix+'.part')
                with part.open('wb') as stream:
                    subprocess.run(['ssh','-o','BatchMode=yes',a.ssh,'cat -- '+shlex.quote(record['path'])],stdout=stream,check=True)
                if not valid(part,record):raise RuntimeError('download SHA mismatch: '+str(part))
                part.replace(target)
            record['frontier_source_path']=record['path'];record['path']=str(target)
    out=Path(a.sources_out).expanduser().resolve();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(status='all15_pretrained_source_assets_SHA_verified',sources=str(out))))


if __name__=='__main__':main()

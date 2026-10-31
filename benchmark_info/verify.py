"""Verify this research snapshot with Python stdlib; never execute archived code."""
from pathlib import Path
import hashlib,json,re,zipfile
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parent

def verify():
    manifest=json.loads((ROOT/'FILES_SHA256.json').read_text(encoding='utf-8'))
    for name,record in manifest.items():
        path=ROOT/name
        assert path.is_file() and path.stat().st_size==record['bytes'],name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],name
    inventory=json.loads((ROOT/'history/FILES.json').read_text(encoding='utf-8'))['files']
    snapshot=json.loads((ROOT/'SOURCE_SNAPSHOTS.json').read_text(encoding='utf-8'))
    rows={r['zip_member']:r for r in inventory};assert len(rows)==len(inventory)
    assert len(rows)==snapshot['research_source_files']
    with zipfile.ZipFile(ROOT/'history/research_records.zip') as archive:
        assert set(archive.namelist())==set(rows)
        for name,record in rows.items():
            assert not name.startswith('/') and '..' not in Path(name).parts,name
            assert Path(name).suffix.lower() not in ('.pt','.pth','.npy','.npz','.tfrecords','.hdf5','.h5'),name
            raw=archive.read(name)
            assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],name
    status=json.loads((ROOT/'CURRENT_STATUS.json').read_text(encoding='utf-8'))
    assert status['selected_seeds']==[0] and status['unique_TRAIN']==3500 and status['epochs']==10
    assert len(status['rows'])==20 and len({(r['model'],r['dataset'],r['seed']) for r in status['rows']})==20
    assert all(r['native_grid']==(32 if r['model']==253 else 64) for r in status['rows'])
    assert len(list((ROOT/'history/experiments').glob('*.md')))==snapshot['experiment_folders']+6
    assert snapshot['experiment_markdown_documents']+snapshot['additional_common_markdown_documents']==562
    checked=0
    for name in ('README.md','history/EXPERIMENT_INDEX.md','history/RECENT_INDEX.md','history/OTHER_DOCUMENTS_INDEX.md'):
        path=ROOT/name;text=path.read_text(encoding='utf-8')
        text=re.sub(r'(?ms)^```.*?^```\s*$','',text)
        for url in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in url or url.startswith('#'):continue
            target=unquote(url.split('#',1)[0]).strip('<>')
            assert (path.parent/target).exists(),(name,target)
            checked+=1
    checkpoint=json.loads((ROOT/'models/legacy240_checkpoint_manifest.json').read_text(encoding='utf-8'))
    assert checkpoint['seeds']==[0,1,2]
    assert '{{' not in (ROOT/'README.md').read_text(encoding='utf-8')
    return dict(status='all_snapshot_files_archive_members_and_index_links_verified',
        generated_files=len(manifest),archived_research_files=len(rows),local_links_checked=checked,
        historical_experiment_folders=snapshot['experiment_folders'],recent_families=6,
        source_Markdown_documents=562,current_conditions=20,training_or_benchmark_code_changed=False)

if __name__=='__main__':print(json.dumps(verify(),indent=2,ensure_ascii=False))

"""Materialize exactly the registered TRAIN tensors from badly chunked HDF5 once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from collaborative_test.SW_0239_native64_best_70000 import data as reference


def spans(chunk):
    if type(chunk) is not int or chunk <= 0:
        raise ValueError('positive original image chunk length required')
    out = []
    for begin, end in ((0, 1000), (1640, 70640)):
        while begin < end:
            stop = min(end, (begin // chunk + 1) * chunk)
            out.append((begin, stop)); begin = stop
    return out


def positions(ids):
    values = reference.validate_ids(ids)
    return np.where(values < 1000, values, values - 640)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(value)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    stage = path.with_name(path.name + f'.{os.getpid()}.tmp')
    stage.write_bytes((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())
    os.replace(stage, path)


def build(dataset, output):
    import h5py
    dataset = Path(dataset); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    lease = (output / 'build.lock').open('a+')
    if os.name != 'nt':
        import fcntl
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        stat = dataset.stat()
        binding = dict(dataset=dict(path=str(dataset), size=stat.st_size, mtime_ns=stat.st_mtime_ns),
            registered_split=reference.split_binding(), registered_input_code_sha256=digest_file(Path(reference.__file__)),
            builder_code_sha256=digest_file(Path(__file__)))
        if (output / 'manifest.json').exists():
            manifest = json.loads((output / 'manifest.json').read_bytes())
            if manifest['binding'] != binding:
                raise ValueError('existing canonical TRAIN cache belongs to different source')
            for name, digest in manifest['file_sha256'].items():
                if digest_file(output/name) != digest:
                    raise ValueError('existing complete input cache changed')
            return manifest
        shapes = {'rgb.npy': (70000,128,128,3), 'labels64.npy': (70000,64,64)}
        required = sum(int(np.prod(shape)) for name,shape in shapes.items() if not (output/name).exists())
        if shutil.disk_usage(output).free < required + 2*1024**3:
            raise OSError('insufficient private canonical input cache reserve')
        progress_path = output / 'build_progress.json'
        progress = json.loads(progress_path.read_bytes()) if progress_path.exists() else dict(binding=binding, blocks=[])
        if progress['binding'] != binding:
            raise ValueError('unfinished cache source changed')
        arrays = {}
        for name,shape in shapes.items():
            path = output/name
            if path.exists():
                value = np.lib.format.open_memmap(path, mode='r+')
                if value.shape != shape or value.dtype != np.uint8:
                    raise ValueError('existing cache geometry differs')
            else:
                value = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=shape)
            arrays[name] = value
        with h5py.File(dataset, 'r') as source:
            if source['image'].shape != (100000,128,128,3) or source['mask'].shape != (100000,128,128,1):
                raise ValueError('registered canonical dataset geometry required')
            chunk = int(source['image'].chunks[0]) if source['image'].chunks else 3125
            plan = spans(chunk)
            for index,(begin,end) in enumerate(plan):
                dest = int(positions([begin])[0]); stop = dest + end-begin
                if index < len(progress['blocks']):
                    block = progress['blocks'][index]
                    if block['source_range'] != [begin,end]:
                        raise ValueError('partial cache block order changed')
                    for name in shapes:
                        if hashlib.sha256(arrays[name][dest:stop].tobytes()).hexdigest() != block['tensor_sha256'][name]:
                            raise ValueError('committed canonical cache block changed')
                    continue
                started = time.monotonic()
                rgb = reference.canonical_rgb(np.asarray(source['image'][begin:end]))
                labels = reference.modal_labels(np.asarray(source['mask'][begin:end]))
                if labels.max() > 255:
                    raise ValueError('lossless uint8 labels required; never truncate IDs')
                labels = labels.astype(np.uint8)
                values = {'rgb.npy':rgb, 'labels64.npy':labels}
                hashes = {}
                for name,value in values.items():
                    arrays[name][dest:stop] = value; arrays[name].flush()
                    digest = hashlib.sha256(value.tobytes()).hexdigest()
                    if hashlib.sha256(arrays[name][dest:stop].tobytes()).hexdigest() != digest:
                        raise ValueError('cache write differs from canonical TRAIN tensor')
                    hashes[name] = digest
                progress['blocks'].append(dict(source_range=[begin,end],destination_range=[dest,stop],
                    tensor_sha256=hashes,elapsed_seconds=time.monotonic()-started))
                write_json(progress_path, progress)
                del rgb,labels,values
        if len(progress['blocks']) != len(plan) or progress['blocks'][-1]['destination_range'][1] != 70000:
            raise ValueError('incomplete canonical TRAIN cache')
        after = dataset.stat()
        if (after.st_size,after.st_mtime_ns) != (stat.st_size,stat.st_mtime_ns):
            raise ValueError('source dataset changed during materialization')
        result = dict(status='complete_exact_registered_TRAIN70000_input_cache', binding=binding,
            blocks=progress['blocks'], file_sha256={name:digest_file(output/name) for name in shapes},
            source_evaluation_masks_read=False, RGB_and_modal_patch_labels_verified_before_each_block_commit=True,
            logical_dataset_changed=False, optimizer_updates=0, GPU_used=False)
        write_json(output/'manifest.json',result)
        return result
    finally:
        lease.close()


def read_batch(output, ids):
    output = Path(output)
    if not (output/'manifest.json').exists():
        raise ValueError('only a complete verified canonical TRAIN cache may serve batches')
    pos = positions(ids)
    rgb = np.load(output/'rgb.npy',mmap_mode='r',allow_pickle=False)[pos].copy()
    labels = np.load(output/'labels64.npy',mmap_mode='r',allow_pickle=False)[pos].astype(np.int64)
    return rgb,labels


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=build(args.dataset,args.output)
    print(json.dumps(dict(status=result['status'],TRAIN_images=70000,GPU_used=False)),flush=True)

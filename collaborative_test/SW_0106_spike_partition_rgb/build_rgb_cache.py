"""Build exact RGB-only training/validation caches without reading instance masks."""
import argparse, hashlib, json, os, time
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5')
OUT=ROOT/'data/SW_0106_spike_partition_rgb'
TRAIN='train_rgb_uint8.npy'; VALID='validation_rgb_uint8.npy'
BASE_BLOCK=256
MAX_BLOCK=4096

def sha_bytes(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def choose_block(images):
 first=images.chunks[0] if images.chunks else None
 if first is not None and first>MAX_BLOCK: raise RuntimeError(f'leading HDF5 chunk {first} exceeds bounded cache block {MAX_BLOCK}; inspect chunk layout before reading full dataset')
 return max(BASE_BLOCK,first or BASE_BLOCK)

def train_ranges(block):
 # Source IDs are ascending in the exact pool order: 0..999, then 1640..70639.
 ranges=[]; pool=0
 for source_start,source_stop in ((0,1000),(1640,70640)):
  for start in range(source_start,source_stop,block):
   count=min(block,source_stop-start); ranges.append((start,start+count,pool)); pool+=count
 if pool!=70000: raise AssertionError(f'training cache rows {pool} != 70000')
 return ranges

def source_reads(images,kind,block):
 """Yield one physical-chunk-aligned read and all selected cache pieces from it."""
 if kind=='train':
  selected=((0,1000,0),(1640,70640,1000))
  total=70000
 else:
  selected=((1320,1640,0),)
  total=320
 chunk=images.chunks[0] if images.chunks else block
 for cstart in range(0,images.shape[0],chunk):
  cstop=min(cstart+chunk,images.shape[0]); pieces=[]
  for sstart,sstop,outstart in selected:
   lo=max(cstart,sstart); hi=min(cstop,sstop)
   if lo<hi: pieces.append((lo,hi,outstart+lo-sstart))
  if pieces: yield cstart,cstop,pieces

def cache_one(dataset,kind,block):
 if kind=='train':
  filename=TRAIN; shape=(70000,128,128,3); ranges=train_ranges(block); ids=np.concatenate((np.arange(1000,dtype='<i8'),np.arange(1640,70640,dtype='<i8'))); ids_hash=hashlib.sha256(ids.tobytes()).hexdigest()
 elif kind=='validation':
  filename=VALID; shape=(320,128,128,3); ranges=[]; pool=0
  for start in range(1320,1640,block):
   count=min(block,1640-start); ranges.append((start,start+count,pool)); pool+=count
  ids_hash=hashlib.sha256(np.arange(1320,1640,dtype='<i8').tobytes()).hexdigest()
 else: raise ValueError(kind)
 final=OUT/filename; partial=OUT/(filename+'.partial')
 meta=OUT/(filename+'.complete.json'); lock=OUT/(filename+'.lock')
 if final.exists() or meta.exists() or partial.exists(): raise FileExistsError(f'existing cache artifact requires inspection: {final}')
 OUT.mkdir(parents=True,exist_ok=True)
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd,str(os.getpid()).encode()); os.close(fd)
 began=time.time(); blocks=[]
 try:
  images=dataset['image']
  if tuple(images.shape)[1:]!=(128,128,3) or images.dtype!=np.dtype('uint8'):
   raise AssertionError(f'unexpected source image shape/dtype: {images.shape}/{images.dtype}')
  src_stat=SOURCE.stat()
  info={'status':'building','kind':kind,'source_path':str(SOURCE),'source_size_bytes':src_stat.st_size,'source_mtime_ns':src_stat.st_mtime_ns,'source_image_shape':list(images.shape),'source_image_dtype':str(images.dtype),'source_chunks':images.chunks,'source_compression':images.compression,'cache_shape':list(shape),'cache_dtype':'uint8','block_size':block,'ids_mapping_sha256':ids_hash,'source_id_ranges':[[x[0],x[1]] for x in ranges],'started':began,'blocks':[]}
  with partial.open('wb') as _:
   pass
  mm=np.lib.format.open_memmap(partial,mode='w+',dtype=np.uint8,shape=shape)
  for chunk_start,chunk_stop,pieces in source_reads(images,kind,block):
   source=np.asarray(images[chunk_start:chunk_stop])
   count=chunk_stop-chunk_start
   if source.shape!=(count,128,128,3) or source.dtype!=np.uint8: raise AssertionError('read HDF5 block shape/dtype mismatch')
   for src_start,src_stop,out_start in pieces:
    piece=source[src_start-chunk_start:src_stop-chunk_start]
    piece_count=src_stop-src_start
    mm[out_start:out_start+piece_count]=piece
    mm.flush()
    copied=np.asarray(mm[out_start:out_start+piece_count])
    if not np.array_equal(copied,piece): raise AssertionError(f'byte mismatch for source IDs [{src_start},{src_stop})')
    digest=sha_bytes(piece)
    if sha_bytes(copied)!=digest: raise AssertionError(f'block hash mismatch at pool row {out_start}')
    blocks.append({'pool_start':out_start,'source_id_start':src_start,'count':piece_count,'sha256':digest,'source_chunk':[chunk_start,chunk_stop]})
   info['blocks']=blocks; info['completed_blocks']=len(blocks); info['updated']=time.time()
   tmpmeta=OUT/(filename+'.progress.tmp'); tmpmeta.write_text(json.dumps(info,indent=2)+'\n'); tmpmeta.replace(OUT/(filename+'.progress.json'))
  mm.flush(); del mm
  if sum(x['count'] for x in blocks)!=shape[0]: raise AssertionError('written row count mismatch')
  os.replace(partial,final)
  info.update(status='complete',completed=time.time(),elapsed_seconds=time.time()-began,cache_file=str(final))
  tmpmeta=OUT/(filename+'.complete.tmp'); tmpmeta.write_text(json.dumps(info,indent=2)+'\n'); tmpmeta.replace(meta)
  try: (OUT/(filename+'.progress.json')).unlink()
  except FileNotFoundError: pass
  return info
 finally:
  try: lock.unlink()
  except FileNotFoundError: pass

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['train','validation','all'],default='all');p.add_argument('--inspect-only',action='store_true');a=p.parse_args()
 selected=('train','validation') if a.kind=='all' else (a.kind,)
 with h5py.File(SOURCE,'r') as ds:
  images=ds['image']; block=choose_block(images)
  if a.inspect_only:
   rows=[]
   for start,stop,pieces in list(source_reads(images,'train',block))[:2]:
    begin=time.time(); value=np.asarray(images[start:stop]); elapsed=time.time()-begin
    rows.append({'physical_chunk_ids':[start,stop],'selected_source_pieces':[[x[0],x[1]] for x in pieces],'count':stop-start,'bytes':int(value.nbytes),'sha256':sha_bytes(value),'elapsed_seconds':elapsed,'images_per_second':(stop-start)/max(elapsed,1e-9)})
   print(json.dumps({'status':'inspect_complete','source_path':str(SOURCE),'source_image_shape':list(images.shape),'source_image_dtype':str(images.dtype),'source_chunks':images.chunks,'source_compression':images.compression,'selected_block_size':block,'first_two_block_reads':rows},indent=2),flush=True); return
  for kind in selected:
   result=cache_one(ds,kind,block);print(json.dumps({'kind':kind,'status':result['status'],'shape':result['cache_shape'],'block_size':block,'blocks':len(result['blocks']),'elapsed_seconds':result['elapsed_seconds']}),flush=True)
if __name__=='__main__': main()

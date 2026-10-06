"""Train official Slot Attention architecture on 70,000 unique HDF5 scenes."""
import argparse, csv, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
import h5py
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def load_model(path):
    path=Path(path).resolve(); sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location("official_slot_model",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def signature(weights):
    return [(tuple(int(x) for x in w.shape), str(w.dtype)) for w in weights]

p=argparse.ArgumentParser()
p.add_argument("--dataset",required=True); p.add_argument("--model-py",required=True)
p.add_argument("--output-dir",required=True); p.add_argument("--seed",type=int,required=True)
p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=64)
p.add_argument("--smoke",action="store_true")
a=p.parse_args()
if a.seed<0 or a.epochs<1 or a.batch_size<1: raise ValueError("invalid arguments")
out=Path(a.output_dir); 
if out.exists() and any(out.iterdir()): raise FileExistsError(out)
out.mkdir(parents=True,exist_ok=True); (out/"checkpoint").mkdir()

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2); tf.config.threading.set_inter_op_parallelism_threads(1)
tf.random.set_seed(a.seed); np.random.seed(a.seed)
global_ids=np.concatenate((np.arange(1000),np.arange(1640,70640))).astype(np.int64)
if a.smoke: global_ids=global_ids[:a.batch_size]; a.epochs=1
with h5py.File(a.dataset,"r") as h5:
    ds=h5["image"]
    if tuple(ds.shape[1:])!=(128,128,3) or max(global_ids)>=len(ds): raise ValueError(ds.shape)
    images=np.empty((len(global_ids),128,128,3),np.float32)
    digest=hashlib.sha256()
    for start in range(0,len(global_ids),128):
        raw=ds[global_ids[start:start+128]]; digest.update(np.ascontiguousarray(raw).tobytes())
        images[start:start+len(raw)]=raw.astype(np.float32)/127.5-1.0

model=load_model(a.model_py)
network=model.build_model((128,128),a.batch_size,11,3,model_type="object_discovery")
tail=len(global_ids)%a.batch_size
aux=None
if tail:
    aux=model.build_model((128,128),tail,11,3,model_type="object_discovery")
    if signature(network.trainable_weights)!=signature(aux.trainable_weights): raise RuntimeError("static signatures differ")
optimizer=tf.keras.optimizers.Adam(4e-4,epsilon=1e-8)
global_step=tf.Variable(0,trainable=False,dtype=tf.int64,name="global_step")
ckpt=tf.train.Checkpoint(network=network,optimizer=optimizer,global_step=global_step)
manager=tf.train.CheckpointManager(ckpt,str(out/"checkpoint"),max_to_keep=a.epochs)
steps_per_epoch=math.ceil(len(global_ids)/a.batch_size); total_steps=steps_per_epoch*a.epochs
warmup=max(1,steps_per_epoch); decay=max(1,total_steps)
protocol={"experiment":"SW0092 Slot on our 70k","seed":a.seed,"dataset":str(Path(a.dataset).resolve()),
 "training_id_segments_inclusive":[[0,999],[1640,70639]],"unique_training_images":len(global_ids),
 "epochs":a.epochs,"batch_size":a.batch_size,"steps_per_epoch":steps_per_epoch,"total_steps":total_steps,
 "num_slots":11,"iterations":3,"learning_rate":4e-4,"warmup_steps":warmup,"decay_steps":decay,
 "decay_rate":.5,"loss":"reconstruction MSE","selected_images_sha256":digest.hexdigest(),
 "checkpoint_source":"scratch official architecture","ground_truth_used_for_training":False}
(out/"training_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
rng=np.random.RandomState(a.seed); started=time.time()
with (out/"training_loss.csv").open("w",newline="") as f:
    writer=csv.writer(f); writer.writerow(["epoch","step","batch_size","learning_rate","loss","elapsed_seconds"])
    for epoch in range(1,a.epochs+1):
        order=rng.permutation(len(global_ids))
        for start in range(0,len(order),a.batch_size):
            rows=order[start:start+a.batch_size]; batch=tf.convert_to_tensor(images[rows])
            active=network
            if len(rows)!=a.batch_size:
                aux.set_weights(network.get_weights()); active=aux
            step=int(global_step.numpy()); lr=4e-4*min(step/warmup,1.0)*(.5**(step/decay)); optimizer.learning_rate.assign(lr)
            with tf.GradientTape() as tape:
                reconstruction,_,_,_=active(batch,training=True)
                loss=tf.reduce_mean(tf.square(batch-reconstruction))
            grads=tape.gradient(loss,active.trainable_weights)
            if any(g is None for g in grads) or not math.isfinite(float(loss.numpy())): raise RuntimeError("invalid training step")
            optimizer.apply_gradients(zip(grads,network.trainable_weights)); global_step.assign_add(1)
            writer.writerow([epoch,int(global_step.numpy()),len(rows),lr,float(loss.numpy()),time.time()-started])
        f.flush(); saved=manager.save(checkpoint_number=int(global_step.numpy()))
        print(f"EPOCH {epoch}/{a.epochs} step={int(global_step.numpy())} loss={float(loss.numpy()):.8f} checkpoint={saved}",flush=True)
(out/"TRAINING_COMPLETED").write_text(f"steps={int(global_step.numpy())}\n")


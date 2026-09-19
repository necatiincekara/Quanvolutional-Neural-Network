#!/usr/bin/env python3
"""Supplemental E3 stress: create overflow by scaling, not by injecting Inf.

This is a controlled stress test, not a reconstruction of historical L4 tensors.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.diagnose_hybrid_precision import TinyHybrid,summary,grad_snapshot
from src.classical_quantum_controls import image_patches
from src.research_protocol import load_ids,write_new,run_record,validate_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);args=p.parse_args()
    base=args.root;torch.set_num_threads(1);t0=time.perf_counter();c0=time.process_time()
    manifest=json.loads((base/'dataset_manifest.json').read_text());protocol=json.loads((base/'protocol.json').read_text())
    images,_=load_ids(manifest,protocol,protocol['train_ids'][:8]);x=image_patches(images)[:,117]
    y=torch.tensor([0,1,2,0,1,2,0,1]);rows=[]
    for family in ('analytic','fixed','v7'):
        for semantics in ('correct','historical_direct_after_unscale'):
            model=TinyHybrid(family,0)
            groups=[[model.angle_parameter()],list(model.stem.parameters())+list(model.head.parameters())]
            opts=[torch.optim.Adam(groups[0],lr=.0005),torch.optim.AdamW(groups[1],lr=.002)]
            scaler=torch.amp.GradScaler('cpu',init_scale=2.**24)
            with torch.autocast('cpu',dtype=torch.float16):
                logits=model(x,'float32');loss=torch.nn.functional.cross_entropy(logits,y)
            scaler.scale(loss).backward();scaled=grad_snapshot(model)
            first=model.first_bad
            for opt in opts:scaler.unscale_(opt)
            unscaled=grad_snapshot(model)
            for group,limit in zip(groups,(.5,1.)):torch.nn.utils.clip_grad_norm_(group,limit)
            counts=[0,0]
            for i,opt in enumerate(opts):
                opt.register_step_post_hook(lambda o,a,k,i=i:counts.__setitem__(i,counts[i]+1))
                if semantics=='correct':scaler.step(opt)
                else:opt.step()
            scaler.update()
            rows.append({'family':family,'semantics':semantics,'loss_scale_initial':2.**24,
                         'loss_scale_after':scaler.get_scale(),'forward_loss':float(loss.detach()),
                         'first_nonfinite_observed':first,'scaled_gradients':scaled,'unscaled_gradients':unscaled,
                         'optimizer_steps_executed':counts,'parameters_finite':all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                         'boundaries':{k:summary(v) for k,v in model.trace.items()},'manual_inf_injection':False})
    record=run_record('astra_e3_scaling_stress_20260915','diagnostic',manifest,protocol,
                      ['scripts/stress_hybrid_scaling.py','scripts/diagnose_hybrid_precision.py'],
                      manifest_path=base/'dataset_manifest.json')
    record['architecture']={'name':'tiny_hybrid_scaling_stress','version':'E3_supplement_v1'}
    record['data']['train_ids']=protocol['train_ids'][:8];record['data']['evaluation_split']='diagnostic'
    record['seeds']['model_init_seed']=0
    record['hyperparameters']={'initial_loss_scale':2**24,'quantum_boundary':'explicit float32','families':['analytic','fixed','v7'],'steps_per_case':1,'targets':'synthetic 3-class labels'}
    record['precision']={'dtype':'CPU autocast float16 with float32 Q boundary','amp':True,'scaler':True}
    record['quantum']={'backend':'default.qubit or analytic','differentiation':'backprop'}
    record['parameter_counts']={'trainable':59,'frozen':0,'note':'maximum; fixed/analytic 47, V7 59'}
    record['optimizer']={'q':'Adam .0005','classical':'AdamW .002'}
    record['gradient_diagnostics']={'status':'recorded in scaling_stress.json'}
    record['status']='completed';record['limitations']=['Stress scale differs from historical default; does not identify original L4 overflow trigger.','Single controlled update, not OCR training.']
    record['runtime']={'wall_seconds':time.perf_counter()-t0,'process_cpu_seconds':time.process_time()-c0}
    validate_run(record);write_new(base/'e3/scaling_stress_run.json',record)
    write_new(base/'e3/scaling_stress.json',rows)
    print(json.dumps([{k:r[k] for k in ('family','semantics','first_nonfinite_observed','parameters_finite','optimizer_steps_executed')} for r in rows],indent=2))


if __name__=='__main__':main()

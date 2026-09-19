#!/usr/bin/env python3
"""E3 tiny controlled numerical diagnostic. Never trains the full V7 network."""
import argparse
import ast
import contextlib
import copy
import json
from pathlib import Path
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pennylane as qml
import torch
from src.classical_quantum_controls import fixed_expectations,image_patches
from src.research_protocol import load_ids,run_record,validate_run,write_new


def source_function(v7):
    p=ROOT/('src/trainable_quantum_model.py' if v7 else 'train_ablation_local.py')
    name='data_reuploading_circuit' if v7 else 'fixed_circuit'
    node=next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    namespace={'qml':qml,'n_qubits':4}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),'exec'),namespace)
    return namespace[name]


def summary(tensor):
    if tensor is None:return {'missing':True}
    a=tensor.detach().cpu().double().flatten();finite=torch.isfinite(a)
    vals=a[finite]
    return {'dtype':str(tensor.dtype),'shape':list(tensor.shape),'finite_mask':finite.tolist(),
            'nan_count':int(torch.isnan(a).sum()),'inf_count':int(torch.isinf(a).sum()),
            'norm':float(a.norm()) if finite.all() else None,
            'mean_finite':float(vals.mean()) if len(vals) else None,
            'std_finite':float(vals.std(unbiased=False)) if len(vals) else None,
            'values':[float(x) if ok else None for x,ok in zip(a,finite)]}


class TinyHybrid(torch.nn.Module):
    def __init__(self,family,seed,backend='default.qubit',method='backprop'):
        super().__init__();self.family=family;torch.manual_seed(seed)
        self.stem=torch.nn.Linear(4,4);self.head=torch.nn.Linear(4,3)
        shape=(2,4,3) if family=='v7' else (4,3)
        if family=='analytic':self.angles=torch.nn.Parameter(torch.empty(shape).uniform_(-.1,.1))
        else:
            node=qml.QNode(source_function(family=='v7'),qml.device(backend,wires=4),interface='torch',diff_method=method)
            self.q=torch.nn.Module() # replaced with actual production interface type
            self.q=qml.qnn.TorchLayer(node,{'weights':shape})
            with torch.no_grad():self.q.weights.uniform_(-.1,.1)
        # Force identical function-space initialization in fixed Q / analytic arms.
        g=torch.Generator().manual_seed(seed+919)
        with torch.no_grad():self.angle_parameter().copy_(torch.rand(shape,generator=g)*.2-.1)
        self.trace={};self.first_bad=None

    def angle_parameter(self):return self.angles if self.family=='analytic' else self.q.weights

    def inspect(self,name,t):
        self.trace[name]=t
        if not torch.isfinite(t).all() and self.first_bad is None:self.first_bad='forward:'+name
        if t.requires_grad:
            t.retain_grad()
            def hook(g):
                if not torch.isfinite(g).all() and self.first_bad is None:self.first_bad='backward:'+name
            t.register_hook(hook)
        return t

    def forward(self,x,boundary='inherit'):
        self.trace={};self.first_bad=None
        z=self.inspect('stem_output',self.stem(x))
        if boundary=='float32':z=z.float()
        z=self.inspect('quantum_input',z)
        with torch.autocast('cpu',enabled=False) if boundary=='float32' else contextlib.nullcontext():
            y=fixed_expectations(z,self.angles.to(z.dtype)) if self.family=='analytic' else self.q(z)
        y=self.inspect('quantum_output',y)
        return self.inspect('head_output',self.head(y))


def grad_snapshot(model):
    return {name:summary(p.grad) for name,p in model.named_parameters()}


def cell(family,seed,setting,x,y,steps):
    model=TinyHybrid(family,seed)
    if setting['dtype']=='float64':model.double()
    x=x.to(next(model.parameters()).dtype)
    qp=[model.angle_parameter()];cp=list(model.stem.parameters())+list(model.head.parameters())
    opts=[torch.optim.Adam(qp,lr=.0005),torch.optim.AdamW(cp,lr=.002)]
    scaler=torch.amp.GradScaler('cpu',enabled=setting['scaler'],init_scale=65536.)
    logs=[];status='completed';error=None;captured=[];t0=time.perf_counter();c0=time.process_time()
    try:
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter('always')
            for step in range(steps):
                reference=TinyHybrid(family,seed).double();reference.load_state_dict(model.state_dict())
                ref_loss=torch.nn.functional.cross_entropy(reference(x.double()),y)
                ref_loss.backward();reference_grads=grad_snapshot(reference)
                for opt in opts:opt.zero_grad()
                dtype=getattr(torch,setting['dtype'])
                with torch.autocast('cpu',dtype=dtype,enabled=setting['amp']):
                    logits=model(x,setting['boundary']);loss=torch.nn.functional.cross_entropy(logits,y)
                scale=scaler.get_scale();scaler.scale(loss).backward()
                injected=setting.get('inject_inf_step')==step
                if injected:
                    qp[0].grad.flatten()[0]=float('inf')
                    if model.first_bad is None:model.first_bad='controlled_injected_parameter_gradient'
                scaled=grad_snapshot(model)
                q_input_scaled=summary(model.trace['quantum_input'].grad)
                q_input_unscaled=summary(model.trace['quantum_input'].grad/scale)
                for opt in opts:scaler.unscale_(opt)
                unscaled=grad_snapshot(model)
                before={name:p.detach().clone() for name,p in model.named_parameters()}
                finite_groups=[all(p.grad is None or torch.isfinite(p.grad).all() for p in group) for group in (qp,cp)]
                clips=[]
                for group,threshold in ((qp,.5),(cp,1.)):
                    norm=torch.nn.utils.clip_grad_norm_(group,threshold)
                    clips.append({'norm_before_clip':float(norm) if torch.isfinite(norm) else None,
                                  'clipped':bool(norm>threshold),'threshold':threshold})
                step_counts=[0,0]
                handles=[o.register_step_post_hook(lambda opt,args,kwargs,i=i:step_counts.__setitem__(i,step_counts[i]+1)) for i,o in enumerate(opts)]
                if setting['semantics']=='correct':
                    for opt in opts:scaler.step(opt)
                else:
                    # Exactly the historical issue: gradients were already unscaled,
                    # but direct optimizer stepping bypasses scaler finite-step skipping.
                    for opt in opts:opt.step()
                scaler.update()
                for handle in handles:handle.remove()
                updates={name:summary(p.detach()-before[name]) for name,p in model.named_parameters()}
                parameter_stats={name:summary(p) for name,p in model.named_parameters()}
                params_finite=all(torch.isfinite(p).all() for p in model.parameters())
                if not params_finite and model.first_bad is None:model.first_bad='optimizer_update'
                logs.append({'step':step,'loss':float(loss.detach()) if torch.isfinite(loss) else None,
                             'reference_loss':float(ref_loss.detach()),'reference_gradients':reference_grads,
                             'scaled_gradients':scaled,'unscaled_gradients':unscaled,
                             'quantum_input_gradient_scaled':q_input_scaled,'quantum_input_gradient_unscaled':q_input_unscaled,
                             'reference_quantum_input_gradient':summary(reference.trace['quantum_input'].grad),
                             'clipping':clips,'finite_groups_before_clip':list(map(bool,finite_groups)),
                             'loss_scale_before':scale,'loss_scale_after':scaler.get_scale(),
                             'optimizer_steps_executed':step_counts,'skipped_optimizer_steps':[c==0 for c in step_counts],
                             'observed_boundaries':{k:summary(v) for k,v in model.trace.items()},
                             'parameter_stats':parameter_stats,'update_stats':updates,'parameters_finite':bool(params_finite),
                             'first_nonfinite_observed':model.first_bad,'deliberate_injection':injected})
                if not params_finite:status='parameter_corruption';break
            captured=sorted({str(w.message) for w in warns})
    except Exception as exc:
        status='unsupported_or_execution_error';error=repr(exc)
    return {'family':family,'seed':seed,'setting':setting,'status':status,'error':error,
            'parameter_counts':{'circuit_or_analytic':model.angle_parameter().numel(),'classical':35},
            'warnings':captured,'steps':logs,'wall_seconds':time.perf_counter()-t0,'process_cpu_seconds':time.process_time()-c0}


def derivative_checks():
    rows=[]
    for v7 in (False,True):
        x=torch.tensor([.2,-.4,.7,1.1],dtype=torch.float64,requires_grad=True)
        shape=(2,4,3) if v7 else (4,3)
        gen=torch.Generator().manual_seed(19)
        w=(torch.rand(shape,generator=gen,dtype=torch.float64)*.2-.1).requires_grad_()
        reference=None
        for backend,method in [('default.qubit','backprop'),('default.qubit','parameter-shift'),('lightning.qubit','adjoint')]:
            row={'circuit':'v7' if v7 else 'fixed','backend':backend,'method':method}
            try:
                node=qml.QNode(source_function(v7),qml.device(backend,wires=4),interface='torch',diff_method=method)
                fn=lambda a,b:torch.stack(node(a,b))
                value=fn(x,w);jac=torch.autograd.functional.jacobian(fn,(x,w))
                if reference is None:reference=(value,jac)
                row.update(status='passed',output_dtype=str(value.dtype),input_jacobian_norm=float(jac[0].norm()),
                           angle_jacobian_norm=float(jac[1].norm()),
                           max_input_gradient_difference=float((jac[0]-reference[1][0]).abs().max()),
                           max_angle_gradient_difference=float((jac[1]-reference[1][1]).abs().max()))
                assert max(row['max_input_gradient_difference'],row['max_angle_gradient_difference'])<1e-8
            except Exception as exc:row.update(status='unsupported_or_failed',error=repr(exc))
            rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();args.root=args.root.resolve();cfg=json.loads(args.config.read_text())
    torch.set_num_threads(1);start=time.perf_counter();cpu=time.process_time()
    assert json.loads((args.root/'e0_checks.json').read_text())['passed']
    manifest=json.loads((args.root/'dataset_manifest.json').read_text());protocol=json.loads((args.root/'protocol.json').read_text())
    out=args.root/'e3';out.mkdir(exist_ok=False)
    images,_=load_ids(manifest,protocol,protocol['train_ids'][:8]);patches=image_patches(images)[:,117,:]
    y=torch.tensor([0,1,2,0,1,2,0,1])
    record=run_record('astra_e3_precision_20260915','diagnostic',manifest,protocol,
                      ['scripts/diagnose_hybrid_precision.py','src/trainable_quantum_model.py','src/classical_quantum_controls.py'],
                      manifest_path=args.root/'dataset_manifest.json')
    record['architecture']={'name':'tiny_linear_stem_existing_circuit_linear_head','version':'E3_v1'}
    record['data']['evaluation_split']='diagnostic';record['hyperparameters']=cfg
    record['data']['train_ids']=protocol['train_ids'][:8]
    record['hyperparameters']['targets']='synthetic 0,1,2,0,1,2,0,1; not OCR labels'
    record['parameter_counts']={'trainable':59,'frozen':0,'note':'maximum across cells; exact 47 or 59 per-cell counts recorded'}
    record['precision']={'dtype':'multiple observed dtypes; see cells','amp':True,'scaler':True}
    record['quantum']={'backend':'default.qubit; lightning.qubit derivative check','differentiation':'backprop; parameter-shift and adjoint derivative checks'}
    record['optimizer']={'quantum':'Adam lr .0005','classical':'AdamW lr .002'}
    record['gradient_diagnostics']={'status':'per-step traces in cell JSON files'}
    record['limitations']=['CPU only; not a recreation of unavailable L4/CUDA runtime.','Deliberate Inf-injection controls establish unsafe update behavior, not the origin of historical Inf.','State-vector internal dtype is not instrumented; Q inputs, outputs, parameters and derivatives are observed.','Ten updates per cell; no claim about converged OCR or barren plateaus.']
    checks=derivative_checks();rows=[]
    for family in cfg['families']:
        for setting in cfg['settings']:
            for seed in cfg['seeds']:
                if time.process_time()-cpu>cfg['cpu_cap_seconds']:raise RuntimeError('E3 cap exceeded')
                result=cell(family,seed,setting,patches,y,cfg['steps'])
                name=f'{family}_{setting["name"]}_seed{seed}.json';write_new(out/name,result)
                rows.append({k:v for k,v in result.items() if k not in ('steps','warnings')})
                print(name,result['status'],flush=True)
    record['runtime']={'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu}
    record['status']='completed';validate_run(record);write_new(out/'run.json',record)
    write_new(out/'precision_summary.json',{'status':'completed','device':'cpu','cells':rows,
                                          'derivative_checks':checks,'cuda_screen':'not_run_no_cuda_available',
                                          'historical_causal_attribution':'partially_resolved; original overflow trigger on L4 remains unverified'})


if __name__=='__main__':main()

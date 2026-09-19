#!/usr/bin/env python3
"""E2 preregistered small validation pilot, sealed historical final test."""
import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import stats
import torch
from src.research_protocol import digest,file_hash,load_ids,metrics,run_record,validate_run,write_new
from src.research_feature_maps import PatchMap,pooled,FAMILIES


def subset_indices(ids,labels,fraction,seed):
    selected=[]
    for cls in sorted(set(labels.tolist())):
        members=[i for i in range(len(ids)) if int(labels[i])==cls]
        members.sort(key=lambda i:digest([seed,ids[i]]))
        selected.extend(members[:max(1,round(fraction*len(members)))])
    return sorted(selected)


def alc(rows,key='macro_f1'):
    rows=sorted(rows,key=lambda r:r['train_count'])
    x=np.log([r['train_count'] for r in rows]);y=[r['metrics'][key] for r in rows]
    return float(np.trapezoid(y,x)/(x[-1]-x[0]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path);args=parser.parse_args();args.root=args.root.resolve()
    cfg=json.loads(args.config.read_text());start=time.perf_counter();cpu=time.process_time();torch.set_num_threads(1)
    assert json.loads((args.root/'e0_checks.json').read_text())['passed']
    assert json.loads((args.root/'e1_verified/replay_results.json').read_text())['status']=='completed'
    out=args.root/'e2';out.mkdir(exist_ok=False)
    manifest=json.loads((args.root/'dataset_manifest.json').read_text());protocol=json.loads((args.root/'protocol.json').read_text())
    train_ids=protocol['train_ids'];val_ids=protocol['validation_ids']
    images,labels=load_ids(manifest,protocol,train_ids+val_ids);n=len(train_ids)
    tuning=[];assessment=[]
    for cls in sorted(set(labels[n:].tolist())):
        members=[i for i in range(n,len(images)) if int(labels[i])==cls]
        members.sort(key=lambda i:digest(['E2-validation-partition',val_ids[i-n]]))
        k=max(1,len(members)//2)
        tuning+=members[:k];assessment+=members[k:]
    write_new(out/'validation_partition.json',{'tuning_ids':[(train_ids+val_ids)[i] for i in tuning],
                                              'assessment_ids':[(train_ids+val_ids)[i] for i in assessment],
                                              'test_ids_accessed':False,'config_sha256':file_hash(args.config)})
    feature_cache={};rows=[]

    def fit(family,bank_seed,subset_seed,train_seed,fraction,stage):
        if time.process_time()-cpu>cfg['cpu_cap_seconds']:raise RuntimeError('E2 stage CPU cap reached')
        idx=subset_indices(train_ids,labels[:n],fraction,subset_seed)
        key=(family,bank_seed);m=PatchMap(family,bank_seed)
        if key not in feature_cache:
            with torch.no_grad():feature_cache[key]=torch.cat([pooled(m(images[i:i+128])) for i in range(0,len(images),128)])
        feats=feature_cache[key]
        mean=feats[idx].mean(0);scale=feats[idx].std(0,unbiased=False).clamp_min(1e-4)
        scaled=(feats-mean)/scale
        torch.manual_seed(train_seed+cfg['readout_init_offset']);head=torch.nn.Linear(256,44)
        learn=family=='learned_conv';params=list(head.parameters())+(list(m.parameters()) if learn else [])
        opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        gen=torch.Generator().manual_seed(train_seed);best=-1.;best_state=None;epochs=[]
        rid=f'{stage}_{family}_b{bank_seed}_s{subset_seed}_t{train_seed}_f{int(fraction*100):03d}'
        record=run_record(rid,'training',manifest,protocol,['src/research_protocol.py','src/research_feature_maps.py','scripts/run_feature_probe.py'],
                          manifest_path=args.root/'dataset_manifest.json')
        record['data']['train_ids']=[train_ids[i] for i in idx]
        record['architecture']={'name':family+'_common_pooled_linear','version':'E2_pilot_v1'}
        record['seeds']={'train_seed':train_seed,'model_init_seed':train_seed+cfg['readout_init_offset'],
                         'split_seed':protocol['split_seed'],'subset_seed':subset_seed,'filter_bank_seed':bank_seed}
        record['hyperparameters']={'config':cfg,'config_sha256':file_hash(args.config),'fraction':fraction,
                                  'stage':stage,'initial_feature_bank_hash':digest(feats.numpy().tobytes()),
                                  'normalization':'per-coordinate mean/std on this training subset, fixed during learning',
                                  'validation_partition':str((out/'validation_partition.json').relative_to(ROOT))}
        record['parameter_counts']={'trainable':sum(p.numel() for p in params),'frozen':sum(b.numel() for b in m.buffers())}
        record['optimizer']={'name':'AdamW','lr':cfg['lr'],'weight_decay':cfg['weight_decay']}
        record['checkpoint_criterion']='maximum tuning-validation 44-class macro-F1; earliest tie'
        record['quantum']={'backend':'analytic classical execution' if family=='analytic_fixed' else 'not_applicable','differentiation':'torch autograd for readout; frozen feature map' if not learn else 'torch autograd'}
        record['limitations']=['Validation-only exploratory pilot; assessment labels never select checkpoint/family.','Historical 466-example final test not loaded.','Shared small readout differs from historical residual classifier.','Learned convolution is the raw-patch 1-to-16 counterpart, not the historical four-channel learned-stem full network.']
        t0=time.perf_counter();c0=time.process_time()
        for epoch in range(cfg['epochs']):
            order=torch.tensor(idx)[torch.randperm(len(idx),generator=gen)]
            total_loss=0.;gradnorms=[]
            for b in order.split(cfg['batch_size']):
                opt.zero_grad()
                z=(pooled(m(images[b]))-mean)/scale if learn else scaled[b]
                loss=torch.nn.functional.cross_entropy(head(z),labels[b]);loss.backward()
                gn=float(torch.nn.utils.clip_grad_norm_(params,1.));gradnorms.append(gn)
                assert torch.isfinite(loss) and math.isfinite(gn)
                opt.step();total_loss+=float(loss.detach())*len(b)
            with torch.no_grad():
                z=(pooled(m(images[tuning]))-mean)/scale if learn else scaled[tuning]
                tm=metrics(labels[tuning],head(z).argmax(1))
            epochs.append({'epoch':epoch+1,'train_loss':total_loss/len(idx),'tuning_macro_f1':tm['macro_f1'],
                           'tuning_accuracy':tm['accuracy'],'classical_grad_norm_mean':float(np.mean(gradnorms)),
                           'clip_fraction':float(np.mean(np.array(gradnorms)>1))})
            if tm['macro_f1']>best:
                best=tm['macro_f1'];best_state={'head':copy.deepcopy(head.state_dict()),'map':copy.deepcopy(m.state_dict()),'epoch':epoch+1}
        head.load_state_dict(best_state['head']);m.load_state_dict(best_state['map'])
        ev=tuning if stage=='screen' else assessment
        with torch.no_grad():
            z=(pooled(m(images[ev]))-mean)/scale if learn else scaled[ev]
            logits=head(z);fm=metrics(labels[ev],logits.argmax(1))
        pred=out/(rid+'_predictions.npz');checkpoint=out/(rid+'.pth')
        np.savez_compressed(pred,sample_ids=np.array([(train_ids+val_ids)[i] for i in ev]),targets=labels[ev].numpy(),logits=logits.numpy())
        torch.save({**best_state,'normalization_mean':mean,'normalization_scale':scale},checkpoint)
        record['epoch_metrics']=epochs;record['final_metrics']=fm;record['status']='completed'
        record['gradient_diagnostics']={'status':'finite','quantum_angles':'not_applicable_frozen_map','classical':'epoch mean preclip norms recorded'}
        record['prediction_artifacts']=[str(pred.relative_to(ROOT)),str(checkpoint.relative_to(ROOT))]
        record['runtime']={'wall_seconds':time.perf_counter()-t0,'process_cpu_seconds':time.process_time()-c0}
        validate_run(record);write_new(out/(rid+'.json'),record)
        row={'id':rid,'family':family,'bank_seed':bank_seed,'subset_seed':subset_seed,'train_seed':train_seed,
             'fraction':fraction,'train_count':len(idx),'stage':stage,'metrics':fm,'runtime':record['runtime']}
        rows.append(row);print(rid,round(fm['macro_f1'],3),flush=True)

    for fam,bank,frac in itertools.product(FAMILIES,cfg['screen_bank_seeds'],cfg['fractions']):
        fit(fam,bank,cfg['screen_subset_seed'],cfg['screen_train_seed'],frac,'screen')
    screen_scores={f:float(np.mean([alc([r for r in rows if r['family']==f and r['bank_seed']==b]) for b in cfg['screen_bank_seeds']])) for f in FAMILIES}
    winner=max((f for f in FAMILIES if f!='analytic_fixed'),key=lambda f:screen_scores[f])
    write_new(out/'screen_selection.json',{'scores_macro_f1_log_count_alc':screen_scores,'selected_classical':winner,
                                         'assessment_not_evaluated_yet':True,'selection_rule':'largest mean tuning ALC across both predeclared screen banks'})
    # Exactly two model families, not a full crossed seven-control benchmark.
    for fam,bank,sub,tr,frac in itertools.product(('analytic_fixed',winner),cfg['pilot_bank_seeds'],cfg['subset_seeds'],cfg['train_seeds'],cfg['fractions']):
        fit(fam,bank,sub,tr,frac,'pilot')
    cells=[]
    for sub,bank,tr in itertools.product(cfg['subset_seeds'],cfg['pilot_bank_seeds'],cfg['train_seeds']):
        values={}
        for fam in ('analytic_fixed',winner):
            rr=[r for r in rows if r['stage']=='pilot' and r['family']==fam and r['subset_seed']==sub and r['bank_seed']==bank and r['train_seed']==tr]
            values[fam]={'macro_f1_alc':alc(rr),'accuracy_alc':alc(rr,'accuracy')}
        cells.append({'subset_seed':sub,'bank_seed':bank,'train_seed':tr,'values':values,
                      'difference':values['analytic_fixed']['macro_f1_alc']-values[winner]['macro_f1_alc']})
    arr=np.array([r['difference'] for r in cells]).reshape(len(cfg['subset_seeds']),len(cfg['pilot_bank_seeds']),len(cfg['train_seeds']))
    rng=np.random.default_rng(cfg['bootstrap_seed']);boot=[]
    for _ in range(5000):
        inds=[rng.integers(0,n,n) for n in arr.shape];boot.append(arr[np.ix_(*inds)].mean())
    ci=np.quantile(boot,[.025,.975]).tolist();grand=arr.mean()
    effects=[arr.mean(axis=tuple(j for j in range(3) if j!=i))-grand for i in range(3)]
    main=sum(e.reshape(tuple(len(e) if j==i else 1 for j in range(3))) for i,e in enumerate(effects))
    decomposition={name:float((e**2).mean()) for name,e in zip(['subset','bank','training'],effects)}
    decomposition['interactions']=float(((arr-grand-main)**2).mean())
    decision='CONDITIONAL GO' if grand>=cfg['meaningful_margin_points'] and ci[0]>0 else 'NO-GO for powered confirmation now'
    report={'status':'completed','selected_classical':winner,'screen_scores':screen_scores,'pilot_cells':cells,
            'mean_q_minus_control_macro_f1_alc':float(grand),'crossed_bootstrap_95_interval':ci,
            'descriptive_functional_variance_components':decomposition,
            'uncertainty_limit':'3 subset, 3 bank and 2 training levels; crossed bootstrap descriptive, shared assessment sample uncertainty not included; not confirmation',
            'nominal_pair_count_not_independent':len(cells),'go_no_go':decision,
            'normal_approx_pairs_for_2_point_effect_exploratory':math.ceil(7.84*float(arr.std(ddof=1))**2/4),
            'power_caveat':'Illustrative only; cells share factors. Independent replication would require variance-component design; no confirmation run.',
            'number_fits':len(rows),'test_accessed':False,'runtime':{'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu}}
    write_new(out/'pilot_summary.json',report)


if __name__=='__main__':main()

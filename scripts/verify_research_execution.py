#!/usr/bin/env python3
"""Post-run evidence verification and explicitly secondary uncertainty checks."""
import argparse
import collections
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pennylane as qml
import torch
from src.research_protocol import metrics,validate_run,write_new,file_hash,load_ids
from src.research_feature_maps import PatchMap,pooled


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();base=args.root;cpu=time.process_time();wall=time.perf_counter();torch.set_num_threads(1)
    runs=[]
    for p in sorted((base/'e2').glob('*.json')):
        r=json.loads(p.read_text())
        if 'experiment_id' not in r:continue
        validate_run(r);pred=np.load(ROOT/r['prediction_artifacts'][0])
        m=metrics(pred['targets'],pred['logits'].argmax(1))
        for key in ('accuracy','macro_f1','balanced_accuracy'):
            assert abs(m[key]-r['final_metrics'][key])<1e-10
        runs.append(r)
    e1=json.loads((base/'e1_verified/replay_results.json').read_text())
    for r in e1['checkpoints']:
        p=np.load(ROOT/r['predictions'])
        assert np.array_equal(p['analytic_logits'].argmax(1),p['reference_logits'].argmax(1))
        assert abs(metrics(p['targets'],p['analytic_logits'].argmax(1))['accuracy']-r['reported_accuracy'])<=.011
    # Independently reload two saved readouts/maps and predict from raw validation pixels.
    manifest=json.loads((base/'dataset_manifest.json').read_text());protocol=json.loads((base/'protocol.json').read_text())
    selected=[]
    for family in ('analytic_fixed','rff'):
        r=next(r for r in runs if r['experiment_id'].startswith('pilot_'+family))
        pred=np.load(ROOT/r['prediction_artifacts'][0]);cp=torch.load(ROOT/r['prediction_artifacts'][1],weights_only=True)
        ids=pred['sample_ids'].tolist();x,y=load_ids(manifest,protocol,ids)
        model=PatchMap(family,r['seeds']['filter_bank_seed']);model.load_state_dict(cp['map'])
        head=torch.nn.Linear(256,44);head.load_state_dict(cp['head'])
        with torch.no_grad():logits=head((pooled(model(x))-cp['normalization_mean'])/cp['normalization_scale'])
        err=float((logits-torch.tensor(pred['logits'])).abs().max());assert err<1e-6
        selected.append({'run':r['experiment_id'],'raw_to_saved_logit_max_abs':err})
    # Assessment-image uncertainty was not included in the primary crossed bootstrap.
    # This secondary sensitivity analysis does not alter selection or rerun training.
    pilot=[r for r in runs if r['hyperparameters']['stage']=='pilot']
    index={r['experiment_id']:r for r in pilot}
    cfg=json.loads((ROOT/'research/configs/e2_pilot.json').read_text());families=('analytic_fixed','rff')
    S,B,T,F=len(cfg['subset_seeds']),len(cfg['pilot_bank_seeds']),len(cfg['train_seeds']),len(cfg['fractions'])
    predictions=[];counts=[];target=None
    for sub in cfg['subset_seeds']:
        for bank in cfg['pilot_bank_seeds']:
            for tr in cfg['train_seeds']:
                for family in families:
                    for frac in cfg['fractions']:
                        rid=f'pilot_{family}_b{bank}_s{sub}_t{tr}_f{int(frac*100):03d}'
                        r=index[rid];p=np.load(ROOT/r['prediction_artifacts'][0]);y=p['targets']
                        if target is None:target=y
                        assert np.array_equal(target,y)
                        predictions.append(p['logits'].argmax(1));counts.append(len(r['data']['train_ids']))
    predictions=np.stack(predictions);counts=np.array(counts).reshape(S,B,T,2,F)
    codes=target[None,:]*44+predictions
    def f1_for_weights(weights):
        result=[]
        for row in codes:
            matrix=np.bincount(row,weights=weights,minlength=44*44).reshape(44,44)
            tp=np.diag(matrix);denom=matrix.sum(0)+matrix.sum(1)
            f1=np.divide(2*tp,denom,out=np.zeros(44),where=denom>0)
            result.append(100*f1.mean())
        return np.array(result).reshape(S,B,T,2,F)
    rng=np.random.default_rng(20260916);boot=[]
    logn=np.log(counts);denom=logn[...,-1]-logn[...,0]
    for _ in range(1000):
        weights=np.bincount(rng.integers(0,len(target),len(target)),minlength=len(target))
        f1=f1_for_weights(weights)
        a=np.trapezoid(f1,logn,axis=-1)/denom
        delta=a[:,:,:,0]-a[:,:,:,1]
        ix=[rng.integers(0,n,n) for n in (S,B,T)]
        boot.append(float(delta[np.ix_(*ix)].mean()))
    by_fraction=[]
    for family in families:
        for frac in cfg['fractions']:
            rr=[r for r in pilot if r['architecture']['name'].startswith(family+'_') and r['hyperparameters']['fraction']==frac]
            # At full data subset seeds encode the same training set; report six unique fits.
            if frac==1.:rr=[r for r in rr if r['seeds']['subset_seed']==cfg['subset_seeds'][0]]
            by_fraction.append({'family':family,'fraction':frac,'training_count':len(rr[0]['data']['train_ids']),
                                'crossed_cells':len(rr),**{k:{'mean':float(np.mean([r['final_metrics'][k] for r in rr])),
                                'sd':float(np.std([r['final_metrics'][k] for r in rr],ddof=1))}
                                for k in ('accuracy','macro_f1','balanced_accuracy')}})
    write_new(base/'e2/secondary_uncertainty.json',{'status':'posthoc_sensitivity_not_confirmatory',
            'paired_image_plus_crossed_factor_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),
            'bootstrap_replicates':1000,'assessment_samples':len(target),'fraction_metrics':by_fraction,
            'limitations':'Example bootstrap assumes independent examples; near-duplicate/writer groups remain unknown. Sparse/absent classes affect macro-F1 bootstrap; no test set used.'})
    probes=[]
    for dtype in (torch.float16,torch.float32,torch.bfloat16):
        for name,fn in [('pennylane_RZ_matrix',lambda:qml.RZ.compute_matrix(torch.tensor(.1,dtype=dtype))),
                        ('torch_complex_exp',lambda:torch.exp(torch.tensor(.1,dtype=dtype)*1j))]:
            try:
                value=fn();probes.append({'operation':name,'dtype':str(dtype),'status':'supported','output_dtype':str(value.dtype)})
            except Exception as exc:probes.append({'operation':name,'dtype':str(dtype),'status':'unsupported','error':repr(exc)})
    write_new(base/'e3/operator_probe.json',{'probes':probes,'device':'cpu','purpose':'Reduce inherited-float16 exception to primitive operation; no inference about CUDA support'})
    e3=json.loads((base/'e3/precision_summary.json').read_text());aggregates=[]
    for family in ('analytic','fixed','v7'):
        for setting in json.loads((ROOT/'research/configs/precision_screen.json').read_text())['settings']:
            rr=[json.loads((base/'e3'/f'{family}_{setting["name"]}_seed{s}.json').read_text()) for s in range(5)]
            aggregates.append({'family':family,'setting':setting['name'],
                               'statuses':dict(collections.Counter(r['status'] for r in rr)),
                               'any_natural_nonfinite':any(step['first_nonfinite_observed'] is not None and not step['deliberate_injection'] for r in rr for step in r['steps'])})
    write_new(base/'e3/aggregate_checks.json',aggregates)
    before=json.loads((base/'before_state.json').read_text())
    changed=[r['path'] for r in before['files'] if not (ROOT/r['path']).exists() or file_hash(ROOT/r['path'])!=r['sha256']]
    assert not changed,changed
    result={'all_passed':True,'e2_run_schemas_and_prediction_metrics_verified':len(runs),
            'e1_prediction_archives_verified':len(e1['checkpoints']),'independent_checkpoint_replays':selected,
            'protected_preexisting_files_verified':len(before['files']),'changed_preexisting_files':changed,
            'runtime':{'wall_seconds':time.perf_counter()-wall,'process_cpu_seconds':time.process_time()-cpu}}
    write_new(base/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

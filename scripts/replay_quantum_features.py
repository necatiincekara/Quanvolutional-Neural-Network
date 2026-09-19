#!/usr/bin/env python3
"""E1 immutable replay: actual circuit → analytic map → cache → checkpoint."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from src.classical_quantum_controls import fixed_bank,fixed_expectations,fixed_features,image_patches
from src.research_protocol import file_hash,write_new,metrics,run_record,validate_run,digest
from src.ablation_models import NonTrainableQuantumClassicalNet


def raw_historical(manifest,membership,split):
    index={r['record_id']:r for r in manifest['records']}
    rows=[index[r] for r in membership['observed_valid_loader_order'][split]]
    arrays=[]
    for r in rows:
        p=ROOT/'set'/r['relative_path']
        assert file_hash(p)==r['byte_sha256']
        arrays.append(cv2.resize(cv2.imread(str(p),cv2.IMREAD_GRAYSCALE),(32,32)))
    return torch.tensor(np.array(arrays)/255.,dtype=torch.float32)[:,None],torch.tensor([r['label'] for r in rows]),rows


def error(a,b):
    a=torch.as_tensor(a).double();b=torch.as_tensor(b).double()
    return {'max_abs':float((a-b).abs().max()),'rms':float(((a-b)**2).mean().sqrt())}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output-name',default='e1')
    args=parser.parse_args();args.root=args.root.resolve();out=args.root/args.output_name;out.mkdir(exist_ok=False)
    start=time.perf_counter();cpu=time.process_time();torch.set_num_threads(1)
    manifest=json.loads((args.root/'dataset_manifest.json').read_text())
    protocol=json.loads((args.root/'protocol.json').read_text())
    membership=json.loads((args.root/'historical_membership.json').read_text())
    assert json.loads((args.root/'e0_checks.json').read_text())['passed']
    report=run_record('astra_e1_replay_20260915','replay',manifest,protocol,
                      ['src/classical_quantum_controls.py','scripts/replay_quantum_features.py','train_ablation_local.py','src/ablation_models.py'],
                      manifest_path=args.root/'dataset_manifest.json')
    report['architecture']={'name':'historical_fixed_4q_map','version':'exact_analytic_v1'}
    report['data']['evaluation_split']='test'
    report['data']['train_ids']=membership['observed_valid_loader_order']['train']
    report['data']['validation_ids']=[]
    report['data']['test_ids']=membership['observed_valid_loader_order']['test']
    report['data']['protocol_checksum']=digest(membership)
    report['limitations'].append('Replay IDs are manifest record IDs preserving duplicate historical occurrences, not the deduplicated future-pilot sample IDs.')
    report['quantum']={'backend':'default.qubit','differentiation':'backprop reference; analytic autograd'}
    report['hyperparameters']={'banks':list(range(42,48)),'feature_atol':2e-7,'logit_atol':3e-5,'logit_rtol':1e-5,'metric_accuracy_tolerance_points':.011,'derivative_atol':1e-10}
    report['limitations']+=['Historical row identity matched to current local order by full tensor equality where cache exists; absent historical sample IDs cannot establish remote training membership.','Only fixed measured map; not V7.','Derivative reference uses CPU float64 backprop.']
    train,train_y,train_rows=raw_historical(manifest,membership,'train')
    test,test_y,test_rows=raw_historical(manifest,membership,'test')
    spec=importlib.util.spec_from_file_location('audit_equivalence',ROOT/'research/verify_fixed_quantum_equivalence.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    circuit=audit.extract_reference('train_ablation_local.py','fixed_circuit')
    rows=[];checkpoint_rows=[];jacobians=[]
    try:
        for seed in range(42,48):
            bank=fixed_bank(seed)
            analytic_test=fixed_features(test,bank)
            # Every patch of 16 distributed train images and 16 test images, each bank.
            picked_train=torch.linspace(0,len(train)-1,16).long();picked_test=torch.linspace(0,len(test)-1,16).long()
            selected=torch.cat((train[picked_train],test[picked_test]))
            patches=image_patches(selected).reshape(-1,4).double()
            qfeatures=[]
            for w in bank:
                chunks=[torch.stack(circuit(patches[i:i+1024],w),-1) for i in range(0,len(patches),1024)]
                qfeatures.append(torch.cat(chunks).reshape(-1,16,16,4).permute(0,3,1,2))
                x=patches[(seed*71)%len(patches)]
                qj=torch.autograd.functional.jacobian(lambda a,b:torch.stack(circuit(a,b)),(x,w))
                cj=torch.autograd.functional.jacobian(fixed_expectations,(x,w))
                j={'seed':seed,'input':error(qj[0],cj[0]),'weights':error(qj[1],cj[1])}
                assert max(j['input']['max_abs'],j['weights']['max_abs'])<1e-10
                jacobians.append(j)
            qfeatures=torch.cat(qfeatures,1)
            selected_analytic=fixed_features(selected,bank,output_dtype=torch.float64)
            qe=error(qfeatures,selected_analytic);assert qe['max_abs']<1e-10
            f32e=error(fixed_features(selected,bank.float()),selected_analytic)
            assert f32e['max_abs']<5e-7
            row={'seed':seed,'weights':bank.tolist(),'quantum_analytic_error_float64':qe,
                 'float32_vs_float64':f32e,'quantum_reference_images':len(selected),
                 'quantum_reference_patch_filter_instances':len(patches)*4,'caches':[]}
            cache_test=None
            for meta_path in sorted((ROOT/'experiments/quantum_cache').glob('*metadata.json')):
                meta=json.loads(meta_path.read_text());payload=meta['cache_payload']
                if payload.get('namespace')!='henderson_local_ablation' or payload.get('seed')!=seed:continue
                for split,raw,labels in [('train',train,train_y),('test',test,test_y)]:
                    fp=meta_path.parent/(meta['cache_key']+'_'+split+'_features.npy')
                    lp=meta_path.parent/(meta['cache_key']+'_'+split+'_labels.npy')
                    cached=np.load(fp); cached_y=np.load(lp)
                    generated=fixed_features(raw,bank)
                    ce=error(cached,generated)
                    assert np.array_equal(cached_y,labels.numpy()),'Label ordering mismatch'
                    assert ce['max_abs']<=2e-7,'Cache mismatch; investigate before E2'
                    row['caches'].append({'split':split,'path':str(fp.relative_to(ROOT)),
                                          'sha256':file_hash(fp),'metadata_sha256':file_hash(meta_path),
                                          'labels_equal':True,'error':ce,'shape':list(cached.shape)})
                    if split=='test':cache_test=torch.tensor(cached)
            # If no historical cache, regenerate all test quantum outputs, then compare logits.
            if cache_test is None:
                full_patches=image_patches(test).reshape(-1,4).double();groups=[]
                for w in bank:
                    chunks=[torch.stack(circuit(full_patches[i:i+4096],w),-1).float() for i in range(0,len(full_patches),4096)]
                    groups.append(torch.cat(chunks).reshape(-1,16,16,4).permute(0,3,1,2))
                cache_test=torch.cat(groups,1)
                row['regenerated_full_test_error']=error(cache_test,analytic_test)
                assert row['regenerated_full_test_error']['max_abs']<=2e-7
            paths=list((ROOT/'experiments/low_data').glob(f'ablation_non_trainable_quantum_*seed{seed}_split42_fraction42.json'))
            full=ROOT/f'experiments/ablation_non_trainable_quantum_seed{seed}_split42.json'
            if full.exists():paths.append(full)
            for rp in sorted(paths):
                historical=json.loads(rp.read_text())
                cp=ROOT/('models/low_data' if rp.parent.name=='low_data' else 'models')/('best_'+rp.stem+'.pth')
                model=NonTrainableQuantumClassicalNet(in_channels=16,num_classes=44)
                model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True));model.eval()
                with torch.no_grad():
                    a=torch.cat([model(analytic_test[i:i+64]) for i in range(0,len(test),64)])
                    b=torch.cat([model(cache_test[i:i+64]) for i in range(0,len(test),64)])
                assert torch.allclose(a,b,atol=3e-5,rtol=1e-5),'Logit mismatch'
                assert torch.equal(a.argmax(1),b.argmax(1)),'Prediction mismatch'
                m=metrics(test_y,a.argmax(1));delta=abs(m['accuracy']-historical['test_acc'])
                assert delta<=.011,'Historical accuracy mismatch'
                pred_path=out/(rp.stem+'_predictions.npz')
                np.savez_compressed(pred_path,record_ids=np.array([r['record_id'] for r in test_rows]),
                                    targets=test_y.numpy(),analytic_logits=a.numpy(),reference_logits=b.numpy())
                checkpoint_rows.append({'result':str(rp.relative_to(ROOT)),'result_sha256':file_hash(rp),
                                        'checkpoint':str(cp.relative_to(ROOT)),'checkpoint_sha256':file_hash(cp),
                                        'seed':seed,'logit_error':error(a,b),'predictions_equal':True,
                                        'reported_accuracy':historical['test_acc'],'accuracy_difference_points':delta,
                                        'metrics':m,'predictions':str(pred_path.relative_to(ROOT))})
                report['prediction_artifacts'].append(str(pred_path.relative_to(ROOT)))
            rows.append(row);print('bank',seed,'verified; checkpoints',len(checkpoint_rows),flush=True)
        report['status']='completed'
    except Exception as exc:
        report['status']='failed';report['limitations'].append(repr(exc))
        raise
    finally:
        report['runtime']={'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu}
        report['gradient_diagnostics']={'status':'verified' if len(jacobians)==24 else 'partial','jacobians':jacobians}
        report['checkpoint_criterion']='existing immutable historical checkpoints; inference only'
        validate_run(report);write_new(out/'run.json',report)
        write_new(out/'replay_results.json',{'status':report['status'],'banks':rows,'checkpoints':checkpoint_rows,
                                           'fixed_map_irreducible_quantum_claims_closed':report['status']=='completed',
                                           'scope':'current fixed RX/Rot/forward-CNOT-chain/Z expectation map only'})


if __name__=='__main__':main()

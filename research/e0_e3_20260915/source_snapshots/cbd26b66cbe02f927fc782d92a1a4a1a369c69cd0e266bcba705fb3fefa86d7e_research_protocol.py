"""Additive Astra protocol. Historical loaders/results are never mutated."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    """Exclusive creation: never silently replace evidence. Reject non-JSON NaNs."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(encoded)


def environment():
    versions = {'python': platform.python_version()}
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'pennylane', 'opencv-python'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'unavailable'
    return {'versions': versions, 'hardware': {'platform': platform.platform(),
            'machine': platform.machine(), 'cuda': torch.cuda.is_available(),
            'mps': torch.backends.mps.is_available(), 'torch_threads': torch.get_num_threads()}}


def git_state(source_paths=()):
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT)
    status = git('status', '--porcelain=v1', '-uall').decode()
    return {'commit': git('rev-parse', 'HEAD').decode().strip(), 'dirty': bool(status),
            'status': status, 'tracked_diff_sha256': digest(git('diff', '--binary', 'HEAD')),
            'source_hashes': {str(p): file_hash(ROOT / p) for p in source_paths}}


def build_manifest(dataset_root=None):
    from src.config import TAGS
    dataset_root = Path(dataset_root or ROOT / 'set')
    records = []
    for split in ('train', 'test'):
        for path in sorted((dataset_root / split).rglob('*')):
            if not path.is_file():
                continue
            relative = str(path.relative_to(dataset_root))
            byte_hash = file_hash(path)
            code = path.name[-6:-4]
            reasons = []
            if path.suffix != '.png': reasons.append('unsupported_extension')
            if code not in TAGS: reasons.append('unknown_label_code')
            arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            pixel_hash = None
            if arr is None:
                reasons.append('decode_failure')
            else:
                pixel_hash = digest(str(arr.shape).encode() + b'\0uint8\0' + arr.tobytes())
            records.append({'record_id': digest(relative.encode() + b'\0' + byte_hash.encode()),
                            'sample_id': 'pixel-sha256:' + pixel_hash if pixel_hash else None,
                            'relative_path': relative, 'original_split': split,
                            'original_source': 'repository set; writer/source identity unverified',
                            'byte_sha256': byte_hash, 'pixel_sha256': pixel_hash,
                            'decoded_shape': list(arr.shape) if arr is not None else None,
                            'label_code': code, 'label': int(code)-1 if code in TAGS else None,
                            'valid': not reasons, 'exclusion_reasons': reasons})
    duplicates = {}
    for key in ('byte_sha256', 'pixel_sha256'):
        groups = {}
        for r in records:
            if r[key]: groups.setdefault(r[key], []).append(r)
        duplicates[key] = [{'hash': k, 'record_ids': [r['record_id'] for r in rs],
                            'paths': [r['relative_path'] for r in rs],
                            'cross_split': len({r['original_split'] for r in rs}) > 1,
                            'label_conflict': len({r['label'] for r in rs if r['valid']}) > 1}
                           for k, rs in sorted(groups.items()) if len(rs) > 1]
    payload = {'schema_version': 1, 'protocol': 'astra_dataset_inventory_v1',
               'dataset_root_relative_to_repo': 'set', 'decode': 'OpenCV IMREAD_GRAYSCALE, native size',
               'label_rule': 'historical filename[-6:-4]; int(code)-1', 'label_map': TAGS,
               'records': records, 'duplicates': duplicates,
               'counts': {split: {'raw': sum(r['original_split']==split for r in records),
                                  'valid': sum(r['original_split']==split and r['valid'] for r in records)}
                          for split in ('train', 'test')},
               'rights_status': 'existing local research data; redistribution/license not verified'}
    payload['manifest_checksum'] = digest(payload)
    return payload


def make_protocol(manifest, split_seed=20260915):
    """Explicit NEW split: deduplicate content, quarantine label conflicts and test overlaps."""
    by_id = {}
    for r in manifest['records']:
        if r['valid']: by_id.setdefault(r['sample_id'], []).append(r)
    train, test, excluded = [], [], []
    representatives = {}
    for sid, rows in sorted(by_id.items()):
        if len({r['label'] for r in rows}) > 1:
            excluded.append({'sample_id': sid, 'reason': 'conflicting_labels'})
            continue
        tests = [r for r in rows if r['original_split'] == 'test']
        representative = min(tests or rows, key=lambda r: r['relative_path'])
        representatives[sid] = representative['record_id']
        if tests:
            test.append(sid)
            if len(tests) != len(rows):
                excluded.append({'sample_id': sid, 'reason': 'historical_test_overlap_excluded_from_new_train'})
        else:
            train.append(sid)
    label = {sid: rows[0]['label'] for sid, rows in by_id.items()}
    val = []
    for cls in sorted(set(label[s] for s in train)):
        members = sorted((s for s in train if label[s] == cls), key=lambda s: digest([split_seed, s]))
        n = min(len(members)-1, max(1, round(.2 * len(members)))) if len(members)>1 else 0
        val.extend(members[:n])
    train = sorted(set(train)-set(val))
    payload = {'protocol': 'astra_deduplicated_validation_pilot_v1',
               'historical_benchmark_changed': False, 'split_seed': split_seed,
               'manifest_checksum': manifest['manifest_checksum'],
               'train_ids': train, 'validation_ids': sorted(val), 'test_ids': sorted(test),
               'representatives': representatives, 'exclusions': excluded,
               'policy': '20% per-class deterministic content-hash validation; preserve at least one train sample; no writer identity claimed; test sealed for E2',
               'counts': {'train':len(train),'validation':len(val),'test':len(test)}}
    payload['protocol_checksum'] = digest(payload)
    validate_splits(payload)
    return payload


def validate_splits(protocol):
    sets = [set(protocol[k]) for k in ('train_ids','validation_ids','test_ids')]
    for k, ids in zip(('train_ids','validation_ids','test_ids'),sets):
        if len(ids) != len(protocol[k]): raise ValueError('Duplicate sample IDs: '+k)
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Cross-split content leakage')


def load_ids(manifest, protocol, ids):
    """Manifest-addressed data with verification; no implicit test loading."""
    records = {r['record_id']: r for r in manifest['records']}
    images, labels = [], []
    for sid in ids:
        row = records[protocol['representatives'][sid]]
        p = ROOT / 'set' / row['relative_path']
        if file_hash(p) != row['byte_sha256']: raise ValueError('Dataset changed: '+str(p))
        arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        images.append(cv2.resize(arr, (32,32)).astype(np.float32)/255)
        labels.append(row['label'])
    return torch.tensor(np.array(images))[:,None], torch.tensor(labels, dtype=torch.long)


def metrics(labels, predictions, classes=44):
    matrix = np.zeros((classes,classes),dtype=np.int64)
    np.add.at(matrix,(np.asarray(labels,dtype=int),np.asarray(predictions,dtype=int)),1)
    tp=np.diag(matrix); support=matrix.sum(1); guessed=matrix.sum(0)
    recall=np.divide(tp,support,out=np.zeros(classes),where=support>0)
    precision=np.divide(tp,guessed,out=np.zeros(classes),where=guessed>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(classes),where=precision+recall>0)
    return {'accuracy':100*float(tp.sum()/max(1,matrix.sum())),
            'macro_f1':100*float(f1.mean()),
            'balanced_accuracy':100*float(recall[support>0].mean()),
            'macro_f1_policy':'all 44 labels; absent classes F1=0',
            'balanced_accuracy_policy':'mean recall over supported classes',
            'class_support':support.tolist(),'per_class_f1':(100*f1).tolist(),
            'per_class_recall':(100*recall).tolist(),'confusion_matrix':matrix.tolist()}


def validate_schema(value, schema, path='$'):
    """Validate the deliberately small JSON-Schema subset used by our run schema.

    Unsupported validation keywords fail closed; external Draft2020 validators can
    consume the same schema. This keeps the local protocol dependency-free.
    """
    supported={'$schema','title','description','type','required','properties','additionalProperties',
               'items','enum','minimum','minLength','minItems'}
    if set(schema)-supported: raise ValueError('Unsupported schema keywords')
    types=schema.get('type'); types=types if isinstance(types,list) else [types]
    checks={'object':lambda: isinstance(value,dict),'array':lambda:isinstance(value,list),
            'string':lambda:isinstance(value,str),'integer':lambda:isinstance(value,int) and not isinstance(value,bool),
            'number':lambda:isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value),
            'boolean':lambda:isinstance(value,bool),'null':lambda:value is None,None:lambda:True}
    if not any(checks[t]() for t in types): raise ValueError(path+': wrong type')
    if 'enum' in schema and value not in schema['enum']: raise ValueError(path+': enum')
    if isinstance(value,dict):
        for k in schema.get('required',[]):
            if k not in value: raise ValueError(path+': missing '+k)
        props=schema.get('properties',{})
        if schema.get('additionalProperties') is False and set(value)-set(props): raise ValueError(path+': extra fields')
        for k,v in value.items():
            if k in props: validate_schema(v,props[k],path+'.'+k)
    if isinstance(value,list):
        if len(value)<schema.get('minItems',0): raise ValueError(path+': minItems')
        for i,v in enumerate(value): validate_schema(v,schema.get('items',{}),f'{path}[{i}]')
    if isinstance(value,str) and len(value)<schema.get('minLength',0): raise ValueError(path+': minLength')
    if value is not None and 'minimum' in schema and value<schema['minimum']: raise ValueError(path+': minimum')


def validate_run(run):
    schema=json.loads((ROOT/'research/schemas/run.schema.json').read_text())
    validate_schema(run,schema)
    validate_splits(run['data'])
    if run['status']=='completed' and run['kind']=='training':
        for key in ('train_seed','model_init_seed','split_seed','subset_seed'):
            if run['seeds'][key] is None: raise ValueError('Training seed missing: '+key)
        if not run['epoch_metrics'] or run['final_metrics'] is None or not run['prediction_artifacts']:
            raise ValueError('Completed training evidence incomplete')
    if run['kind']=='training' and run['data']['evaluation_split']=='test':
        raise ValueError('Pilot training may not evaluate final test')


def run_record(experiment_id, kind, manifest, protocol, source_paths):
    env=environment()
    return {'schema_version':1,'experiment_id':experiment_id,'kind':kind,'status':'running',
            'architecture':{'name':'pending','version':'astra_v1'},'git':git_state(source_paths),
            'data':{'manifest':'research/e0_e3_20260915/dataset_manifest.json',
                    'manifest_checksum':manifest['manifest_checksum'],
                    'protocol_checksum':protocol['protocol_checksum'],
                    **{k:protocol[k] for k in ('train_ids','validation_ids','test_ids')},
                    'evaluation_split':'validation'},
            'seeds':{k:None for k in ('train_seed','split_seed','subset_seed','model_init_seed','filter_bank_seed')},
            'hyperparameters':{},'parameter_counts':{'trainable':0,'frozen':0},
            'optimizer':{},'scheduler':{'name':'none'},'augmentation':{'name':'none','location':'none'},
            'precision':{'dtype':'float32','amp':False,'scaler':False},
            'quantum':{'backend':'not_applicable','differentiation':'not_applicable'},
            'framework_versions':env['versions'],'hardware':env['hardware'],
            'runtime':{'wall_seconds':0.,'process_cpu_seconds':0.},
            'checkpoint_criterion':'not_applicable','epoch_metrics':[],
            'gradient_diagnostics':{'status':'not_applicable'},'prediction_artifacts':[],
            'final_metrics':None,'limitations':[]}

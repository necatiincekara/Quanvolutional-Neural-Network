#!/usr/bin/env python3
"""Freeze a NEW Astra manifest without altering the historical benchmark."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research_protocol import build_manifest,make_protocol,write_new,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args(); start=time.perf_counter(); cpu=time.process_time()
    manifest=build_manifest(); protocol=make_protocol(manifest)
    # Snapshot today's observed directory iteration separately. It is not claimed
    # to recover the unknown ordering on a historical remote filesystem.
    index={r['relative_path']:r for r in manifest['records']}
    observed={split:[index[split+'/'+name]['record_id'] for name in os.listdir(ROOT/'set'/split)
                     if split+'/'+name in index and index[split+'/'+name]['valid']]
              for split in ('train','test')}
    legacy={'status':'current_local_order_snapshot; not a reconstruction of remote historical IDs',
            'original_folder_membership':{s:[r['record_id'] for r in manifest['records'] if r['original_split']==s] for s in ('train','test')},
            'observed_valid_loader_order':observed,
            'historical_internal_train_validation_ids':'unknown; do not infer from seed alone'}
    write_new(args.output_dir/'dataset_manifest.json',manifest)
    write_new(args.output_dir/'protocol.json',protocol)
    write_new(args.output_dir/'historical_membership.json',legacy)
    checks={'passed':True,'manifest_rebuild_identical':manifest==build_manifest(),
            'counts':manifest['counts'],'new_protocol_counts':protocol['counts'],
            'invalid_records':[r for r in manifest['records'] if not r['valid']],
            'duplicate_groups':{k:len(v) for k,v in manifest['duplicates'].items()},
            'cross_split_duplicate_groups':{k:sum(r['cross_split'] for r in v) for k,v in manifest['duplicates'].items()},
            'conflicting_label_groups':sum(r['label_conflict'] for r in manifest['duplicates']['pixel_sha256']),
            'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu}
    assert checks['manifest_rebuild_identical']
    write_new(args.output_dir/'e0_checks.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()

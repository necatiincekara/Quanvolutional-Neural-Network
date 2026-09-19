"""Small scientific-integrity checks, not an OCR training suite."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from src.research_protocol import build_manifest,make_protocol,metrics,run_record,validate_run,write_new
from src.classical_quantum_controls import image_patches
from src.research_feature_maps import PatchMap,FAMILIES,pooled


class Foundation(unittest.TestCase):
    def test_duplicate_quarantine_and_determinism(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'train').mkdir();(p/'test').mkdir()
            for name in ['train/a01.png','test/b01.png','train/c02.png']:
                cv2.imwrite(str(p/name),np.ones((4,4),np.uint8))
            cv2.imwrite(str(p/'train/invalid00.png'),np.zeros((4,4),np.uint8))
            a=build_manifest(p);self.assertEqual(a,build_manifest(p))
            self.assertEqual(len(a['duplicates']['pixel_sha256']),1)
            self.assertTrue(a['duplicates']['pixel_sha256'][0]['cross_split'])
            self.assertTrue(a['duplicates']['pixel_sha256'][0]['label_conflict'])
            q=make_protocol(a);self.assertFalse(q['train_ids']);self.assertFalse(q['test_ids'])

    def test_patch_and_feature_contract(self):
        x=torch.arange(2*32*32,dtype=torch.float32).reshape(2,1,32,32)/2048
        expected=torch.stack([x[0,0,r:r+2,c:c+2].flatten() for r in range(0,32,2) for c in range(0,32,2)])
        self.assertTrue(torch.equal(image_patches(x)[0],expected))
        for family in FAMILIES:
            a=PatchMap(family,73);b=PatchMap(family,73)
            self.assertEqual(a(x).shape,(2,16,16,16));self.assertEqual(pooled(a(x)).shape,(2,256))
            self.assertTrue(torch.equal(a(x),b(x)));self.assertTrue(torch.isfinite(a(x)).all())

    def test_schema_rejects_incomplete_training_and_test_access(self):
        d=ROOT/'research/e0_e3_20260915'
        m=json.loads((d/'dataset_manifest.json').read_text());p=json.loads((d/'protocol.json').read_text())
        r=run_record('test','training',m,p,[]);validate_run(r)
        a=copy.deepcopy(r);del a['seeds']['subset_seed']
        with self.assertRaises(ValueError):validate_run(a)
        a=copy.deepcopy(r);a['status']='completed'
        with self.assertRaises(ValueError):validate_run(a)
        a=copy.deepcopy(r);a['data']['evaluation_split']='test'
        with self.assertRaises(ValueError):validate_run(a)
        a=copy.deepcopy(r);a['data']['validation_ids'].append(a['data']['train_ids'][0])
        with self.assertRaises(ValueError):validate_run(a)

    def test_metrics_and_exclusive_evidence(self):
        m=metrics([0,0,1],[0,1,1],classes=2)
        self.assertAlmostEqual(m['accuracy'],200/3);self.assertAlmostEqual(m['macro_f1'],200/3)
        self.assertAlmostEqual(m['balanced_accuracy'],75)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';write_new(p,{'a':1})
            with self.assertRaises(FileExistsError):write_new(p,{'a':2})
            self.assertEqual(json.loads(p.read_text()),{'a':1})


if __name__=='__main__':unittest.main()

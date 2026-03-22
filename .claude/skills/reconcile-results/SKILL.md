---
name: reconcile-results
description: JSON loglar, checkpointler, notebooklar ve dokumanlar arasindaki metrik celiskilerini cozerek calismanin bugunku gercegini bul.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
  - Task
---

# Reconcile Results

Bu skill, README guncellemesi, paper yazimi veya durum raporu oncesi kullanilmalidir.

## Adimlar

1. `CLAUDE.md` ve varsa `AGENTS.md` oku.
2. Kanit topla:
   - `experiments/*.json`
   - `models/`
   - `docs/EXPERIMENTS.md`
   - `paper/draft.md`
   - `README.md`
   - `train_v7_colab.ipynb`
3. Celiski listesi cikar:
   - current-best iddiasi
   - val/test farklari
   - yapildi denilen ama hala pending duran deneyler
   - reproducibility bosluklari
4. Makine tarafindan uretilmis artefaktlari anlati dokumanlarina tercih et.
5. Cikti olarak ver:
   - bugun desteklenen durum
   - stale claim listesi
   - sync edilmesi gereken dosyalar

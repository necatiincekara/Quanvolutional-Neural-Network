---
name: paper-sync
description: Makale, tez ve publication-facing metinleri mevcut repo kanitlariyla senkronize et.
disable-model-invocation: true
allowed-tools:
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - Task
---

# Paper Sync

Bu skill `paper/draft.md`, tez ozeti veya yayin ozetleri duzenlenmeden once kullanilir.

## Adimlar

1. `CLAUDE.md`, `docs/PUBLICATION_STRATEGY_2026-03-22.md` ve `paper/draft.md` oku.
2. `experiments/*.json`, `docs/EXPERIMENTS.md` ve notebook ciktilariyla metrikleri eslestir.
3. Su maddeleri kontrol et:
   - current-best model claim
   - trainable vs non-trainable quantum claim
   - pending vs completed experiment anlatisi
   - publication route ile uyum
4. Sonra:
   - desteklenmeyen cumleleri isaretle
   - duzeltilmis claim hiyerarsisi oner
   - gerekiyorsa ilgili dosyalari guncelle

## Notlar

- Bu repo artik "quantum wins" anlatisiyla otomatik yazilmamali.
- Daha savunulabilir ana cizgi: benchmark disiplini, failure analysis, hybrid QML engineering.

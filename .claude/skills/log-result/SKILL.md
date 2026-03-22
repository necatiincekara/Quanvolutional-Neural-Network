---
name: log-result
description: Deney sonucunu guncel artefaktlarla uyumlu sekilde kaydet. Stale claim uretmeden docs ve result loglarini gunceller.
disable-model-invocation: true
allowed-tools:
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - AskUserQuestion
---

# Experiment Result Logger

## Adimlar

1. Sonucu dogrula:
   - `experiments/*.json`
   - ilgili checkpoint
   - notebook veya terminal ciktilari
2. Sonucun nereye ait oldugunu belirle:
   - `docs/EXPERIMENTS.md`
   - `paper/draft.md`
   - `README.md`
   - `CLAUDE.md`
3. Su ayrimi koru:
   - current result
   - historical result
   - pending experiment
4. Guncelleme oncesi stale claim dogurup dogurmayacagini kontrol et.

## Notlar

- Sonuc loglarken `experiments/*.json` ve dogrulanmis artefaktlar, anlati metninden once gelir.
- Publication-facing metin degisecekse `paper-sync` veya `reconcile-results` ile birlikte kullan.

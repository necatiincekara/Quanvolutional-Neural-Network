---
name: compare
description: Model, circuit veya egitim konfigurasyonlarini guncel sonuclara gore karsilastir. Tarihsel anlatidan degil, artefaktlardan ilerler.
disable-model-invocation: true
allowed-tools:
  - Read
  - Grep
  - Glob
  - AskUserQuestion
---

# Model Comparison

## Adimlar

1. Once guncel kaynaklari oku:
   - `experiments/*.json`
   - `docs/EXPERIMENTS.md`
   - `paper/draft.md`
   - `CLAUDE.md` veya `AGENTS.md`
2. Karsilastirma tipini belirle:
   - `current`: bugunku en guclu modeller
   - `quantum`: trainable vs non-trainable quantum
   - `thesis`: thesis HQNN/CNN sonuclari vs repo sonuclari
   - `architecture`: mimari trade-off
3. Su tabloyu olustur:

```text
| Model | Test Acc | Val Acc | Params | Compute Cost | Not |
```

4. Yorumda su ayrimi acik yap:
   - current factual result
   - historical claim
   - unsupported / stale claim

## Notlar

- `CLAUDE.md` benchmark tablosunu tek kaynak kabul etme.
- Metrik celiskisinde `experiments/*.json` daha ustundur.

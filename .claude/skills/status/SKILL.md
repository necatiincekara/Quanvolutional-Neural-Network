---
name: status
description: Projenin guncel durumunu ozetle. Kod, deney, dokumantasyon ve yayin hazirligini bugunku gercege gore raporla.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Project Status Overview

Bu skill, proje durumunu tarihsel roadmap yerine guncel artefaktlara gore ozetler.

## Adimlar

1. `CLAUDE.md` ve varsa `AGENTS.md` oku.
2. `experiments/*.json`, `models/`, `docs/EXPERIMENTS.md`, `paper/draft.md` ve `git status` uzerinden guncel durumu topla.
3. Su sorulara net cevap ver:
   - Su anda en iyi test sonucu hangi modelde?
   - Hangi belgeler stale?
   - Bir sonraki teknik adim ne?
   - Bir sonraki yazi / paper adimi ne?
4. Tarihsel roadmap hedeflerini mevcut gercek gibi sunma.

## Rapor Formati

```text
=== PROJE DURUMU (YYYY-MM-DD) ===

Aktif odak:
En guclu mevcut sonuc:
En kritik stale belge:
Bir sonraki teknik adim:
Bir sonraki paper adimi:

GUNCEL SONUCLAR:
- classical_conv:
- param_linear:
- non_trainable_quantum:
- V7 trainable:

RISKLER:
1.
2.
3.
```

## Notlar

- Makine tarafindan uretilmis artefaktlari, anlati dokumanlarindan daha guvenilir kabul et.
- Yayin stratejisi gerekiyorsa `docs/PUBLICATION_STRATEGY_2026-03-22.md` kullan.

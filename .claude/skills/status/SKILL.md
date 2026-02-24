---
name: status
description: Proje durumu ozeti - kod, model, deney, roadmap ve ortam durumunu tek bakista goster. Sonraki adimlari belirle.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Project Status Overview

Projenin genel durumunu ozetle ve sonraki adimlari belirle.

## Adimlar

1. **Kod durumu**: `git status`, `git log --oneline -10`, degisiklik ozeti
2. **Model durumu**: `models/` checkpoint'lari, en iyi performans, aktif versiyon
3. **Deney durumu**: `docs/EXPERIMENTS.md`'den son sonuclar, accuracy trendi
4. **Roadmap ilerleme**: `docs/AUDIT_REPORT.md`, `docs/IMPLEMENTATION_GUIDE.md`'den hedefler
5. **Ortam durumu**: Python/venv, dependencies, platform, GPU

## Rapor Formati

```
=== PROJE DURUMU (YYYY-MM-DD) ===

Aktif Versiyon: V[X]
En Iyi Accuracy: X% (V[X])
Son Deney: [aciklama]
Platform: [Mac/Colab]

ILERLEME:
[x] V1-V4 base implementasyon
[x] V6 agresif reduction deneyi
[ ] V7 gradient stabilization (hedef: 25%)
[ ] V8 multi-scale processing (hedef: 40%)
[ ] V9 selective quantum (hedef: 60%)
[ ] V10 fully trainable (hedef: 90%)

SIRADAKI ADIMLAR:
1. [en oncelikli]
2. [ikinci oncelik]
3. [ucuncu oncelik]

ONERILEN SKILL'LER:
- /train v7, /gradient-check, /experiment
```

## Arguman

`$ARGUMENTS`: "brief" (kisa), "detailed" (detayli), "roadmap" (sadece roadmap)

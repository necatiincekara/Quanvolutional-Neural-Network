---
name: log-result
description: Deney sonuclarini docs/EXPERIMENTS.md dosyasina kaydet. Benchmark tablolarini guncelle.
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

Deney sonuclarini sistematik olarak dokumante et.

## Adimlar

1. **Mevcut loglari oku**: `docs/EXPERIMENTS.md` - son versiyon numarasini bul
2. **Bilgileri topla** (kullanicidan veya `$ARGUMENTS`'tan):
   - Model versiyonu, konfigürasyon detaylari
   - Accuracy, loss, epoch time sonuclari
   - Gradient analizi bulgulari
   - Gozlemler ve yorumlar
3. **Formatla ve ekle**: `docs/EXPERIMENTS.md`'ye yeni kayit
4. **Benchmark guncelle**: Onemli milestone ise `README.md` ve `CLAUDE.md` tablolarini guncelle

## Log Formati

```markdown
### V[X] - [Deney Adi]
**Tarih:** YYYY-MM-DD | **Platform:** [M4 Mac / Colab A100]

| Parametre | Deger | Metrik | Deger |
|-----------|-------|--------|-------|
| Model | [class] | Train Acc | X% |
| Feature Map | [NxN] | Val Acc | X% |
| Circuit | [type] | Epoch Time | Xs |
| LR | [lr] | Q Calls/Batch | X |
| Epochs | [n] | Q Grad Norm | X.Xe-X |

**Gozlemler:** [bulgular]
**Onceki ile Karsilastirma:** V[X-1] X% -> V[X] X%
**Sonraki Adimlar:** [eylemler]
```

## Arguman

`$ARGUMENTS`: Dogrudan sonuc bilgileri verilebilir, ornegin "V7 gradient-stabilized 25% accuracy 45s/epoch"

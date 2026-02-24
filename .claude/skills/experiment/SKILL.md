---
name: experiment
description: Deney calistir ve sonuclari takip et. Ablation study, circuit karsilastirma, gradient analizi gibi deneyleri yonetir.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - AskUserQuestion
---

# Experiment Runner

Deney calistir ve sonuclari sistematik olarak takip et.

## Adimlar

1. **Mevcut deneyleri tara**: `experiments/run_experiments.py` ve `docs/EXPERIMENTS.md` oku
2. **Deney sec** (veya `$ARGUMENTS`'tan al):
   - **baseline**: Fixed vs Trainable quantum layer
   - **circuit**: strongly_entangling vs data_reuploading vs hardware_efficient
   - **ablation**: Komponent onem analizi (residual, attention, mixup)
   - **gradient**: Quantum/classical gradient flow
   - **expressivity**: Circuit ifade gucu olcumu
   - **custom**: Ozel deney
3. **Yapilandir ve calistir**: Secilen deneyi baslat
4. **Sonuclari analiz et**: Accuracy, loss, training egrileri, istatistik
5. **Logla**: `docs/EXPERIMENTS.md`'ye kaydet

## Log Formati

```markdown
### V[X] - [Deney Adi] (YYYY-MM-DD)

**Konfigürasyon:**
- Model: [detay] | Circuit: [tip] | Feature Map: [NxN]
- LR: [lr] | Epochs: [n] | Batch: [bs]

**Sonuclar:**
- Train/Val Accuracy: X% / X%
- Epoch Time: Xs | Quantum Calls/Batch: X

**Gozlemler:** [bulgular]
**Sonraki Adimlar:** [oneriler]
```

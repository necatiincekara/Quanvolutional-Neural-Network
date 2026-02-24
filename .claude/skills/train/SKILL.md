---
name: train
description: Training yonetimi - modeli sifirdan egit veya checkpoint'tan devam et. Quantum device, epoch sayisi, model versiyonu secimi yapar.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Edit
  - Grep
  - Glob
  - AskUserQuestion
---

# Training Management

Kullanici training baslatmak veya devam ettirmek istiyor.

## Adimlar

1. **Konfigurasyonu oku**: `src/config.py` dosyasini oku, mevcut ayarlari anla
2. **Checkpoint kontrolu**: `models/` klasorunde `best_quanv_net.pth` ve `checkpoint_latest.pth` var mi kontrol et
3. **Kullaniciya sor**:
   - Sifirdan mi, checkpoint'tan mi devam?
   - Model versiyonu: V4 base (`src/model.py`), V7+ trainable (`src/trainable_quantum_model.py`), enhanced (`src/enhanced_training.py`)
   - Quantum device: `default.qubit` (CPU/Mac) veya `lightning.gpu` (Colab/GPU)
   - Epoch sayisi
4. **Config guncelle**: Gerekirse `src/config.py` dosyasini duzenle
5. **Training baslat**:
   - Base: `python -m src.train` veya `python -m src.train --resume`
   - Enhanced: `python -m src.enhanced_training`
6. **Izle ve raporla**: Loss, accuracy, quantum output std, gradient durumu

## Platform Notlari

- M4 Mac'te `lightning.gpu` KULLANILAMAZ - sadece `default.qubit`
- Mixed precision (AMP) sadece CUDA GPU'da calisir
- Ilk epoch yavas (quantum kernel derleme), sonrakiler hizlanir
- Gradient vanishing gorulurse `/gradient-check` oner

## Arguman

`$ARGUMENTS` ile parametre verilebilir: "resume", "v7", "5 epoch", "cpu" vb.

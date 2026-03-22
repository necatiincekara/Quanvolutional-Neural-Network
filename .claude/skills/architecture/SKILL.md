---
name: architecture
description: Model mimarisi tasarla veya analiz et. Mevcut OCR-QML calismasinda trainable, non-trainable ve classical baseline yollarini guncel duruma gore ele al.
disable-model-invocation: true
allowed-tools:
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - AskUserQuestion
  - Task
---

# Model Architecture Design

Bu skill tarihsel V7-V10 hedeflerini oldugu gibi tekrar etmek icin degil, mevcut repo gercegine gore mimari karar vermek icin kullanilir.

## Adimlar

1. Once su dosyalari incele:
   - `src/model.py`
   - `src/trainable_quantum_model.py`
   - `src/ablation_models.py`
   - `src/thesis_models.py`
   - `src/benchmark_protocol.py`
   - `train_ablation_local.py`
   - `train_thesis_models.py`
   - `docs/PUBLICATION_STRATEGY_2026-03-22.md`
2. Islem turunu belirle:
   - `analyze`: mevcut mimariyi acikla, bottleneck ve parametre dagilimini ver
   - `hqnn2`: tezdeki HQNN-II faithful reproduction planla
   - `baseline`: guclu classical baseline ekle
   - `trainable`: trainable quantum yolunda iyilestirme planla
3. Tasarimda su sorulara cevap ver:
   - Bu degisiklik adil karsilastirma mi sagliyor?
   - Parametre ve veri butcesi makul mu?
   - Yayin stratejisine hizmet ediyor mu?
4. Cikti olarak sun:
   - kisa mimari diagrami
   - beklenen avantaj / risk
   - parametre etkisi
   - egitim maliyeti tahmini

## Notlar

- "Quantum advantage" varsayma.
- Mimari oneriyi current-best classical baseline'lara gore degerlendir.
- Tezle devam edilecekse HQNN-II ile bugunku Henderson-style non-trainable modeli karistirma.

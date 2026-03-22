---
name: train
description: Training yonetimi. Base, V7 trainable ve yerel ablation akislari arasinda dogru yolu sec ve calistir.
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

## Adimlar

1. Aktif hedefi belirle:
   - `base`: `python -m src.train`
   - `v7`: `python train_v7.py`
   - `ablation`: `python train_ablation_local.py ...`
2. Once ilgili kod yolunu oku:
   - `src/config.py`
   - `src/model.py` veya `src/trainable_quantum_model.py`
   - gerekiyorsa `src/enhanced_training.py` ve `src/ablation_models.py`
3. Checkpoint ve mevcut sonuc dosyalarini kontrol et.
4. Platformu dogru sec:
   - M4 Mac: `default.qubit`, yerel ablation veya smoke test
   - Colab GPU: V7 trainable veya pahali quantum egitimleri
5. Egitimden sonra:
   - sonucu `experiments/*.json` veya ilgili loga yaz
   - paper claim etkisini kisaca belirt

## Hedefe Gore Notlar

- Henderson-style non-trainable quantum yerelde cache + training olarak kosabilir.
- V7 trainable quantum compute-agir bir yol; lokal M4 icin birincil hedef degildir.
- Faithful HQNN-II reproduction, mevcut Henderson-style modelle ayni sey degildir.

## Ornekler

```bash
python train_ablation_local.py --model classical_conv --epochs 50
python train_ablation_local.py --model non_trainable_quantum --epochs 50
python train_v7.py
```

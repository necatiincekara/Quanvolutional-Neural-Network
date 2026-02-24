---
name: sync-colab
description: Mac ve Google Colab arasinda kod ve sonuc senkronizasyonu. GitHub push/pull, Colab ortam hazırligi, checkpoint transferi.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - AskUserQuestion
---

# Mac/Colab Sync Workflow

Mac ve Google Colab arasinda kod ve sonuc senkronizasyonu yap.

## Adimlar

1. **Git durumunu kontrol et**: `git status`, `git log --oneline -5`
2. **Sync yonunu belirle** (`$ARGUMENTS` veya kullaniciya sor):

### push - Mac -> Colab
1. Commit edilmemis degisiklikleri kontrol et
2. Commit mesaji oner, `git add/commit/push`
3. Colab'da `!git pull` talimati ver

### pull - Colab -> Mac
1. Indirilecek dosyalari belirle (checkpoint'lar, loglar)
2. Google Drive transfer talimati ver
3. `/log-result` ile sonuclari kaydetmeyi oner

### setup - Colab Ortam Hazırlığı
Colab icin setup kodunu olustur:
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/[user]/Quanvolutional-Neural-Network.git
!pip install pennylane pennylane-lightning-gpu
import torch; print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
```

## Onemli

- Checkpoint'lari (`.pth`) git'e EKLEME - Google Drive kullan
- `.gitignore`'da `models/`, `*.pth`, `wandb/` olmali
- Colab session'lari gecici - sonuclari hemen kaydet

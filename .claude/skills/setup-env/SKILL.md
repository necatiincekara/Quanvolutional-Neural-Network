---
name: setup-env
description: Gelistirme ortamini platforma gore kur ve dogrula. M4 Mac ve Colab icin otomatik konfigürasyon.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Edit
  - Grep
  - AskUserQuestion
---

# Environment Setup

Gelistirme ortamini platform'a gore kur ve dogrula.

## Adimlar

1. **Platform tespit et**: macOS/Linux, GPU (CUDA/MPS), Python versiyonu
2. **Platforma gore kur**:

### M4 Mac (Gelistirme)
```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision pennylane "numpy<2.0"
pip install scikit-learn opencv-python pillow tensorboard wandb tqdm
```
- Config: `QUANTUM_DEVICE = 'default.qubit'`

### Google Colab (Training)
```python
!pip install pennylane pennylane-lightning-gpu
from google.colab import drive; drive.mount('/content/drive')
```
- Config: `QUANTUM_DEVICE = 'lightning.gpu'`

3. **Config ayarla**: `src/config.py`'de device ve dataset yollarini guncelle
4. **Dogrulama testleri**: Import kontrolu, quantum circuit testi, dataset erisimi, model testi
5. **Sorunlari raporla**

## Bilinen Sorunlar

- M4 Mac: `lightning.gpu` yok (CUDA yok), `default.qubit` kullan
- Python 3.13: Bazi paketler desteklemiyor, 3.12.x oner
- NumPy 2.0: PennyLane uyumsuz, `numpy<2.0` zorunlu

## Arguman

`$ARGUMENTS`: "mac", "colab", "check" (sadece dogrulama)

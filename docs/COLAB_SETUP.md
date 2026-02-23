# Google Colab Kurulum ve Optimizasyon Rehberi

## Hızlı Başlangıç

### 1. Colab Notebook Başlatma
```python
# GPU'yu etkinleştir: Runtime > Change runtime type > GPU > T4

# Google Drive'ı bağla
from google.colab import drive
drive.mount('/content/drive')

# Proje dizinine git
import os
os.chdir('/content/Quanvolutional-Neural-Network')
```

### 2. Bağımlılıkları Yükleme
```bash
# CUDA-enabled PennyLane için
!pip install pennylane pennylane-lightning-gpu torch torchvision tqdm scikit-learn PyYAML tensorboard

# Kurulumu doğrula
import pennylane as qml
print(f"PennyLane version: {qml.__version__}")
print(f"Available devices: {qml.device('lightning.gpu', wires=4)}")
```

### 3. Training Başlatma
```python
# Tek komutla başlat
!python -m src.train

# Checkpoint'ten devam et (session timeout sonrası)
!python -m src.train --resume
```

## Google Colab Tipleri ve Öneriler (2025 Kasım)

### Ücretsiz Tier
- **GPU**: T4 (16GB VRAM)
- **RAM**: ~12GB
- **Disk**: ~80GB
- **Limit**: ~12 saat session
- **Maliyet**: $0
- **Bu Proje İçin**: ✅ Yeterli (V6 architecture için)

### Colab Pro ($10/ay)
- **GPU**: T4/V100 (16-32GB VRAM)
- **RAM**: ~32GB
- **Session**: ~24 saat
- **Background execution**: Evet
- **Bu Proje İçin**: ✅ İdeal (batch size artırabilirsiniz)

### Colab Pro+ ($50/ay)
- **GPU**: A100 (40GB VRAM)
- **RAM**: ~52GB
- **Session**: ~24 saat
- **Priority access**: Evet
- **Bu Proje İçin**: ⚠️ Overkill (gerek yok)

## Performans Optimizasyonları

### 1. Runtime Bağlantısını Koruma
```python
# Colab'ın timeout yapmaması için
import time
from IPython.display import clear_output

def keep_alive():
    while True:
        time.sleep(3600)  # Her saat
        print("Still training...", end='\r')

# Arka planda çalıştır (opsiyonel)
import threading
threading.Thread(target=keep_alive, daemon=True).start()
```

### 2. Checkpoint Stratejisi
```python
# config.py'de:
CHECKPOINT_INTERVAL = 1  # Her epoch checkpoint kaydet

# Training sonrası Drive'a kaydet
!cp models/best_quanv_net.pth /content/drive/MyDrive/checkpoints/
```

### 3. TensorBoard İzleme
```python
# TensorBoard'u başlat
%load_ext tensorboard
%tensorboard --logdir runs

# Başka bir cell'de training'i çalıştır
!python -m src.train
```

### 4. Memory Management
```python
# Training öncesi GPU memory'yi temizle
import torch
torch.cuda.empty_cache()

# Batch size'ı optimize et
# Ücretsiz T4 için: 64-128
# Pro V100 için: 128-256
# Pro+ A100 için: 256-512
```

## Session Timeout Sonrası Devam Etme

```bash
# 1. Drive'ı tekrar bağla
from google.colab import drive
drive.mount('/content/drive')

# 2. Repo'yu kopyala (eğer yoksa)
!git clone https://github.com/[your-username]/Quanvolutional-Neural-Network.git
%cd Quanvolutional-Neural-Network

# 3. Latest checkpoint'ten devam et
!python -m src.train --resume
```

## Veri Yönetimi

### Dataset Konumu
Projeniz halihazırda Drive kullanıyor:
```python
# src/config.py
TRAIN_PATH = '/content/drive/MyDrive/set/train'
TEST_PATH = '/content/drive/MyDrive/set/test'
```

### Hızlandırma İçin Local Copy (Opsiyonel)
```bash
# Drive'dan local'e kopyala (daha hızlı I/O)
!cp -r /content/drive/MyDrive/set /content/
```

Sonra [config.py](src/config.py) güncelleyin:
```python
TRAIN_PATH = '/content/set/train'
TEST_PATH = '/content/set/test'
```

## Tahmini Eğitim Süreleri (Google Colab)

### V6 Architecture (Mevcut Best)
- **Ücretsiz T4**: ~2 saat/epoch, 10 epoch = ~20 saat
- **Pro V100**: ~1 saat/epoch, 10 epoch = ~10 saat
- **Pro+ A100**: ~40 dk/epoch, 10 epoch = ~7 saat

### Memory Kullanımı
- Model: ~50MB
- Batch (128): ~8GB GPU RAM
- Quantum cache: ~2GB
- **Toplam**: ~10-12GB (T4'te rahat çalışır)

## Troubleshooting

### "Runtime disconnected"
```python
# 1. Yeniden bağlan: Runtime > Reconnect
# 2. Checkpoint'ten devam et:
!python -m src.train --resume
```

### "Out of memory"
```python
# config.py'de batch size'ı düşür:
BATCH_SIZE = 64  # 128 yerine
```

### "Cannot find dataset"
```python
# Drive bağlantısını kontrol et:
!ls /content/drive/MyDrive/set/train
```

## En İyi Pratikler

1. ✅ **Her epoch sonrası checkpoint kaydedin**
2. ✅ **Critical checkpoint'leri Drive'a kopyalayın**
3. ✅ **TensorBoard ile ilerlemeyi izleyin**
4. ✅ **Session timeout'a karşı hazırlıklı olun**
5. ✅ **Uzun training'lerde Pro kullanın** ($10/ay)
6. ❌ **M4 Mac'te production training yapmayın**
7. ❌ **Local disk'e önemli dosyalar kaydetmeyin** (session bitince siler)

## Sonuç

Bu proje için **Google Colab ücretsiz tier bile yeterlidir**. M4 Mac Mini'yi sadece kod geliştirme ve küçük testler için kullanın.

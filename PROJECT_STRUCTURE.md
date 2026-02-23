# Proje Dizin Yapısı

## Güncellenmiş Yapı (16 Kasım 2025)

```
Quanvolutional-Neural-Network/
│
├── 📂 src/                          # Ana kaynak kodlar
│   ├── config.py                    # Yapılandırma ve hiperparametreler
│   ├── dataset.py                   # Osmanlı karakter veri yükleme
│   ├── model.py                     # Temel kuantum-klasik hibrit (V4/V6)
│   ├── train.py                     # Eğitim pipeline'ı (AMP, checkpointing)
│   ├── trainable_quantum_model.py   # Gelişmiş eğitilebilir kuantum devreleri
│   └── enhanced_training.py         # İleri düzey eğitim framework'ü
│
├── 📂 docs/                         # Dokümantasyon
│   ├── README.md                    # Dokümantasyon indeksi
│   ├── COMPUTING_RESOURCES_2025.md  # 🆕 Python 3.12, M4 Mac, Colab Pro + VS Code rehberi ⭐
│   ├── AUDIT_REPORT.md              # Kapsamlı kod incelemesi ve yol haritası
│   ├── EXPERIMENTS.md               # Deneysel sonuçlar log'u (V1-V6)
│   ├── IMPLEMENTATION_GUIDE.md      # Adım adım geliştirme rehberi
│   ├── QUANTUM_ML_RECOMMENDATIONS.md # QML en iyi pratikleri
│   ├── RESEARCH_ROADMAP.md          # Yayın stratejisi ve araştırma zaman çizelgesi
│   ├── TRAINING_PLATFORM_GUIDE.md   # Platform karşılaştırması (eski, bkz. COMPUTING_RESOURCES)
│   └── COLAB_SETUP.md               # Google Colab kurulum rehberi
│
├── 📂 experiments/                  # Deneysel scriptler
│   └── run_experiments.py           # Otomatik ablasyon çalışmaları
│
├── 📄 improved_model.py             # Alternatif mimari tasarımları
├── 📄 improved_training.py          # Eğitim optimizasyonları
├── 📄 improved_quantum_circuit.py   # Gelişmiş devre tasarımları
├── 📄 performance_optimizations.py  # Performans ölçümleme araçları
│
├── 📄 CLAUDE.md                     # AI asistan talimatları
├── 📄 README.md                     # Ana proje README ⭐
├── 📄 requirements.txt              # Python bağımlılıkları
│
├── 📓 colab_training_optimized.ipynb # Google Colab notebook
├── 🔧 setup_mac.sh                  # Mac kurulum scripti
├── 🔧 workflow_sync.sh              # Workflow senkronizasyon aracı
│
└── 📂 models/                       # Kaydedilmiş model checkpoint'leri (runtime)
    ├── best_quanv_net.pth
    └── checkpoint_latest.pth
```

## Değişiklik Özeti

### ✅ Eklenenler (16 Kasım 2025)
- **`docs/COMPUTING_RESOURCES_2025.md`** ⭐ - Python 3.12/3.13, M4 Mac analizi (CUDA yok!), Colab Pro + VS Code tam rehberi
- `docs/AUDIT_REPORT.md` - Kapsamlı audit raporu ve V7-V10 yol haritası
- `docs/README.md` - Dokümantasyon navigasyon rehberi
- Tüm dokümantasyon `docs/` klasöründe organize edildi

### ❌ Silinenler
- `prd.md` - Eski ve güncelliğini yitirmiş PRD
- `IMMEDIATE_ACTION_PLAN.md` - Platforma özgü, gereksiz

### 🔄 Taşınanlar
- `experiments.md` → `docs/EXPERIMENTS.md`
- `QUANTUM_ML_RECOMMENDATIONS.md` → `docs/`
- `IMPLEMENTATION_GUIDE.md` → `docs/`
- `RESEARCH_ROADMAP.md` → `docs/`
- `TRAINING_PLATFORM_GUIDE.md` → `docs/`
- `COLAB_SETUP.md` → `docs/`

### 📝 Güncellenenler (16 Kasım 2025)
- **`requirements.txt`** - Python 3.12/3.13, PyTorch 2.6+, PennyLane 0.43+, platform-spesifik notlar
- `README.md` - Python 3.12, Colab Pro + VS Code önerisi, M4 Mac uyarısı
- `CLAUDE.md` - Güncel performans metrikleri, 2025 geliştirme öncelikleri
- `docs/AUDIT_REPORT.md` - Platform ve Python sürümü önerileri eklendi
- Tüm doküman referansları güncel yolları gösteriyor

## Hızlı Başlangıç (2025 Kasım)

1. **🆕 Ortam Kurulumu (Python, Colab, VS Code)**: [docs/COMPUTING_RESOURCES_2025.md](docs/COMPUTING_RESOURCES_2025.md) ⭐ **İLK ÖNCE BU!**
2. **Proje durumunu anlamak için**: [README.md](README.md)
3. **Detaylı teknik analiz için**: [docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md)
4. **Deney sonuçları için**: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
5. **V7-V10 geliştirme için**: [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)

### ⚠️ Önemli Uyarılar

- **M4 Mac Mini**: Kod geliştirme için mükemmel AMA quantum training için UYGUN DEĞİL (CUDA yok)
- **Google Colab Pro**: Quantum training için ŞART (A100 GPU + CUDA 12.1)
- **Python**: 3.12.x öneriliyor (production stability)

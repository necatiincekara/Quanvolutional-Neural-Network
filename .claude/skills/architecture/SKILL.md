---
name: architecture
description: Model mimarisi tasarla veya analiz et. V7-V10 roadmap'ine gore yeni versiyon tasarimi, katman analizi, parametre hesaplama.
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

Yeni model versiyonu tasarla veya mevcut mimariyi analiz et.

## Adimlar

1. **Mevcut mimarileri oku**:
   - `src/model.py` (V4/V6 base)
   - `src/trainable_quantum_model.py` (V7+ trainable)
   - `improved_model.py` (attention, multi-scale)
   - `docs/IMPLEMENTATION_GUIDE.md` (V7-V10 roadmap)

2. **Islem sec** (`$ARGUMENTS` veya sor):

### analyze - Mimari Analiz
- Katman yapisi diagrami, parametre sayilari, bottleneck tespiti

### v7 - Gradient Stabilization (Hedef: 25%)
- Residual connections, gradient scaling, layer normalization

### v8 - Multi-Scale Processing (Hedef: 40%)
- Birden fazla olcekte feature extraction, attention mechanism

### v9 - Selective Quantum (Hedef: 60%)
- Gating ile quantum/classical secimi, bilgi iceren patch'lerde quantum

### v10 - Fully Trainable (Hedef: 90%)
- Tam egitimli quantum, data re-uploading, hardware-efficient ansatz

3. **Tasarla**: PyTorch model sinifi, forward pass, parametre sayisi, memory tahmini
4. **Diagram olustur**:
```
Input (1,32,32) -> [Conv2d+GN+GELU+Pool] -> (16,16,16)
  -> [Conv2d+GN+GELU+Pool] -> (32,8,8)
  -> [QuanvLayer 4q] + [Residual Skip] -> (1,8,8)
  -> [Conv+FC] -> (44,)
```
5. **Onaylanirsa implementasyonu baslat**

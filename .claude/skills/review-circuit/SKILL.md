---
name: review-circuit
description: Quantum circuit implementasyonunu uzman gozuyle incele. Barren plateau, expressivity, gradient flow, gate optimization analizi. quantum-ml-reviewer agent'ini kullanir.
disable-model-invocation: true
allowed-tools:
  - Read
  - Grep
  - Glob
  - Task
  - AskUserQuestion
---

# Quantum Circuit Review

Quantum circuit implementasyonunu derinlemesine incele. **quantum-ml-reviewer** agent'ini kullanir.

## Adimlar

1. **Hedefi belirle** (`$ARGUMENTS` veya kullaniciya sor):
   - `base` -> `src/model.py` (QuanvLayer, V4/V6)
   - `trainable` -> `src/trainable_quantum_model.py` (V7+)
   - `improved` -> `improved_quantum_circuit.py`
   - Veya belirli bir dosya yolu

2. **Circuit dosyalarini oku** ve yapıyı anla

3. **quantum-ml-reviewer agent'ini cagir** - su konulari incelet:
   - Ansatz secimi uygunlugu
   - Entanglement yapisi ve expressivity
   - Barren plateau riski
   - Gradient computation yontemi (adjoint vs parameter-shift)
   - Gate sayisi ve circuit depth optimizasyonu

4. **Raporu sun**:
   - Guclu yonler
   - Kritik sorunlar (duzeltilmesi gereken)
   - Optimizasyon onerileri (kodla)
   - Best practice uyumu

## Analiz Kontrol Listesi

- [ ] Qubit sayisi yeterli mi? (4 qubit - 44 sinif icin)
- [ ] Encoding stratejisi uygun mu? (Angle vs Amplitude)
- [ ] Entanglement: linear, circular, all-to-all?
- [ ] Rotation gate secimi: RY, RZ, Rot?
- [ ] Residual path var mi? (V6 collapse onlemi)
- [ ] Initialization stratejisi barren plateau'ya karsi uygun mu?
- [ ] Feature map boyutu information bottleneck yaratıyor mu?

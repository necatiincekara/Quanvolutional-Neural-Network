---
name: gradient-check
description: Quantum layer gradient diagnostigi. Vanishing/exploding gradient, barren plateau tespiti yapar. V6 gradient collapse sorununu debug etmek icin kritik.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - Glob
---

# Gradient Diagnostics

Quantum layer'lardaki gradient sorunlarini teshis et. V6'nin 0% accuracy sorununun ana sebebi gradient vanishing - bu skill bunu tespit eder.

## Adimlar

1. **Model dosyalarini oku**: `src/model.py`, `src/trainable_quantum_model.py`, `improved_quantum_circuit.py`
2. **Diagnostik scripti olustur ve calistir**:

```python
import torch
import sys
sys.path.insert(0, '.')

def analyze_gradients(model, sample_input):
    model.train()
    output = model(sample_input)
    loss = output.sum()
    loss.backward()

    print("=== GRADIENT FLOW ANALYSIS ===")
    quantum_grads, classical_grads = [], []

    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            mean = param.grad.abs().mean().item()
            status = "VANISHING" if norm < 1e-7 else ("EXPLODING" if norm > 10 else "OK")
            print(f"{status:10s} | {name:40s} | norm={norm:.2e} mean={mean:.2e}")

            bucket = quantum_grads if any(k in name.lower() for k in ['quantum', 'qnode', 'quanv']) else classical_grads
            bucket.append(norm)
        else:
            print(f"{'NO GRAD':10s} | {name:40s}")

    if quantum_grads:
        print(f"\nQuantum grad avg: {sum(quantum_grads)/len(quantum_grads):.2e}")
    if classical_grads:
        print(f"Classical grad avg: {sum(classical_grads)/len(classical_grads):.2e}")
```

3. **Analiz et**:
   - Vanishing: norm < 1e-7
   - Exploding: norm > 10
   - Barren plateau: Quantum parametrelerde uniform sifira yakin
   - Quantum vs Classical gradient orani

4. **Cozum oner**:
   - Gradient clipping, residual connections, ayri learning rate
   - Circuit depth azaltma, parameter initialization
   - `/architecture` ile mimari degisiklik

## Bilinen Sorunlar

| Versiyon | Feature Map | Gradient Durumu |
|----------|------------|-----------------|
| V4 (8x8) | 16 calls | Stabil, yavas ogrenme |
| V6 (6x6) | 9 calls | TAMAMEN COLLAPSE - 0% accuracy |

---
name: compare
description: Farkli model versiyonlarini, circuit tasarimlarini veya konfigurasyonlari karsilastir. Performans tablolari ve analiz uretir.
disable-model-invocation: true
allowed-tools:
  - Read
  - Grep
  - Glob
  - AskUserQuestion
---

# Model Version Comparison

Farkli model versiyonlarini ve konfigurasyonlarini karsilastir.

## Adimlar

1. **Veri topla**: `docs/EXPERIMENTS.md`, `CLAUDE.md` benchmark tablosu, model dosyalari
2. **Karsilastirma tipini belirle** (`$ARGUMENTS` veya sor):
   - `versions` / `v4 v6`: Versiyon karsilastirmasi
   - `circuits`: Circuit tasarim karsilastirmasi
   - `architectures`: Model mimari karsilastirmasi
   - `all`: Genel karsilastirma

3. **Karsilastirma tablosu olustur**:

```
| Metrik           | V4 (Base) | V6 (Reduced) | V7 (Stable) |
|------------------|-----------|--------------|-------------|
| Feature Map      | 8x8       | 6x6          | 8x8         |
| Quantum Calls    | 16        | 9            | 16          |
| Circuit Type     | fixed     | fixed        | trainable   |
| Epoch Time       | ~1.5h     | ~117s/batch  | TBD         |
| Accuracy         | 8.75%     | 0.00%        | TBD         |
| Gradient Status  | stable    | vanishing    | TBD         |
```

4. **Analiz**: Accuracy vs speed trade-off, gradient stability, mimari etkiler
5. **Oneri**: Sonraki adimlar ve ilgili skill'ler (`/train`, `/architecture`)

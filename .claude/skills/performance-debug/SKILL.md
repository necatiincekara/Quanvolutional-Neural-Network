---
name: performance-debug
description: Yavaslik, scheduler hatasi, dead quantum signal veya hybrid egitim cokusu gibi repo-ozel performans sorunlarini analiz et.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Performance Debug

Bu skill gradient-check'ten daha genis sorunlar icin kullanilir.

## Adimlar

1. Aktif training path'i tespit et.
2. Sorunu ayir:
   - sadece ilk epoch yavas
   - tum egitim yavas
   - loss dusmuyor
   - accuracy cokuyor
   - NaN / dtype sorunu var
3. Repo-ozel sebepleri kontrol et:
   - fazla quantum patch sayisi
   - `q_out.std()` cok kucuk
   - scheduler yanlis yerde step ediliyor
   - AMP / float16 quantum sinirinda sorun cikartiyor
   - trainable vs non-trainable yol karistirilmis
4. Cikti olarak ver:
   - en olasi sebep
   - minimum duzeltme
   - rerun gerekip gerekmedigi

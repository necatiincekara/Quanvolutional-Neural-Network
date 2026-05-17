# Danışman Güncellemesi

**Tarih:** 16 Mayıs 2026

Bu not, çalışmanın tezden makaleye geçiş sürecindeki mevcut durumunu kısa ve gönderilebilir bir formatta özetlemek için güncellenmiştir.

## 1. Kısa Özet

Çalışma, yüksek lisans tezinden çıkan Osmanlı-Türkçesi el yazısı karakter tanıma problemi üzerine kurulu kuantum-klasik hibrit bir OCR araştırmasıdır. Tez aşamasında oluşturulan problem tanımı, veri seti kullanımı ve ilk CNN/HQNN karşılaştırmaları korunmuştur. Mevcut çalışma ise bunu daha sıkı bir yayın protokolüne taşımaktadır.

Bugün gelinen noktada tez boşa çıkarılmış değildir. Tersine, tez:

- problem tanımını kuran temel çalışma olarak korunmaktadır,
- ilk model ailesini sağlayan ana çıkış noktasıdır,
- makale çalışmasında "tez-faithful reproduction" başlığı altında doğrudan yeniden değerlendirilmektedir.

Makale yönünde yapılan şey, tezi reddetmek değil; tezin bulgularını daha güçlü deney disiplini, çoklu seed, ayrı benchmark aileleri, düşük-veri ölçekleme kontrolü ve daha açık mühendislik analizi ile genişletmektir.

## 2. Tezle İlişki Nasıl Kuruluyor?

Makaledeki anlatı şu şekilde kurulmaktadır:

1. **Tez, başlangıç çerçevesini sağlar.**
   Osmanlı-Türkçesi karakter tanıma problemi, veri kullanımı ve ilk CNN/HQNN karşılaştırma zemini tezde kurulmuştur.

2. **Makale, tezdeki ana modelleri yeniden üretir.**
   Özellikle `CNN-III`, `CNN-IIII` ve `HQNN-II` için tez-faithful yeniden üretim hattı kurulmuş ve aynı repo içinde kayıt altına alınmıştır.

3. **Makale, tezin üstüne yeni iki katman ekler.**
   - unified `publication_v1` benchmark protokolü
   - V1--V7 trainable quantum engineering hattı
   - Mayıs 2026 düşük-veri ölçekleme doğrulaması
   - modern classical upper-bound kontrolü (`resnet18_cifar_gray`)

4. **Makale, tek bir "quantum kazandı / kaybetti" söylemine indirgenmez.**
   Bunun yerine:
   - tez-faithful yeniden üretimler,
   - güncel matched-budget lokal ablation'lar,
   - modern classical upper bound,
   - düşük-veri ölçekleme,
   - trainable-quantum engineering case-study
   ayrı ayrı ele alınır.

Bu nedenle tez ile makale arasında çatışma değil, metodolojik olgunlaşma ilişkisi kurulmaktadır.

## 3. Mevcut Sonuç Tablosu

### 3.1 Tez-faithful aile

| Model | Çalıştırma Sayısı | Test Sonucu | Yorum |
|---|---:|---:|---|
| `thesis_cnniiii` | 3 | **85.26 ± 0.97** | tez-faithful en güçlü classical model |
| `thesis_cnn3` | 3 | 79.33 ± 1.26 | classical pairwise referans |
| `thesis_hqnn2` | 3 | 78.61 ± 0.69 | tez-faithful en güçlü quantum reproduction |

### 3.2 Güncel matched-budget lokal aile

| Model | Çalıştırma Sayısı | Test Sonucu | Yorum |
|---|---:|---:|---|
| `classical_conv` | 3 | **81.40 ± 1.06** | güncel matched-budget en güçlü model |
| `param_linear` | 3 | 81.12 ± 2.27 | quantum yerine matched classical replacement |
| `non_trainable_quantum` | 3 | 80.40 ± 0.69 | Henderson-style non-trainable quantum baseline |

### 3.3 Modern classical üst sınır

| Model | Çalıştırma Sayısı | Test Sonucu | Yorum |
|---|---:|---:|---|
| `resnet18_cifar_gray` | 3 | **88.13 ± 0.82** | aynı split üzerinde güçlü modern classical upper bound |

### 3.4 Düşük-veri ölçekleme

Mayıs 2026 doğrulaması, current-local ailede `classical_conv` ile `non_trainable_quantum` modelini 10%, 25%, 50% ve 100% train fraction seviyelerinde karşılaştırdı. Seed 42/43/44 ortalamalarında `non_trainable_quantum` bu dört noktada da `classical_conv` üzerinde kaldı:

| Fraction | `classical_conv` Test | `non_trainable_quantum` Test | Yorum |
|---:|---:|---:|---|
| 0.10 | 48.42 ± 2.31 | **50.71 ± 2.93** | current-local quantum önde |
| 0.25 | 66.24 ± 1.78 | **69.88 ± 0.99** | current-local quantum önde |
| 0.50 | 75.61 ± 1.02 | **76.75 ± 0.50** | current-local quantum önde |
| 1.00 | 80.47 ± 0.57 | **80.76 ± 0.99** | current-local quantum çok dar farkla önde |

Bu sonuç yalnızca current-local matched-budget aile için dar bir düşük-veri rekabet sinyalidir. Tez-faithful düşük-veri pilotunda `thesis_cnniiii`, `thesis_hqnn2` modelinin önünde kalmaktadır.

### 3.5 Trainable quantum case-study

| Model | Durum | Test Sonucu | Yorum |
|---|---|---:|---|
| `V7_trainable_quantum_rerun` | 6 Nisan 2026 resumed Colab L4 rerun | 72.53 | benchmark lideri değil; mühendislik ve stabilizasyon sonucu |
| `V7_trainable_quantum_clean_20260427` | 27-28 Nisan 2026 clean non-resumed Colab L4 rerun | 65.88 | notebook çıktısından yeniden inşa edilmiş JSON satırı; Drive'da checkpoint var, experiments metadata eksik |
| `V7_trainable_quantum_documented` | eski dokümante sonuç | 65.02 | tarihsel karşılaştırma satırı |

## 4. Bu Sonuçlar Nasıl Yorumlanmalı?

En güvenli bilimsel yorum şudur:

- Çalışma şu anda "quantum advantage" iddiası taşıyacak durumda değildir.
- Buna rağmen çalışma yayın değeri taşımaktadır.
- Yayın değeri şu dört eksende oluşmaktadır:
  - adil karşılaştırmalı benchmark,
  - negatif sonuçların dürüst raporlanması,
  - current-local ailede dar düşük-veri rekabet sinyali,
  - hibrit quantum-classical eğitimde mühendislik dersleri.

Burada önemli nokta şu: bu sonuçlar tezi geçersiz kılmamaktadır. Daha doğru ifade şudur:

> Tez, bu problem alanındaki ilk modelleme ve karşılaştırma çerçevesini kurmuştur. Makale ise bu çerçeveyi daha sıkı protokol, çoklu seed, yeniden üretilebilirlik ve mühendislik analizi ile genişletmektedir.

Başka deyişle tez "ilk araştırma katmanı", makale ise "daha yüksek ispat standardına sahip ikinci katman" olarak konumlanmaktadır.

## 5. Makale İçin Yeni Konumlandırma

Makale artık şu başlık altında konumlandırılmaktadır:

**"Ottoman-Turkish handwritten character recognition üzerinde quanvolutional modeller için adil benchmark ve hibrit QML mühendislik analizi"**

Bu konumlandırmada:

- tez-faithful quantum sonuçları korunur,
- classical sonuçların güçlü olduğu dürüstçe söylenir,
- current-local düşük-veri sonucu dar ve doğru kapsamda kullanılır,
- V7 hattı accuracy winner olarak değil, engineering contribution olarak sunulur.

Bu yaklaşım, hem tezi değersizleştirmeden devam etmeyi sağlar hem de yayın için daha savunulabilir bir bilimsel anlatı kurar.

## 6. Bir Sonraki Adımlar

Kısa vadeli plan:

1. makale, benchmark özetleri ve Word export'larını aynı Mayıs 2026 claim setinde tutmak,
2. submission/advisor paketini final hale getirmek,
3. V7 veya düşük-veri deneylerini varsayılan olarak tekrar koşmamak,
4. yalnızca reviewer-risk azaltıyorsa confidence interval/significance testi, ikinci dataset ya da robustness ekseni eklemeyi değerlendirmek.

## 7. Danışmana Gönderilebilecek Kısa Mesaj

Aşağıdaki metin doğrudan e-posta/mesaj gövdesi olarak kullanılabilir:

> Hocam merhaba, tezden çıkan Osmanlı-Türkçesi karakter tanıma çalışmasını yayın odaklı biçimde yeniden yapılandırdım. Tezdeki ana CNN/HQNN modellerini daha sıkı ve yeniden üretilebilir bir protokol altında tekrar koştum; ayrıca matched-budget classical/quantum ablation'lar, güçlü bir modern classical üst sınır ve düşük-veri ölçekleme kontrolü ekledim. Güncel tam-veri sonuçları en güçlü genel kanıtın classical modeller lehine olduğunu gösteriyor. Buna karşılık Mayıs 2026 düşük-veri doğrulamasında current-local `non_trainable_quantum` modeli, `classical_conv` karşısında 10/25/50/100% train fraction seviyelerinde dar ama tutarlı bir rekabet sinyali verdi. Bu nedenle çalışmayı "generic quantum advantage" iddiasıyla değil, "fair benchmark + scoped low-data signal + hybrid QML engineering lessons" ekseninde konumluyorum. Bunu tezi boşa çıkaran bir durum olarak değil, tezin kurduğu problemi ve model ailesini daha yüksek ispat standardıyla genişleten bir ikinci aşama olarak değerlendiriyorum. İsterseniz güncel benchmark tablosunu, düşük-veri figürünü ve revize makale taslağını paylaşabilirim.

## 8. Referans Dosyalar

Bu özetin dayandığı ana repo dosyaları:

- `docs/BENCHMARK_SUMMARY.md`
- `docs/SUBMISSION_BENCHMARK_2026-03-25.md`
- `docs/PUBLICATION_STRATEGY_2026-03-22.md`
- `docs/LOW_DATA_SUMMARY.md`
- `paper/draft.md`
- `paper/figures/low_data_scaling.pdf`
- `experiments/*.json`

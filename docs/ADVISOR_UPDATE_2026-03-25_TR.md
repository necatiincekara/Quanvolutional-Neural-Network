# Danışman Güncellemesi

**Tarih:** 25 Mart 2026

Bu not, çalışmanın tezden makaleye geçiş sürecindeki mevcut durumunu kısa ve gönderilebilir bir formatta özetlemek için hazırlanmıştır.

## 1. Kısa Özet

Çalışma, yüksek lisans tezinden çıkan Osmanlı-Türkçesi el yazısı karakter tanıma problemi üzerine kurulu kuantum-klasik hibrit bir OCR araştırmasıdır. Tez aşamasında oluşturulan problem tanımı, veri seti kullanımı ve ilk CNN/HQNN karşılaştırmaları korunmuştur. Mevcut çalışma ise bunu daha sıkı bir yayın protokolüne taşımaktadır.

Bugün gelinen noktada tez boşa çıkarılmış değildir. Tersine, tez:

- problem tanımını kuran temel çalışma olarak korunmaktadır,
- ilk model ailesini sağlayan ana çıkış noktasıdır,
- makale çalışmasında "tez-faithful reproduction" başlığı altında doğrudan yeniden değerlendirilmektedir.

Makale yönünde yapılan şey, tezi reddetmek değil; tezin bulgularını daha güçlü deney disiplini, çoklu seed, ayrı benchmark aileleri ve daha açık mühendislik analizi ile genişletmektir.

## 2. Tezle İlişki Nasıl Kuruluyor?

Makaledeki anlatı şu şekilde kurulmaktadır:

1. **Tez, başlangıç çerçevesini sağlar.**
   Osmanlı-Türkçesi karakter tanıma problemi, veri kullanımı ve ilk CNN/HQNN karşılaştırma zemini tezde kurulmuştur.

2. **Makale, tezdeki ana modelleri yeniden üretir.**
   Özellikle `CNN-III`, `CNN-IIII` ve `HQNN-II` için tez-faithful yeniden üretim hattı kurulmuş ve aynı repo içinde kayıt altına alınmıştır.

3. **Makale, tezin üstüne yeni iki katman ekler.**
   - unified `publication_v1` benchmark protokolü
   - V1--V7 trainable quantum engineering hattı

4. **Makale, tek bir "quantum kazandı / kaybetti" söylemine indirgenmez.**
   Bunun yerine:
   - tez-faithful yeniden üretimler,
   - güncel matched-budget lokal ablation'lar,
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

### 3.3 Trainable quantum case-study

| Model | Durum | Test Sonucu | Yorum |
|---|---|---:|---|
| `V7 trainable quantum` | dokümante | 65.02 | benchmark lideri değil; mühendislik ve stabilizasyon sonucu |

## 4. Bu Sonuçlar Nasıl Yorumlanmalı?

En güvenli bilimsel yorum şudur:

- Çalışma şu anda "quantum advantage" iddiası taşıyacak durumda değildir.
- Buna rağmen çalışma yayın değeri taşımaktadır.
- Yayın değeri şu üç eksende oluşmaktadır:
  - adil karşılaştırmalı benchmark,
  - negatif sonuçların dürüst raporlanması,
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
- V7 hattı accuracy winner olarak değil, engineering contribution olarak sunulur.

Bu yaklaşım, hem tezi değersizleştirmeden devam etmeyi sağlar hem de yayın için daha savunulabilir bir bilimsel anlatı kurar.

## 6. Bir Sonraki Adımlar

Kısa vadeli plan:

1. makale metninin tamamını yeni benchmark hiyerarşisine göre yeniden yazmak,
2. gerekiyorsa Colab üzerinde V7 için bir confirmatory rerun almak,
3. en az bir güçlü modern classical baseline eklemeyi değerlendirmek,
4. submission versiyonunda ana iddiayı "fair benchmark + engineering lessons" ekseninde sabitlemek.

## 7. Danışmana Gönderilebilecek Kısa Mesaj

Aşağıdaki metin doğrudan e-posta/mesaj gövdesi olarak kullanılabilir:

> Hocam merhaba, tezden çıkan Osmanlı-Türkçesi karakter tanıma çalışmasını yayın odaklı biçimde yeniden yapılandırdım. Bu süreçte tezdeki ana modelleri daha sıkı ve yeniden üretilebilir bir protokol altında tekrar koştum; ayrıca yeni matched-budget classical ve quantum ablation'lar ekledim. Güncel sonuçlar, en güçlü mevcut kanıtın classical modeller lehine olduğunu gösteriyor; dolayısıyla çalışmayı artık "quantum advantage" iddiasıyla değil, "fair benchmark + hybrid QML engineering lessons" ekseninde konumluyorum. Bunu tezi boşa çıkaran bir durum olarak değil, tezin kurduğu problemi ve model ailesini daha yüksek ispat standardıyla genişleten bir ikinci aşama olarak değerlendiriyorum. İsterseniz güncel benchmark tablosunu ve revize makale iskeletini paylaşabilirim.

## 8. Referans Dosyalar

Bu özetin dayandığı ana repo dosyaları:

- `docs/BENCHMARK_SUMMARY.md`
- `docs/SUBMISSION_BENCHMARK_2026-03-25.md`
- `docs/PUBLICATION_STRATEGY_2026-03-22.md`
- `paper/draft.md`
- `experiments/*.json`

# Danışman İnceleme Notu

> **Tarihsel belge:** Drive originals, low-data checkpoint metrikleri, Kaggle lisansı ve yazar bilgileri 9 Ağustos 2026'da güncellenmiştir. Güncel danışman notu: `docs/ADVISOR_UPDATE_2026-08-09_TR.md`.

**Tarih:** 28 Temmuz 2026  
**Çalışma:** Osmanlı-Türkçesi el yazısı karakter tanıma için hibrit quantum-classical OCR  
**İstenen karar süresi:** 48 saat

## Kısa Karar Özeti

Makale, 2024 yüksek lisans tezindeki ilk quantum hipotezinin daha güçlü 2025--2026 deneyleriyle yeniden sınanması olarak tamamlanmıştır. Güncel kanıt genel bir "quantum advantage" göstermemektedir. En güçlü full-data modeller klasiktir. Buna karşılık fixed quantum preprocessing, altı seed'li low-data analizinde klasik convolution kontrolünden ortalamada `0.28--1.79` puan yüksek çıkmıştır; ancak tüm paired %95 güven aralıkları sıfırı kesmekte ve Holm-düzeltilmiş p-değerleri `1.0` olmaktadır. Bu nedenle low-data sonucu yalnız hipotez üretici bir sinyaldir.

Önerim: yeni quantum model arayışıyla aylar kaybetmeden, mevcut çalışmayı **adil benchmark + hipotez revizyonu + hybrid-QML mühendislik dersleri** olarak özel alan bir venue'ya hızla göndermek. İsteğe bağlı tek ek deney, mevcut `param_linear` kontrolünün low-data grididir; yeni mimari değildir ve bir günlük sert süre sınırıyla yapılabilir.

Kapsam ve biçim açısından varsayılan ana hedef olarak **Quantum Machine Intelligence — Research Article** öneriyorum. Dergi uygulamalı hibrit quantum--classical çalışmaları doğrudan kapsıyor, mevcut ayrıntı düzeyindeki araştırma makalelerine keyfî uzunluk tanıyor ve DOCX kabul ediyor. Yedek olarak, açık erişim ücreti karşılanabiliyor ve dataset beyanı kapatılabiliyorsa *Scientific Reports* düşünülebilir.

## Güncel Kanıt

| Aile | Model | Test accuracy | Macro-F1 | Yorum |
|---|---|---:|---:|---|
| modern-classical | ResNet-18 grayscale | `88.13 ± 0.82` | `81.85 ± 1.53` | Genel en güçlü model |
| thesis-faithful | CNN-IIII | `85.26 ± 0.97` | `79.73 ± 1.22` | Tez ailesinde en güçlü model |
| current-local | classical conv | `81.40 ± 1.06` | `71.09 ± 0.43` | Full-data current-local lideri |
| current-local | param-linear control | `81.12 ± 2.27` | `71.29 ± 5.03` | Quantum bloğa matched classical replacement |
| current-local | fixed quanvolution | `80.40 ± 0.69` | `72.76 ± 1.52` | Accuracy lideri değil; macro-F1 deskriptif olarak yüksek |
| thesis-faithful | HQNN-II | `78.61 ± 0.69` | `71.93 ± 1.44` | Güçlü klasik tez modelinin gerisinde |
| trainable-quantum | V7 April reruns | `65.88--72.53` | mevcut değil | Tekil engineering case-study |

Low-data current-local altı-seed sonucu:

| Train fraction | Classical | Fixed quantum | Q-C | Paired %95 CI |
|---:|---:|---:|---:|---:|
| `0.10` | `49.14 ± 2.49` | `50.93 ± 2.72` | `+1.79` | `[-2.93, 6.50]` |
| `0.25` | `67.88 ± 2.13` | `69.17 ± 1.12` | `+1.29` | `[-1.88, 4.45]` |
| `0.50` | `75.36 ± 1.43` | `76.00 ± 1.36` | `+0.65` | `[-1.43, 2.72]` |
| `1.00` | `80.62 ± 0.44` | `80.90 ± 0.95` | `+0.28` | `[-0.44, 1.00]` |

## Makalenin Bilimsel Anlatısı

1. Tez, veri setini, OCR problemini ve ilk CNN/HQNN hipotezini kurdu.
2. Daha sonra multi-seed tekrarlar, daha güçlü ResNet-18 ve parametre-eşlenmiş klasik kontroller eklendi.
3. Bu sonuçlar full-data genel quantum üstünlüğü hipotezini desteklemedi.
4. Hipotez, "fixed quanvolution veri kıtlığında rekabetçi olabilir mi?" ve "trainable hibrit yol hangi koşullarda öğrenir?" sorularına daraltıldı.
5. Low-data analizi sonradan motive edildiği için doğrulayıcı değil, açıkça exploratory/hypothesis-generating olarak yazıldı.

Bu, bilimsel olarak kabul edilebilir bir hipotez-revizyonudur. Tezden kopuş değil; daha sıkı kanıtla tezin hipotezini test edip daraltmaktır.

ResQuNN'ın residual quanvolution yaklaşımını yayımlamış olması nedeniyle residual mimariyi tek başına yenilik olarak kullanmıyoruz. Makalenin yeniliği Osmanlı OCR benchmark'ı, family separation, sınıf-duyarlı kanıt, provenance şeffaflığı ve başarısızlık analizi olarak kurulmuştur.

## Yeni Model Kararı

**Ana submission için yeni model eğitilmesini önermiyorum.**

- V7 seed'i 13--22 saat; güvenilir mimari/hyperparameter kıyası en az üç seed ve matched control ister.
- Dropout, head, pooling veya LR değişimi accuracy'yi artırsa bile kazanç quantum katmana atfedilemez.
- Devre derinliği gradient ve runtime riskini artırır.
- En güçlü klasik modellerle aradaki farkı kapatması düşük olasılıklıdır.
- Yayını haftalar geciktirir ve ana kanıt mesajını güçlendirmez.

İsteğe bağlı tek hızlı kontrol: `param_linear` için 4 fraction × 6 seed low-data grid. Bu deney fixed quantum sinyalinin yalnız seçilen `classical_conv` kontrolüne bağlı olup olmadığını gösterir. Submission için zorunlu değildir.

## Kapatılan Teknik Blocker'lar

- Makaledeki V7 circuit, head, Q-call, attention ve output-gain açıklamaları koda göre düzeltildi.
- Low-data altı seed yerel özetlendi; paired t, exact sign-flip ve Holm düzeltmesi eklendi.
- 21 full-data checkpoint'ten macro-F1, balanced accuracy ve confusion matrix üretildi.
- Yeni V7 kod yolu deterministic split/order ve best-checkpoint testine geçirildi; bu düzeltme eski sonuçlara geriye dönük mal edilmedi.
- Reconstructed JSON'lar açık provenance etiketiyle yerelleştirildi.
- Exact local dependency lock ve SHA-256 artifact manifesti eklendi.
- Makale tez-hypothesis revision anlatısına ve güncel ResQuNN sonrası konuma getirildi.

## Submission Öncesi Kalan İnsan Kararları

1. Dataset'in paylaşım/lisans yetkisi ve etik/onam durumu.
2. Yazar listesi, sırası, kurumları, ORCID ve CRediT katkıları.
3. Funding ve competing-interests beyanları.
4. Ana venue ve bir yedek venue.
5. Bir günlük `param_linear` kontrolünü yapıp yapmama kararı.

Byte-original low-data Drive JSON/checkpointleri de indirilecektir. Bulunamazsa mevcut reconstructed etiket korunacak; sırf dosya hijyeni için pahalı eğitim tekrarlanmayacaktır.

## Sizden İstenen Beş Karar

1. Hipotez-revizyonu ve negatif/karma sonuç anlatısını onaylıyor musunuz?
2. Yazar listesi ve sırası nasıl olmalı; sizin CRediT katkınız hangi rollerdir?
3. Dataset'i açık arşivde veya talep üzerine paylaşmak için yetki/lisans yolu nedir?
4. Ana hedef olarak *Quantum Machine Intelligence — Research Article* rotasını onaylıyor musunuz; onaylamıyorsanız gerekçeli yedek hangisi olmalı?
5. Hemen venue formatına mı geçelim, yoksa bir günlük `param_linear` low-data kontrolünü ekleyelim mi?

## Önerilen Hızlı Takvim

- **Gün 0--2:** Yukarıdaki kararlar, Drive originals ve dataset hakkı.
- **Gün 3--5:** Ana venue template'i, yazar/deklarasyon alanları, optional kontrol kararı.
- **Gün 6--8:** Final dil/biçim, kaynakça ve tüm-yazar onayı.
- **Gün 9--10:** Versioned release, submission PDF ve gönderim.

## Kısa Mesaj Taslağı

> Hocam merhaba, 2024 tezindeki quantum hipotezini daha güçlü klasik kontroller, multi-seed tekrarlar ve trainable V7 deneyleriyle yeniden sınadım. Güncel full-data sonuçları klasik modeller lehine; bu nedenle quantum advantage iddiası kullanmıyorum. Fixed quantum preprocessing altı seed'li low-data analizinde küçük ortalama üstünlükler gösterse de tüm güven aralıkları sıfırı kesiyor; bunu yalnız hipotez üretici sinyal olarak sunuyorum. Makaleyi tezden bağımsızlaşan fakat kronolojisi açık bir "fair benchmark + hypothesis revision + hybrid-QML engineering" çalışmasına dönüştürdüm ve teknik blocker'ları büyük ölçüde kapattım. Kapsam uyumu nedeniyle ilk hedef olarak Quantum Machine Intelligence Research Article öneriyorum. Ekte neredeyse tamamlanmış makale ile kısa karar notu var. Sizden 48 saat içinde özellikle yazar listesi, dataset paylaşım yetkisi, bu venue ve bir günlük ek classical-control deneyini yapıp yapmama konusunda karar rica ediyorum.

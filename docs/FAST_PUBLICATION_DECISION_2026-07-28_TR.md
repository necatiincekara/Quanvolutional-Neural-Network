# Hızlı Yayın Kararı ve Güncellenmiş Aday Matrisi

**İlk sürüm:** 28 Temmuz 2026  
**Son uzlaştırma:** 9 Ağustos 2026  
**Amaç:** En kısa sürede savunulabilir, kaliteli bir yayın üretmek; gereksiz model aramasını durdurmak.

## Yönetici Kararı

Ana yol, **yeni bir quantum mimari eğitmeden mevcut makaleyi hipotez-revizyonu ve adil benchmark çalışması olarak tamamlamaktır**. 2024 tezindeki geniş "quantum yaklaşım yararlı/üstün olabilir" hipotezinin 2025--2026 güçlü klasik kontroller ve yeniden üretimler sonucunda daralması bilimsel olarak kabul edilebilir. Bunun için kronoloji açıkça yazılmalı, sonradan seçilen low-data analizi "exploratory/hypothesis-generating" olarak etiketlenmeli ve sonuçlar preregistered confirmation gibi sunulmamalıdır.

Hız/kalite dengesi için tek düşük maliyetli ek deney adayı, mevcut `param_linear` kontrolünü low-data gridinde aynı seed ve splitlerle çalıştırmaktır. Bu yeni mimari değildir; V7-benzeri quantum yolundaki 25 eğitilebilir parametreyi, 25 parametreli klasik bir linear dönüşümle değiştirir. Low-data fixed-quantum hattının kusursuz tek-değişkenli ikizi değildir, fakat alternatif kapasite kontrollü klasik referanstır. Bu deney özel alan submission'ı için zorunlu değildir ve danışman incelemesini geciktirmemelidir.

V7 katman, dropout, learning-rate, devre derinliği veya kanal sayısı taraması şu anda önerilmez. Bir V7 seed'i yaklaşık 13--22 saat sürmekte, çoklu değişiklik quantum katkısını ayırt etmemekte ve daha iyi doğruluk elde edilse bile ResNet-18 ile `thesis_cnniiii` karşısındaki temel sonucu değiştirmemektedir.

## Hipotez Revizyonu Bilimsel Olarak Kabul Edilebilir mi?

Evet. Önerilen anlatı:

1. **2024 keşif aşaması:** Tez, Osmanlıca el yazısı karakter tanımada CNN/HQNN karşılaştırma alanını ve ilk quantum hipotezini kurdu.
2. **2025--2026 stres testi:** Daha güçlü klasik kontrol, parametre-eşlenmiş kontrol, multi-seed tekrarlar ve trainable V7 mühendislik hattı eklendi.
3. **Hipotez değişimi:** Full-data sonuçları genel quantum üstünlüğünü desteklemedi; araştırma sorusu "hangi dar rejimde rekabetçi olabilir?" ve "trainable hibrit yol neden öğreniyor/çöküyor?" biçiminde daraldı.
4. **Şeffaf statü:** Low-data analizi full-data hiyerarşisi görüldükten sonra motive edildiği için doğrulayıcı değil, hipotez üreticidir.
5. **Sonuç:** Negatif/karma sonuç saklanmadan, daha güçlü bir bilimsel katkıya dönüştürülür.

Bu çerçeve HARKing riskini azaltır; çünkü hipotezin ne zaman ve hangi kanıtla değiştiği söylenir. Teze sıkı mimari sadakat artık bilimsel zorunluluk değildir. Tez, başlangıç noktası ve tarihsel karşılaştırma ailesidir; makale ise bağımsız ve daha katı bir yeniden değerlendirmedir.

## ResQuNN Sonrası Yenilik Konumu

ResQuNN, residual learning ile quanvolution birleşimini 2025'te yayımladığı için yalnızca "residual quantum CNN yaptık" cümlesi güncel bir yenilik iddiası değildir ([DOI](https://doi.org/10.1038/s41598-025-06035-4)). Bu fırsat kaybı makaleyi değersizleştirmez; yenilik iddiası şu dört noktaya taşınmalıdır:

- 44 sınıflı Osmanlı-Türkçe el yazısı OCR için family-separated, artifact-backed QML benchmark;
- tez hipotezinin güçlü klasik kontrollerle açıkça revize edilmesi;
- accuracy yanında macro-F1, balanced accuracy ve nadir sınıf hata analizi;
- V1--V7 başarısızlıklarının, AMP sınırının ve provenance açıklarının dürüst mühendislik analizi.

Literatür, klasik kontrollerin kapasitesi ve veri erişiminin QML iddialarını belirlediğini açıkça göstermektedir ([Huang et al.](https://doi.org/10.1038/s41467-021-22539-9), [Ceschini et al.](https://doi.org/10.1007/s42484-025-00241-z)). Data re-uploading düşük qubit sayısında ifade gücünü artırabilir ([Pérez-Salinas et al.](https://doi.org/10.22331/q-2020-02-06-226)), fakat devre derinliği gradient ve gürültü riskini büyütür ([McClean et al.](https://doi.org/10.1038/s41467-018-07090-4), [Cerezo et al.](https://doi.org/10.1038/s41467-021-21728-w), [Ahmed et al.](https://doi.org/10.1038/s41598-025-17769-6)). Bu nedenle sırf yeni görünmek için derin QCNN, quantum attention veya NAS eklemek hızlı ve güvenli yol değildir.

## Güncellenmiş Aday Karar Matrisi

Puanlar 1 (zayıf/pahalı) ile 5 (güçlü/ucuz) arasındadır. Maliyet sütununda yüksek puan daha düşük maliyeti ifade eder.

| Aday | Bilimsel gerekçe | Çalışmaya uyum | Adil karşılaştırma | Gradient/devre riski | Maliyet | Beklenen bilgi kazancı | Yayın katkısı | Karar |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Mevcut kanıtla hipotez-revizyon makalesi | 5 | 5 | 5 | 5 | 5 | 4 | 5 | **Ana yol** |
| Low-data `param_linear`, 4 fraction × 6 seed | 5 | 5 | 4 | 5 | 5 | 5 | 4 | **İsteğe bağlı tek ek kontrol; danışman paketini geciktirme** |
| V7 learning-rate/dropout/head/pooling taraması | 2 | 4 | 2 | 2 | 1 | 2 | 2 | No-go |
| V7 devre derinliği / data re-upload sayısı | 3 | 4 | 3 | 1 | 1 | 3 | 3 | No-go; reviewer isterse |
| V7-Lite 4→1 kanal bottleneck + matched linear control | 4 | 5 | 5 | 3 | 3 | 4 | 4 | Yalnız venue trainable yenilik isterse |
| QCNN / quantum attention / CQNAS | 2 | 2 | 2 | 1 | 1 | 2 | 2 | Bu makale için no-go |
| Tarihî belge bozulmalarına zero-shot dayanıklılık | 5 | 5 | 4 | 5 | 4 | 5 | 5 | 4--6 haftalık güçlü genişletme |

QCNN ve quantum attention güncel ve aktif alanlardır ([QCNN survey](https://doi.org/10.1109/TNNLS.2026.3677762), [quantum attention](https://doi.org/10.1016/j.engappai.2025.111705)); quantum-classical architecture search de 2026'da yayımlanmıştır ([CQNAS](https://doi.org/10.1016/j.neucom.2026.133236)). Ancak bunları bu veri setine eklemek bağımsız bir araştırma projesi, matched controls ve ciddi hesaplama ister. Hız hedefiyle uyumlu değildir.

## Var Olan V7 Yapısını Değiştirmek Fayda Sağlar mı?

Mutlak validation/test doğruluğunu artırabilir; fakat yayın katkısını artıracağı garanti değildir.

| Değişiklik | Olası fayda | Bilimsel sorun | Karar |
|---|---|---|---|
| Classifier dropout 0.5/0.3 ve head genişliği | Over/underfitting dengesi iyileşebilir | Kazanç quantum bloğa atfedilemez | Şimdi tarama yapma |
| Adaptive pooling 2×2→1×1 | Parametre/maliyet düşer | Bilgi kaybı ve farklı head bütçesi confound yaratır | Yalnız matched control ile |
| Quantum/classical LR oranı | V7 varyansını azaltabilir | Çoklu seed olmadan cherry-picking riski | Reviewer talebine bırak |
| Output gain α / skip β başlangıcı | Quantum/skip yol dengesini değiştirir | Component ablation gerekir | Tek başına sonuç olarak sunma |
| Devre derinliği / re-upload sayısı | İfade gücü artabilir | Barren-plateau, runtime ve gürültü riski | No-go |
| 4 quantum giriş kanalı→1 kanal bottleneck | 64→16 circuit instance; yaklaşık 4× quantum iş yükü azalması | Yeni bottleneck ve classical adapter kontrolü gerekir | V7-Lite yedek adayı |

Mevcut kodda `n_layers` arayüzü ile qnode içindeki katman sayısı tam parametrik değildir; dolayısıyla basit CLI sweep bilimsel olarak güvenilir bir devre-derinliği deneyi sayılmaz. Ayrıca output `gradient_scale`, saf bir gradient operatörü değil öğrenilebilir çıktı kazancıdır. Bu iki nokta düzeltilmeden elde edilen hiperparametre sonuçları kolayca yanlış yorumlanabilir.

## Tek Düşük Maliyetli Ek Kontrol

**Deney:** `param_linear` için `0.10/0.25/0.50/1.00` fraction, seed `42--47`, split seed `42`, fraction seed `42`.

- Toplam: 24 kısa klasik koşu.
- Amaç: fixed quantum'ın low-data ortalama sinyalinin yalnız `classical_conv` seçimine bağlı olup olmadığını sınamak.
- Birincil metrik: paired test accuracy farkı; ikincil macro-F1 ancak tüm checkpointler korunursa.
- Başarı ölçütü: quantum, hem `classical_conv` hem `param_linear` karşısında aynı yönde ve en az iki düşük-data fractionında anlamlı/istikrarlı etki göstermeli.
- No-go: ilk iki fraction tamamlandığında `param_linear` quantum'a eşit/üstünse veya farklar seed yönünde tutarsızsa kalan grid yayın için zorunlu değildir.
- İddia sınırı: olumlu sonuç bile "quantum advantage" değil, bu veri/protokole özgü düşük-veri rekabetçiliğidir.

## Trainable Yenilik Zorunlu Olursa: V7-Lite

Yalnız hedef venue trainable-quantum yeniliği şart koşarsa tek ana mimari adayı:

`8×8×4 -> learned 1×1 Conv (4→1) -> 4-qubit PQC (16 instance/image) -> learned expansion -> classical SE/residual head`

Zorunlu kontrol: PQC yerine aynı giriş/çıkış boyutlu, parametre-eşlenmiş linear/trigonometric classical map. En az üç seed gerekir; ideal olarak seed 42--47. İlk gate seed 42, üç epoch:

- non-finite değer varsa durdur;
- validation <%35 ise durdur;
- median quantum gradient `<1e-5` ise durdur;
- eski V7'ye göre en az 3× hızlanmıyorsa durdur;
- seed-42 test/validation eski temiz V7'nin 2 puan içinde değilse çoklu seed'e geçme.

Bu aday mevcut hızlı gönderim yoluna dahil değildir.

## Alana Yeni Soluk Getirebilecek En Uygun Fikir

En uyumlu yeni eksen, **tarihî belge bozulması altında veri kıtlığı × dayanıklılık** etkileşimidir: blur, erosion/dilation, ink bleed, kâğıt arka planı, düşük kontrast ve tarama gürültüsü. Önce mevcut checkpointler zero-shot değerlendirilir; böylece eğitim maliyeti olmadan hangi temsilin tarihî bozulmalara daha dayanıklı olduğu ölçülür. Ardından yalnız sinyal varsa kontrollü fine-tuning düşünülür.

Bu fikir Osmanlı OCR ile doğal olarak ilişkilidir ve ikinci veri seti olmadan dış-geçerlilik eksenini güçlendirebilir. Ancak "ilk çalışma" iddiası hedefli literatür taraması yapılmadan kullanılmamalıdır. Hızlı gönderimde gelecek çalışma olarak, 4--6 haftalık daha güçlü rotada ana ek deney olarak konumlandırılmalıdır.

## Yayın Stratejisi

- **Hızlı dergi rotası:** *Quantum Machine Intelligence*, Research Article. Derginin resmî kapsamı quantum machine learning, quantum neural networks, quantum image/signal processing ve uygulama odaklı hibrit quantum--classical çalışmaları açıkça içerir. Research Article için keyfî uzunluk, DOCX/LaTeX kaynak dosyası, 150--250 kelimelik özet ve yazar--yıl kaynak biçimi kabul edilir. Dergi hibrittir ve publication/page charge “None” belirtir; isteğe bağlı açık erişim maliyeti ayrıca doğrulanmalıdır.
- **Hakemli bildiri rotası:** *ICPRAM 2027* Regular Paper. Pattern recognition, document analysis ve image understanding kapsamı uygundur; son tarih 15 Eylül 2026'dır. Mevcut taslak 10.000--50.000 karaktere ve kabul sonrası 12 sayfalık full/8 sayfalık short formata sıkıştırılmalı, double-blind hazırlanmalı ve en az bir yazar Malta'da sunum yapmalıdır. Erken non-member speaker kaydı 620 €'dur; seyahat ayrıca gerekir.
- Geniş kapsamlı dergi alternatifleri ancak maliyet, veri politikası ve editör uyumu güncel resmî sayfalardan doğrulandıktan sonra seçilmelidir.
- Aynı tam makale eşzamanlı olarak bir dergiye ve arşivsel bildiriye gönderilmemelidir.
- Hedef venue izin veriyorsa preprint + tek arşivsel submission yapılabilir.
- Farklı ve açıkça kısa/non-archival bir workshop özeti düşünülebilir; içerik ve çift-yayın politikası venue bazında doğrulanmalıdır.
- En hızlı savunulabilir rota: özel alan QML / applied ML / digital humanities-OCR dergisi veya konferansı.
- Geniş kapsamlı güçlü venue rotası: robustness ekseni veya ikinci veri seti ve daha güçlü istatistik gerektirir.

## İki Haftalık Hızlı Yol Haritası

| Gün | Çıktı | Stop/başarı ölçütü |
|---|---|---|
| 0--1 | İlk-yazar metadata/COI formu ve QMI/ICPRAM kararı | Drive ve dataset blocker'ları kapalıdır |
| 0--5 | Danışman ve ortak yazar bilimsel incelemesi | Claim dili: advantage yok, low-data exploratory |
| 1--3 | Venue template'ine paralel manuscript aktarımı | Tüm tablo/şekil/sayfa sınırı uyumlu |
| 2--3 | İsteğe bağlı `param_linear` low-data kararı | Danışman paketini geciktirme; en fazla bir gün, açık uçlu tuning yok |
| 7--9 | Dil, kaynakça, declarations, data/code availability | Placeholder kalmamalı |
| 9--10 | Versioned release ve SHA manifest | Gönderilecek PDF ile artifact paketi aynı sürüm |
| 10--14 | Final tüm-yazar onayı ve submission | Tek arşivsel venue; politikalar doğrulanmış |

## Dört--Altı Haftalık Güçlendirilmiş Yol Haritası

Bu rota yalnız hedef venue daha güçlü dış-geçerlilik isterse seçilmelidir:

1. Hafta 1: Tarihî bozulma protokolü ve severity seviyeleri; test-only, deterministic corruption seti.
2. Hafta 2: Mevcut checkpointlerle zero-shot robustness; classical, fixed-quantum ve ResNet kontrolleri.
3. Hafta 3: Paired degradation istatistiği, macro-F1 ve per-class morphology analizi.
4. Hafta 4: Sinyal varsa tek kontrollü fine-tuning/augmentation ablation; yoksa negatif robustness sonucu.
5. Hafta 5: İkinci dataset ancak hukuken ve teknik olarak erişilebiliyorsa external validation.
6. Hafta 6: Makale yeniden konumlandırma, ek reviewer audit ve venue submission.

Bu rotada da V7 hyperparameter sweep varsayılan değildir. Robustness ekseni mevcut checkpointleri kullanarak daha yüksek bilgi kazancı sağlar.

## Kesin Karar

1. **Yeni model eğitilsin mi?** Hayır. Ana submission için yeni quantum model eğitilmemeli.
2. **İsteğe bağlı tek ek deney nedir?** Mevcut `param_linear` low-data matched control; bir günlük sert süre sınırıyla.
3. **V7 hiperparametre taraması yapılsın mı?** Hayır. Bilgi kazancı/süre oranı düşük ve quantum katkısını izole etmiyor.
4. **Novel architecture ne zaman?** Yalnız reviewer/venue trainable yenilik isterse V7-Lite; mevcut makaleyi geciktirmemeli.
5. **Venue kararı nedir?** Maliyet/seyahat istemiyorsanız *Quantum Machine Intelligence*; deadline'lı hakemli bildiri istiyorsanız *ICPRAM 2027*. Danışmandan venue seçmesi beklenmemelidir.

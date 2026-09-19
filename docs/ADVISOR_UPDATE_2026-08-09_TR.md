# Çalışmanın Güncel Durumu ve Makale Taslağı

**Tarih:** 11 Ağustos 2026  
**Çalışma:** Osmanlı-Türkçe el yazısı karakter tanımada hibrit kuantum-klasik OCR  
**Yazarlar:** Necati İncekara, Erdem Bilgili

Hocam merhaba,

2024 yılında tamamladığım yüksek lisans tezinden çıkan çalışmayı, tez sonrasında yaptığım yeni deneylerle genişleterek makale hâline getirdim. Bu süreçte tezdeki modelleri yeniden çalıştırdım; daha güçlü klasik modeller, farklı rastgele başlangıçlarla tekrarlanan deneyler, parametre sayısı gözetilerek hazırlanan kontroller ve eğitilebilir kuantum model deneyleri ekledim. Ayrıca sonuçları yalnız doğruluk üzerinden değil, macro-F1 ve sınıf bazlı performans açısından da yeniden değerlendirdim.

Yeni deneyler, başlangıçtaki genel kuantum üstünlüğü beklentisini desteklemedi. Bununla birlikte çalışma; küçük ve dengesiz bir Osmanlıca karakter veri setinde kuantum ve klasik modellerin adil biçimde karşılaştırılması, düşük veri koşullarında görülen sınırlı rekabet sinyali ve eğitilebilir kuantum modellerde karşılaşılan optimizasyon sorunları bakımından anlamlı sonuçlar verdi. Makaleyi de bu doğrultuda, tezdeki ilk hipotezin daha güçlü deneysel kanıtlarla yeniden değerlendirilmesi olarak kurguladım.

## Başlıca Sonuçlar

| Model grubu | Model | Test doğruluğu | Kısa değerlendirme |
|---|---|---:|---|
| Modern klasik model | ResNet-18 | **88.13 ± 0.82%** | Çalışmadaki en yüksek test sonucu |
| Teze sadık klasik model | `thesis_cnniiii` | **85.26 ± 0.97%** | Tez modelleri içindeki en güçlü sonuç |
| Teze sadık kuantum model | `thesis_hqnn2` | **78.61 ± 0.69%** | En güçlü teze sadık klasik modelin gerisinde |
| Güncel yerel klasik kontrol | `classical_conv` | **81.40 ± 1.06%** | Bu karşılaştırma grubundaki en yüksek ortalama |
| Parametre-eşlenmiş klasik kontrol | `param_linear` | **81.12 ± 2.27%** | Kuantum katmanın yerine küçük bir klasik dönüşüm kullanıyor |
| Sabit kuantum ön işleme | `non_trainable_quantum` | **80.40 ± 0.69%** | Klasik kontrolleri aşmıyor |
| Eğitilebilir kuantum model | V7 | **65.88–72.53%** | Performans lideri değil; modelin eğitilebilir hâle gelmesi açısından inceleniyor |

Tam veriyle yapılan deneylerde en güçlü sonuçlar klasik modellerden geldi. Düşük veri deneylerinde ise `non_trainable_quantum`, `classical_conv` ortalamasının dört veri oranında da 0.28 ile 1.79 puan üzerinde kaldı. Ancak güven aralıklarının tamamı sıfırı kesiyor ve çoklu karşılaştırma düzeltmesinden sonra anlamlı bir fark elde edilmiyor. Bu nedenle makalede bu bulguyu bir üstünlük sonucu olarak değil, ileride ayrı bir çalışmayla sınanabilecek sınırlı bir gözlem olarak aktardım.

Kaydedilmiş model ağırlıkları üzerinden hesaplanan macro-F1 değerleri de benzer bir eğilim gösteriyor. Kuantum model ile klasik kontrol arasındaki macro-F1 farkları yüzde 10, 25, 50 ve 100 eğitim verisi için sırasıyla +3.00, +0.74, +0.35 ve +2.46 puan. Bu değerleri doğruluk sonuçlarına eşlik eden betimleyici ölçümler olarak kullanıyorum; ayrıca doğrulanmış bir kuantum üstünlüğü iddiasına dönüştürmüyorum.

## Makalede Benimsediğim Çerçeve

Tezi çalışmanın başlangıç noktası olarak korudum. Makalede 2024'teki ilk CNN/HQNN karşılaştırmalarını, sonradan eklenen daha güçlü kontrollerden ve eğitilebilir kuantum model deneylerinden ayrı ele alıyorum. Böylece farklı amaçlarla kurulmuş modeller tek bir sıralama içinde karşılaştırılmıyor.

Makalede öne çıkan katkılar şunlar:

- Osmanlı-Türkçe el yazısı karakter tanıma problemi için farklı rastgele başlangıçlarla tekrarlanan karşılaştırmalı deneyler;
- teze sadık modeller, güncel yerel kontroller ve eğitilebilir kuantum modeller arasında açık ayrım;
- doğruluk yanında macro-F1, dengeli doğruluk ve sınıf bazlı hata incelemesi;
- V1-V7 sürecinde karşılaşılan bilgi kaybı, gradyan çökmesi ve karma hassasiyetli eğitimden kaynaklanan sorunların kayda geçirilmesi;
- olumlu ve olumsuz sonuçların aynı deneysel çerçeve içinde raporlanması.

Bu anlatımda “quantum advantage” iddiasında bulunmuyorum. Sonuçları, kuantum modellerin bu veri setindeki sınırlarını ve hangi koşullarda rekabetçi görünebildiğini ortaya koyan karşılaştırmalı bir çalışma olarak yorumluyorum.

## Veri ve Deney Kayıtları

Kaggle'da yayımlanan *Ottoman Turkish Characters* veri setinin 3.894 görüntüsü ile deneylerde kullandığım yerel dosyaların tamamı karşılaştırıldı ve içerik bakımından aynı olduğu doğrulandı. Veri seti sayfasında lisans “GPL 2” olarak belirtiliyor. Makalede görüntüleri yeniden dağıtmak yerine özgün Kaggle sayfasına bağlantı veriyorum:

https://www.kaggle.com/datasets/alpbintuuzun/ottoman-turkish-characters

Düşük veri deneylerine ait 56 sonuç dosyası ve kaydedilmiş 56 model ağırlığı doğrulandı. Daha önce notebook çıktılarından yeniden oluşturulan sonuçlar, Drive'dan indirilen özgün JSON dosyalarıyla karşılaştırıldı ve sayısal sonuçların eşleştiği görüldü. V7 deneylerinin model ağırlıkları mevcut; çalışma sırasında üretilen bazı sonuç JSON'ları ise Colab oturumu kapandığı için Drive'a aktarılamamış. Bu durum makalede açıkça belirtiliyor ve V7 sonuçları yalnızca bir mühendislik incelemesi olarak kullanılıyor.

## Yazar Bilgileri ve Beyanlar

Taslakta yazar sırasını şu şekilde yazdım:

1. Necati İncekara
2. Erdem Bilgili — Faculty of Engineering, Piri Reis University, Istanbul, Türkiye

Katkı rollerini şimdilik aşağıdaki biçimde hazırladım:

- **Necati İncekara:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing — original draft.
- **Erdem Bilgili:** Supervision, Methodology, Writing — review & editing, Project administration.

Bu dağılımın fiilî katkıları doğru yansıtıp yansıtmadığı konusunda görüşünüzü rica ederim. Çalışma için dış fon kullanılmadığından funding beyanını “This research received no external funding” şeklinde yazdım. Competing interests bulunmadığını da teyit ederseniz ilgili beyanı “The authors declare no competing interests” olarak tamamlayacağım.

## Görüşünüzü Rica Ettiğim Noktalar

Taslağı incelerken özellikle şu konulardaki değerlendirmeniz benim için önemli:

1. Tezden makaleye uzanan araştırma çizgisi ve hipotezin yeni sonuçlar ışığında yeniden değerlendirilmesi sizce doğru kurulmuş mu?
2. Sonuçların yorumunda gereğinden güçlü veya eksik bulduğunuz bir ifade var mı?
3. Yazar sırası, kurum bilgisi ve katkı rolleri uygun mu?
4. Bilimsel içerik bakımından eklenmesi, çıkarılması veya daha açık anlatılması gereken bir bölüm görüyor musunuz?

Uygun olduğunuzda taslak üzerindeki notlarınızı ve gerekli gördüğünüz düzeltmeleri paylaşabilirseniz, metni son hâline getirip gönderim hazırlıklarını tamamlamak istiyorum.

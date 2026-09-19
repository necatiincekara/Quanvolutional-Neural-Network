# Kritik Submission Blocker Kapatma Kılavuzu

**İlk sürüm:** 28 Temmuz 2026  
**Son uzlaştırma:** 9 Ağustos 2026  
**Kullanıcı:** Necati İncekara  
**Amaç:** Makaleyi hız kaybetmeden danışman incelemesine, ardından tek bir peer-reviewed venue'ya hazır hâle getirmek.

## Güncel Karar

Repository ve veri tarafındaki kritik blocker'lar kapatıldı. Danışmana belge göndermek için yeni model, yeni eğitim, Drive indirme veya ek dataset izni beklemeyin. Dış submission öncesinde yalnız insan kararı gerektiren alanlar tamamlanmalıdır.

## Blocker Tablosu

| Konu | Durum | Kanıt / yapılacak iş |
|---|---|---|
| Low-data Drive JSON | **Kapatıldı** | 56/56 beklenen JSON yerel; 40 eski reconstruction byte-original dosyalarla uzlaştırıldı |
| Low-data checkpoint | **Kapatıldı** | 56/56 best checkpoint yerel ve yüklenebilir |
| Low-data class-aware metrik | **Kapatıldı** | 48 current-local koşu için macro-F1, balanced accuracy ve confusion matrix üretildi |
| Dataset kaynağı/lisansı | **Kapatıldı** | Kaggle kaynağı açık; sayfa lisansı “GPL 2”; 3.894/3.894 yerel PNG resmî arşivle içerik olarak aynı |
| V7 raw JSON | Açık fakat blocker değil | Drive'da checkpoint var, raw JSON yok; mevcut satırlar dürüstçe reconstructed olarak kalacak |
| Yazar listesi | Büyük ölçüde kapalı | Necati İncekara, Erdem Bilgili; sıra ve final onay iki yazarca doğrulanacak |
| Erdem Bilgili kurumu | Hazır, teyit gerekli | Mühendislik Fakültesi, Piri Reis Üniversitesi, İstanbul, Türkiye |
| Funding | **Kapatıldı** | “This research received no external funding.” |
| İlk yazar bilgileri | **Kritik açık** | Kurum, bölüm, e-posta, ORCID ve corresponding-author kararı |
| COI ve CRediT | **Kritik açık** | İki yazarın yazılı teyidi |
| Hedef venue/template | Açık | Danışmanın seçmesi gerekmiyor; yazarın hız/maliyet tercihine göre seçilecek |
| Final ortak-yazar onayı | **Kritik açık** | Gönderilecek nihai dosyaya yazılı onay |

## 1. Tamamlanan Teknik İşler — Sizden İşlem Gerekmiyor

`experiments/Drive/quanv_results` altındaki klasörün tamamı incelendi:

- 138 dosya, yaklaşık 20,6 MB;
- 91 JSON ve 44 Drive checkpoint'i;
- mevcut seed-42 dosyalarıyla birlikte canonical dizinlerde 56 low-data JSON ve 56 checkpoint;
- daha önce notebook çıktısından oluşturulmuş 40 JSON'un sonuç alanları Drive originals ile birebir uyumlu;
- 48 current-local low-data checkpoint'iyle sınıf-duyarlı test metrikleri yeniden hesaplandı;
- V7 klasörlerinde iki resumed ve iki clean checkpoint var, fakat raw sonuç JSON'u yok.

Denetim dosyaları:

- `experiments/drive_artifact_reconciliation_20260809.json`
- `experiments/low_data_classification_metrics_20260809.json`
- `docs/LOW_DATA_CLASSIFICATION_METRICS_2026-08-09.md`
- `docs/ARTIFACT_PROVENANCE_2026-07-28.md`

Drive klasörlerini tekrar indirmeniz veya dosyaları elle taşımanız gerekmiyor.

## 2. Dataset İzni — Sonuç ve Kullanım Kuralı

Kaynak: https://www.kaggle.com/datasets/alpbintuuzun/ottoman-turkish-characters

- Dataset adı: *Ottoman Turkish Characters*
- Dataset yazarları: Alperen Özer ve Alp Bintuğ Uzun
- Kaggle metadata lisans etiketi: **GPL 2**
- Resmî arşiv: 3.894 PNG; yerel 3.894 PNG'nin tamamı içerik-hash bakımından aynı
- Resmî ZIP SHA-256: `35b68d7f7e677e591d2305c573ce350914042f0f7e5b2fa79d2cb16415563885`

Makalede kullanılacak metin:

> The *Ottoman Turkish Characters* dataset by Alperen Özer and Alp Bintuğ Uzun is publicly available at https://www.kaggle.com/datasets/alpbintuuzun/ottoman-turkish-characters under the dataset-page license label “GPL 2.” The experiments use its original train/test directories. On August 9, 2026, all 3,894 local PNG files were verified content-identical to the official version-1 archive. To preserve source attribution and license context, the research package directs users to Kaggle rather than redistributing the images.

Uygulama kuralı: dataset görsellerini makale artefakt ZIP'ine yeniden koymayın. Kaggle kaynağına bağlantı verin. Görseller daha sonra yeniden dağıtılacaksa GPL 2.0 lisans ve attribution yükümlülüklerini ayrıca koruyun.

## 3. Sizin Dolduracağınız Yazar Formu

Aşağıdaki alanları tek mesajda doldurun. Bilmediğiniz ORCID için “yok” yazabilirsiniz; ORCID çoğu venue'da önerilir ama her zaman zorunlu değildir.

```text
Necati İncekara — kurum:
Bölüm/birim:
Şehir/ülke:
Submission için kullanılacak e-posta:
ORCID:
Corresponding author: Necati / Erdem
Yazar sırası: Necati İncekara, Erdem Bilgili — onaylı mı? Evet/Hayır
Competing interests: None / açıklama
Erdem Bilgili'nin ORCID'i (biliniyorsa):
```

Bilinen ikinci-yazar bilgisi:

```text
Erdem Bilgili
Faculty of Engineering, Piri Reis University, Istanbul, Türkiye
ebilgili@pirireis.edu.tr
```

Önerilen CRediT taslağı; iki yazarın onayı olmadan final değildir:

- **Necati İncekara:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing — original draft.
- **Erdem Bilgili:** Supervision, Methodology, Writing — review & editing, Project administration.

Danışmanınıza “bu roller fiilî katkıyı doğru yansıtıyor mu?” diye sorun; gerekirse değiştirin.

## 4. `param_linear` Nedir?

`param_linear`, ayrı bir büyük model veya yeni bir quantum mimari değildir. V7 benzeri ağda quantum katmanın bulunduğu yere küçük bir **klasik doğrusal dönüşüm** koyan kontrol modelidir.

Quantum yol 24 devre açısı + 1 öğrenilebilir çıktı katsayısı olmak üzere 25 eğitilebilir yol parametresi kullanır. `param_linear` da aynı 2×2 yamaları işler ve 20 linear parametre + 1 ölçek + 4 kanal bias olmak üzere tam 25 eğitilebilir parametre kullanır. Ağın geri kalanı aynı V7-benzeri iskeleti korur.

Bu kontrol şu soruyu sorar:

> Gözlenen sonuç quantum devresine özgü bir özellikten mi geliyor, yoksa aynı yerde yalnızca küçük ve öğrenilebilir bir dönüşüm bulunması yeterli mi?

Full-data sonucu zaten vardır: `param_linear` **81.12 ± 2.27%**, fixed `non_trainable_quantum` **80.40 ± 0.69%** test accuracy. Dolayısıyla mevcut full-data kanıt quantum'a özgü bir kazanım göstermiyor.

Önemli sınır: low-data karşılaştırmasındaki `non_trainable_quantum` sabit ve önceden hesaplanan quantum özelliklerini kullandığı için `param_linear` onun kusursuz tek-değişkenli ikizi değildir; alternatif, kapasite kontrollü bir klasik referanstır. Low-data gridini eklemek reviewer dayanıklılığını artırabilir, fakat danışmana bugünkü gönderimi geciktirecek kritik blocker değildir.

**Hız odaklı karar:** Makaleyi bugün danışmana gönderin. `param_linear` low-data deneyi ancak siz ayrıca onaylarsanız, en fazla bir günlük sert bütçe ve önceden sabitlenmiş protokolle daha sonra çalıştırılsın. Açık uçlu tuning yapılmasın.

## 5. Venue Kararı — Danışmana Yüklemeyin

Danışmanınız venue seçimine karışmıyorsa ondan yalnız bilimsel anlatı ve ortak-yazarlık onayı isteyin. Operasyonel venue kararını siz verin.

### Ana hızlı rota: Quantum Machine Intelligence

- Konu uyumu en yüksek dergi seçeneği: QML, quantum neural networks, quantum preprocessing, quantum image processing ve hybrid uygulamalar doğrudan kapsamda.
- Research Article için keyfî uzunluk kabul ediliyor.
- DOCX kabul ediliyor; author-year kaynakça mevcut taslakla uyumlu.
- Resmî sayfada publication/page charge “None”; açık erişim seçilirse isteğe bağlı APC ayrıca doğabilir.
- Sabit konferans deadline'ı yok; yazar alanları kapanınca submission yapılabilir.

Resmî sayfalar:

- https://link.springer.com/journal/42484/aims-and-scope
- https://link.springer.com/journal/42484/submission-guidelines

### Bildiri rotası: ICPRAM 2027

- Pattern recognition, document analysis ve image understanding kapsamına uyuyor.
- Regular paper son tarihi: **15 Eylül 2026 (AoE)**.
- Double-blind PDF gerekir; mevcut uzun taslağın 10.000–50.000 karaktere ve kabul sonrası 12 sayfalık full / 8 sayfalık short formata sıkıştırılması gerekir.
- Proceedings DOI alır ve Scopus/DBLP vb. indekslemeye sunulur; indekslenme garantisi olarak yazılmamalıdır.
- En az bir yazar sunum yapmalı; erken non-member speaker registration **620 €**, seyahat/konaklama ayrıca gerekir.
- Konferans 20–22 Şubat 2027'de Malta'dadır.
- AI-generated text disclosure/citation politikası özellikle uygulanmalıdır.

Resmî sayfalar:

- https://icpram.scitevents.org/CallforPapers.aspx
- https://icpram.scitevents.org/ImportantDates.aspx
- https://icpram.scitevents.org/Guidelines.aspx
- https://icpram.scitevents.org/RegistrationFees.aspx

**Önerim:** Danışmana şimdi venue-agnostic bilimsel taslağı gönderin. Maliyet ve seyahat istemiyorsanız QMI ana yoludur. Hakemli bildiri ve kesin deadline motivasyonu istiyorsanız, 620 € + seyahati kabul ederek ICPRAM'ı seçin. Aynı tam metni eşzamanlı olarak dergiye ve arşivsel konferansa göndermeyin.

## 6. Danışmana Bugün Gönderilecek Paket

İlk e-postada yalnız iki dosya gönderin:

1. `paper/draft.docx` — incelemeye hazır ana İngilizce makale.
2. `docs/ADVISOR_UPDATE_2026-08-09_TR.docx` — üç sayfayı aşmayan Türkçe yönetici özeti, bulgular ve istenen onaylar.

İlk e-postaya blocker kılavuzunu, büyük artefakt klasörünü, checkpointleri veya ayrıntılı karar matrisini eklemeyin. Danışman teknik kanıt isterse ikinci turda şu dosyaları paylaşın:

- `docs/LOW_DATA_CLASSIFICATION_METRICS_2026-08-09.md`
- `docs/STATISTICAL_EVIDENCE_2026-05-17.md`
- `docs/ARTIFACT_PROVENANCE_2026-07-28.md`
- `experiments/submission_artifact_manifest_20260728.json`

Danışmandan şu dört kararı isteyin:

1. Tezden makaleye hipotez-revizyonu ve “quantum advantage yok” anlatısı bilimsel olarak uygun mu?
2. Necati İncekara — Erdem Bilgili yazar sırası ve önerilen CRediT rolleri doğru mu?
3. Erdem Bilgili'nin kurum/e-posta bilgisi ve “no external funding” beyanı doğru mu; competing interests “none” olarak yazılabilir mi?
4. Ana metin bu düzeltmelerle submission'a ilerleyebilir mi? Lütfen belirli cümle/tablo değişikliklerini işaretler misiniz?

Venue seçmesini veya `param_linear` kararını danışmana yüklemeyin; bunlar hız, maliyet ve benchmark stratejisi kararlarıdır.

## 7. Bugün Adım Adım Yapacağınız İş

1. Bu kılavuzdaki yazar formunu doldurup Codex'e gönderin.
2. Güncellenmiş iki DOCX'i açın; adınızın yazımı ve metnin genel tonu için hızlı göz kontrolü yapın.
3. Aşağıdaki e-posta metnini danışmanınıza gönderin ve iki DOCX'i ekleyin.
4. Beş iş günü içinde yanıt isteyin; üç iş gününde yanıt yoksa kısa hatırlatma gönderin.
5. Aynı gün, kendi bütçenize göre QMI veya ICPRAM kararını verin. Venue formatlamasını danışman geri bildirimi beklerken paralel yapın.
6. Danışman düzeltmeleri gelince paper'da uygulayın; iki yazar final dosyayı yazılı onayladıktan sonra tek venue'ya gönderin.

## 8. E-posta Taslağı

**Konu:** Osmanlıca hybrid quantum-classical OCR makalesi — bilimsel inceleme ve yazar onayı

> Hocam merhaba, 2024 yüksek lisans tezindeki çalışmayı daha güçlü klasik kontroller, çoklu seed deneyleri, sınıf-duyarlı metrikler ve trainable-quantum mühendislik analiziyle güncelleyerek makale hâline getirdim. Sonuçlar genel bir quantum advantage göstermiyor; makale bunu saklamadan, tez hipotezinin yeni kanıtlarla daralması ve adil benchmark katkısı olarak konumlandırıyor. Drive artefaktları ve Kaggle dataset kaynağı/lisansı da doğrulandı. Ekte İngilizce ana taslak ile kısa Türkçe karar notu var. Özellikle bilimsel anlatının uygunluğunu, yazar sırası/katkı rollerini, beyanları ve submission'a ilerleme onayınızı rica ediyorum. Mümkünse beş iş günü içinde doğrudan belge üzerinde düzeltme iletebilir misiniz? Teşekkür ederim.

## 9. Submission Öncesi Son İnsan Kontrolü

- [ ] Necati İncekara kurum, e-posta, ORCID ve corresponding-author bilgisi final.
- [ ] Erdem Bilgili kurum/e-posta bilgisi ve yazar sırası teyitli.
- [ ] CRediT rolleri iki yazarca onaylı.
- [ ] Funding “no external funding”; COI beyanı iki yazarca onaylı.
- [ ] Venue seçilmiş ve güncel template uygulanmış.
- [ ] Metinde “quantum advantage” veya “confirmed low-data advantage” iddiası yok.
- [ ] V7 reconstructed JSON, original gibi sunulmuyor.
- [ ] Dataset kaynak bağlantısı ve lisans etiketi metinde var; dataset görselleri artefakt paketine eklenmiyor.
- [ ] İki yazar gönderilecek nihai dosyayı yazılı onayladı.
- [ ] Aynı tam çalışma başka bir arşivsel venue'da incelemede değil.

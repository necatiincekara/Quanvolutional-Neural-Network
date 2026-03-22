# Codex Workflows

**Tarih:** 22 Mart 2026

Bu belge, bu repoda Codex'i daha etkili kullanmak icin eklenen yeni calisma modlarini ve tekrar kullanilabilir komutlari ozetler.

## 1. Eklenen Yeni Codex Ozellikleri

Bu turda aktif olarak eklenen ilave Codex gucleri:

- config profilleri
- non-interactive `codex exec` script'leri
- makale senkronizasyonu icin `paper-sync` skill'i
- daha genis runtime/ogrenme analizi icin `performance-debug` skill'i
- write-focused `paper_writer` subagent'i

Bu eklemelerin amaci, ayni repoda arastirma, debugging ve makale yazimini ayri ama tekrar kullanilabilir akislara ayirmak.

## 2. Profiller

`/.codex/config.toml` icinde uc yeni profil var:

- `paper`
  - makale ve tez duzeltmeleri
  - yuksek reasoning
  - `live` web search
  - daha noro akademik dil icin `personality = "none"`
- `review`
  - read-only audit ve teknik inceleme
  - repo icinde risk arama, circuit review, status audit
- `fast_local`
  - daha hizli yerel iterasyon
  - orta seviye reasoning

Ornek kullanim:

```bash
codex -p paper
codex exec -p review "Use the status skill and audit the current study state."
```

## 3. Hazir Script'ler

Tekrar tekrar prompt yazmamak icin su script'ler eklendi:

```bash
./scripts/codex-study-status.sh
./scripts/codex-paper-audit.sh
./scripts/codex-circuit-review.sh
./scripts/codex-circuit-review.sh src/model.py
```

Bu script'ler `codex exec` kullanir, uygun profili secip `--include-plan-tool` ve web search destegi ile calisir.

## 4. Etkili Interactive Akis

Bu repo icin onerilen Codex komut dizisi:

1. `/status`
2. `/plan`
3. goreve gore ilgili skill
4. uzun oturumlarda `/compact`
5. commit oncesi `/review`
6. subagent ciktilari icin `/agent`

Ozellikle asagidaki durumlarda bu akis yuksek deger uretir:

- birden fazla belge ve deney sonucu celisiyorsa
- paper veya thesis metninde claim guncellenecekse
- quantum katman davranisi tekrar incelenecekse

## 5. Makale Bitirme Akisi

Sen bana son Claude session'unu verdiginde bu repoda onerilen ilerleme sirasi su olacak:

1. `./scripts/codex-study-status.sh`
2. `./scripts/codex-paper-audit.sh`
3. `docs/PUBLICATION_STRATEGY_2026-03-22.md` ile hedef yayin rotasini sabitle
4. gerekirse `paper_writer` ile hedefli draft duzeltmesi
5. `README.md`, `docs/EXPERIMENTS.md` ve `paper/draft.md` arasinda son senkronizasyon

## 6. Neden Bunlar Eklendi

Bu repo klasik bir kod reposu degil. Burada ayni anda:

- model gelistirme
- deney takibi
- hata ayiklama
- Colab senkronizasyonu
- makale yazimi

yurutuluyor. Tek bir genel Codex oturumu bunlari yapsa da, profil + skill + subagent + reusable script kombinasyonu daha hizli ve daha tutarli calisiyor.

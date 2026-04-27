# Codex Workflows

**Tarih:** 22 Mart 2026

> Durum notu, 5 Nisan 2026:
> Bu belge ilk workflow katmanini anlatiyordu. Guncel operasyon modeli icin artik su belgeleri birlikte kullan:
> - `docs/CODEX_CAPABILITY_AUDIT_2026-04-05.md`
> - `docs/CODEX_GAP_ANALYSIS_2026-04-05.md`
> - `docs/CODEX_OPERATING_MODEL_2026-04-05.md`
> - `docs/NEXT_PHASE_MASTER_PLAN_2026-04-05.md`
>
> Durum notu, 27 Nisan 2026:
> Yuksek-rigor status, paper audit, benchmark triage, roadmap, capability audit ve gap audit script'leri artik `deep` profilini ve JSON schema ciktilarini kullanir. Capability/gap audit akislari MCP'yi zorla kapatmaz.

Bu belge, bu repoda Codex'i daha etkili kullanmak icin eklenen yeni calisma modlarini ve tekrar kullanilabilir komutlari ozetler.

## 1. Eklenen Yeni Codex Ozellikleri

Bu repoda artik aktif olarak kullanilan veya devreye alinmis Codex gucleri:

- config profilleri
- nested `AGENTS.md`
- repo-local hooks
- repo-local shell rules
- non-interactive `codex exec` script'leri
- makale senkronizasyonu icin `paper-sync` skill'i
- daha genis runtime/ogrenme analizi icin `performance-debug` skill'i
- write-focused `paper_writer` subagent'i
- benchmark triage / roadmap / artifact-pack katmani

Bu eklemelerin amaci, ayni repoda arastirma, debugging ve makale yazimini ayri ama tekrar kullanilabilir akislara ayirmak.

## 2. Profiller

`/.codex/config.toml` icinde aktif profil seti:

- `paper`
  - makale ve tez duzeltmeleri
  - `gpt-5.5` + `xhigh` reasoning
  - `live` web search
  - daha noro akademik dil icin `personality = "none"`
- `review`
  - read-only audit ve teknik inceleme
  - `gpt-5.5` + `xhigh` reasoning
  - repo icinde risk arama, circuit review, status audit
- `deep`
  - artifact reconciliation, workflow audit, paper audit ve benchmark planning gibi yuksek-rigor non-interactive isler
  - `gpt-5.5` + `xhigh` reasoning
  - `live` web search
- `fast_local`
  - daha hizli yerel iterasyon
  - orta seviye reasoning
- `benchmark`
  - benchmark triage
  - result reconciliation
  - experiment planning
  - `gpt-5.5` + `xhigh` reasoning
- `colab`
  - Colab handoff
  - remote training orchestration planning

Ornek kullanim:

```bash
codex -p paper
./scripts/codex-study-status.sh
```

## 3. Hazir Script'ler

Tekrar tekrar prompt yazmamak icin aktif script katmani:

```bash
./scripts/codex-study-status.sh
./scripts/codex-paper-audit.sh
./scripts/codex-circuit-review.sh
./scripts/codex-circuit-review.sh src/model.py
./scripts/codex-capability-audit.sh
./scripts/codex-gap-audit.sh
./scripts/codex-benchmark-triage.sh
./scripts/codex-roadmap-plan.sh
./scripts/codex-colab-handoff.sh
./scripts/codex-artifact-pack.sh
./scripts/codex-model-benchmark.sh
python scripts/aggregate_benchmarks.py
```

Bu script'ler `codex exec` kullanir. 27 Nisan 2026 itibariyla yuksek-rigor script'ler
repo-local `deep` profilini kullanir ve onemli denetim ciktilarini `schemas/codex/*.schema.json`
ile makine-okunabilir hale getirir. Script-first kullanim hala onerilir; cunku dogru profil,
sandbox ve schema ayarini tek giris noktasinda toplar.

## 4. Etkili Interactive Akis

Bu repo icin onerilen interactve akis:

1. `/status`
2. gerekiyorsa `/debug-config`
3. goreve gore ilgili skill
4. buyuk read-heavy islerde `/agent`
5. uzun oturumlarda `/compact`
6. commit oncesi `/review`

Ozellikle asagidaki durumlarda bu akis yuksek deger uretir:

- birden fazla belge ve deney sonucu celisiyorsa
- paper veya thesis metninde claim guncellenecekse
- quantum katman davranisi tekrar incelenecekse

## 5. Onerilen Akislar

### 5.1 Capability ve workflow audit

1. `./scripts/codex-capability-audit.sh`
2. `./scripts/codex-gap-audit.sh`
3. gerekirse `./scripts/codex-model-benchmark.sh`

### 5.2 Bilimsel sonraki adim secimi

1. `./scripts/codex-benchmark-triage.sh`
2. `./scripts/codex-roadmap-plan.sh`

### 5.3 Colab handoff

1. `./scripts/codex-colab-handoff.sh`
2. Colab'da planlanan komut
3. sonuc donunce `./scripts/codex-study-status.sh`

### 5.4 Makale ve share paketleri

1. `./scripts/codex-paper-audit.sh`
2. `paper_writer` ile hedefli duzeltme
3. `./scripts/codex-artifact-pack.sh`

## 6. Neden Bunlar Eklendi

Bu repo klasik bir kod reposu degil. Burada ayni anda:

- model gelistirme
- deney takibi
- hata ayiklama
- Colab senkronizasyonu
- makale yazimi

yurutuluyor. Tek bir genel Codex oturumu bunlari yapsa da, profil + skill + subagent + reusable script kombinasyonu daha hizli ve daha tutarli calisiyor.

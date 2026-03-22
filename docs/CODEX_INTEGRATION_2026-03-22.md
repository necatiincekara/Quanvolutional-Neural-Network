# Codex Entegrasyonu

**Tarih:** 22 Mart 2026

Bu dokuman, bu repodaki Claude Code odakli yapilarin Codex karsiliklarini ve bu proje icin alinmis entegrasyon kararlarini ozetler.

## 1. Resmi Ozellik Eslemesi

| Claude Code ozelligi | Claude resmi yapisi | Codex karsiligi | Bu repodaki karar |
|---|---|---|---|
| Proje talimat dosyasi | `CLAUDE.md` | `AGENTS.md` | Yeni ana talimat dosyasi `AGENTS.md` oldu. `CLAUDE.md` korunuyor ama tarihsel kabul ediliyor. |
| Proje skill'leri | `.claude/skills/<name>/SKILL.md` | `.agents/skills/<name>/SKILL.md` | Mevcut Claude skill'leri Codex bicimine tasindi ve proje gercegine gore guncellendi. |
| Ozel agent/subagent tanimlari | `.claude/agents/*.md` | `.codex/agents/*.toml` | `quantum_ml_reviewer` yeniden yazildi, ek olarak sonuc mutabakati ve belge tutarliligi agent'lari eklendi. |
| Proje ayarlari / izin mantigi | `.claude/settings.local.json` | `.codex/config.toml` | Codex icin repo-scoped config eklendi. Makineye ozel trust/permission detaylari kullanici tarafinda kalmali. |
| Slash command: ajanlar | `/agents` | `/agent` | Birebir ayni degil ama amac ayni: aktif subagent thread'lerini kullanmak ve degistirmek. |
| Slash command: izinler | `/permissions` | `/permissions` | Dogrudan karsilik var. |
| Slash command: review | `/review` | `/review` | Dogrudan karsilik var. |
| Slash command: status | `/status` | `/status` | Dogrudan karsilik var. |
| Slash command: init | `/init` | `/init` | Dogrudan karsilik var; Codex tarafinda `AGENTS.md` iskelesi uretir. |
| Slash command: mcp | `/mcp` | `/mcp` | Dogrudan karsilik var. |
| Komut kurallari / approval policy | Claude permissions + project settings | `codex/rules/*.rules` + `config.toml` approval/sandbox ayarlari | Resmi destek var, fakat repo-portable olmadigi icin bu ilk entegrasyonda commitlemedim; gerektiginde ekip politikasi olarak eklenmeli. |
| Hook sistemi | Claude hooks | Net repo-local 1:1 karsilik yok | Bu repoda hooks yerine `AGENTS.md` + skills + subagents + gerekirse dis otomasyon onerildi. |
| Ozel slash command'ler | `.claude/commands/` | Codex'te ayni ana desen belgelerde vurgulanmiyor | Bu repoda `.claude/commands/` zaten yok; komut ihtiyaclari skill yapisina tasindi. |

## 2. Bu Projede Yapilan Codex Katmanlari

Eklenen yapilar:

- `AGENTS.md`
- `.codex/config.toml`
- `.codex/agents/quantum-ml-reviewer.toml`
- `.codex/agents/result-reconciler.toml`
- `.codex/agents/paper-consistency-reviewer.toml`
- `.codex/agents/paper-writer.toml`
- `.agents/skills/*`
- `scripts/codex-*.sh`
- `docs/CODEX_WORKFLOWS.md`

Yeni Codex katmani iki temel probleme gore tasarlandi:

1. Repo icindeki tarihsel belgeler ile guncel deney artefaktlari birbirini her yerde dogrulamiyor.
2. Bu calisma standart bir ML reposu degil; QML mimarisi, gradient stabilitesi, Colab egitimi ve arastirma yazimi ayni yerde yuruyor.

Bu nedenle sadece "Claude dosyalarini kopyalama" yaklasimi kullanilmadi. Bunun yerine Codex'e ozel olarak:

- gercek kaynak hiyerarsisi tanimlandi
- sonuc mutabakati icin ayri agent ve skill eklendi
- belge/paper tutarliligi icin ayri reviewer eklendi
- stale dosyalarin nasil ele alinacagi `AGENTS.md` icine yazildi
- approval/sandbox politikasi `.codex/config.toml` icinde proje icin makul varsayilanlarla ayarlandi
- daha sert komut politikalarinin ise gerekirse sonradan `codex/rules/` altinda eklenmesi tercih edildi
- `profiles.<name>.*` kullanilarak gorev bazli Codex modlari eklendi
- `codex exec` tabanli tekrar kullanilabilir script'ler olusturuldu
- `.cursor/rules` altindaki faydali ama stale olmayan mantik, yeni Codex skill'lerine secilerek tasindi

## 3. Projeye Ozel Calisma Kurallari

Codex bu repoda asagidaki sekilde kullanilmali:

- Durum raporu, README guncellemesi veya tez/paper metni yazmadan once `reconcile-results` kullan.
- Quantum circuit veya trainable quanvolution incelemesinde `review-circuit` ve gerekirse `quantum_ml_reviewer` agent'ini kullan.
- Sonuclar birbiriyle celisiyorsa once `experiments/*.json`, checkpoint'ler ve notebook ciktilarina bak; sonra anlatim belgelerine gec.
- `README.md`, `CLAUDE.md` ve `docs/AUDIT_REPORT.md` dosyalarini varsayilan gercek kaynak gibi kabul etme.

## 4. Kodlanmis Bilgi

`AGENTS.md` icine su proje-gercekleri acikca yerlestirildi:

- V7 trainable-quantum hattinin var oldugu
- Yerel ablation sonuclarinin V7 dokumante sonucundan daha iyi oldugu
- `experiments/run_experiments.py` dosyasinin stale oldugu
- `train_v7.py --target` akisinin tam baglanmadigi
- veri yukleyicinin hatali isimlendirilmis bir dosyayi atladigi

Bu bilgiler, gelecekteki Codex oturumlarinin ayni eski varsayimlara geri donmesini engellemek icin ozellikle eklendi.

## 5. Nasil Kullanilir

Ornek prompt'lar:

- `Bu calismanin guncel durumunu Turkce raporla. reconcile-results kullan.`
- `src/trainable_quantum_model.py icindeki circuit yapisini review-circuit ile incele.`
- `README ve paper icindeki stale claim'leri bul. paper_consistency_reviewer kullan.`
- `Yeni bir ablation planla ve experiment skill'i ile yurutu.` 

Yararli Codex komutlari:

- `/status`
- `/debug-config`
- `/permissions`
- `/review`
- `/agent`

## 6. Hook Konusu

Claude Code hooks resmi olarak mevcut. 22 Mart 2026 itibariyla Codex belgelerinde buna tam repo-local birebir denk dusen, ayni gelistirici akisina sahip bir sistem bu entegrasyon icin temel arac olarak dokumante edilmiyor. Bu repoda hook davranislari yerine su katmanlar tercih edildi:

- `AGENTS.md` ile kalici repo talimatlari
- `.agents/skills` ile gorev odakli davranis paketleri
- `.codex/agents` ile dar ve uzman subagent'ler
- gerektiginde harici script veya CI otomasyonu

## 7. Resmi Kaynaklar

Anthropic:

- Claude Code slash commands: `https://docs.anthropic.com/en/docs/claude-code/slash-commands`
- Claude Code settings: `https://docs.anthropic.com/en/docs/claude-code/settings`
- Claude Code skills: `https://docs.anthropic.com/en/docs/claude-code/skills`
- Claude Code hooks: `https://docs.anthropic.com/en/docs/claude-code/hooks`

OpenAI:

- Codex overview: `https://developers.openai.com/codex/overview`
- Codex `AGENTS.md`: `https://developers.openai.com/codex/guides/agents-md`
- Codex skills: `https://developers.openai.com/codex/skills`
- Codex subagents: `https://developers.openai.com/codex/subagents`
- Codex slash commands: `https://developers.openai.com/codex/cli/slash-commands`
- Codex config reference: `https://developers.openai.com/codex/config-reference`
- Codex rules: `https://developers.openai.com/codex/rules`

# Game L10n Engine (L10nShield)

İkili (binary) oyun arşiv dosyalarını ayrıştıran, çoklu LLM sağlayıcıları (Gemini, Claude, Ollama) ile bağlamsal ve toplu (batch) olarak yerelleştiren, ardından tekrar oyun motoruna uygun formatta paketleyen modüler çeviri platformu.

## Özellikler

- **Çoklu Format & Binary Ayrıştırma (Parsers):** Rebellion Asura Engine (`.asr`), Capcom Archive (`.arc`) ve Valve KeyValues (`.vdf`) dosya yapılarını okuma, metin bloklarını ayıklama ve yeniden paketleme (repack).
- **Çoklu LLM Sağlayıcı Desteği:** Google Gemini, Anthropic Claude ve yerel Ollama modelleriyle esnek entegrasyon.
- **Regex Kalkanı (Regex Shield):** Oyun içi renk kodları, değişkenler (`%s`, `{0}`) ve kontrol karakterlerinin (`\n`, `\r`) çeviri sırasında bozulmasını önleyen koruma katmanı.
- **Akıllı Paketleme & Maliyet Tahmini:** Kota sınırlarına (rate limit) takılmadan bağlamsal gruplama (`context_batcher`) ve çeviri öncesi token/maliyet hesaplama (`cost_estimator`).
- **Sözlük (Glossary) ve Kalite Kontrol:** Özel terim tutarlılığı için `glossary.json` entegrasyonu ve çeviri sonrası doğrulama (`sanity_checker`).
- **Stüdyo Arayüzü (GUI):** Dosya analizi, API anahtarı yönetimi ve anlık çeviri takibi sağlayan masaüstü arayüzü.

## Kurulum

1. Depoyu klonlayın ve proje dizinine gidin:
   git clone https://github.com/0xWandererx0/game-l10n-engine.git
   cd game-l10n-engine

2. Gerekli Python kütüphanelerini yükleyin:
   pip install -r requirements.txt

3. `.env.example` dosyasının adını `.env` olarak değiştirip kendi API anahtarlarınızı girin (veya doğrudan uygulama arayüzü üzerinden tanımlayın).

## Kullanım

Grafik arayüzü (GUI) başlatmak için:
python main.py

Birim testlerini çalıştırmak için:
pytest
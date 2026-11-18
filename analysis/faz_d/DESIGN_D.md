# Faz D — Bütün CNN-GRU koşularını tek GPU'ya (RTX 3090) almak

**Yazıldı:** 2026-11-05 akşam, hiçbir Faz D koşusundan önce. Bu plan, sonuçlar görülmeden yazıldı. Sonradan yapılan her değişiklik aşağıdaki "Sapmalar" bölümüne tarihli olarak eklenecek.

**Statü:** İkincil (Faz A2/A4'ün tekrarı). Ana model seçimini (B2 + D3) değiştirmez. Etiket: [S].

## Gerekçe
Kullanıcı kararı (2026-11-05 akşam): makaledeki CNN-GRU koşuları iki ayrı GPU'da yapılmış olmasın. Bütün koşular RTX 3090'da olacak ve iki uygulama (PyTorch portu ve ilk Keras mimarisinin yeniden eğitimi) makalede kalacak.

Şu anki dağılım:
- **RTX 3090:** Faz A S1 katları (PyTorch, birincil), PyTorch S0 tohum 42, bütün Keras koşuları (S1 katları, S0 tohum 42).
- **GTX 1650 SUPER:** PyTorch S0 tohum 1–4 (A2.2, `faz_a2/12_cnn_s0_seeds.py`) ve A4.3a'daki S1 katlarının yeniden eğitimi (`faz_a4/13_cnn_s1_local.py`).

Tek GPU için gereken tek koşu: **PyTorch CNN-GRU, S0 tohum 1–4, RTX 3090'da.** A4.3a'nın GTX S1 yeniden eğitimi yalnız GTX'teki S0 tohumlarıyla eşleşsin diye yapılmıştı. Artık gereksiz olduğu için tekrarlanmayacak.

## Koşu (D.1)
- **Betik:** `faz_d/01_cnn_s0_seeds_3090.py`. `faz_a2/12_cnn_s0_seeds.py` modülünü olduğu gibi içe aktarır: aynı mimari, eğitim, erken durdurma, iç doğrulama ve tohum şeması (bölme `random_state` = tohum, iç doğrulama sınıf-tabakalı %10 `random_state` = tohum, model tohumu 20261104 + tohum).
- Yalnız çıktı yolları değişir. Dosyalar `reports/faz_d/01_cnn_s0_runs.csv` ve `reports/faz_d/oof/cnngru__S0s{tohum}__f0.csv.gz` altına yazılır. A2 dosyalarına dokunulmaz.
- **Ortam:** Uzak makinedeki `~/mmwave-revizyon/.venv` (PyTorch 2.6.0+cu124), `tools/remote_start.sh` ile ayrık iş olarak. Cihaz CUDA (RTX 3090) olmalı; CPU'ya düşerse koşu geçersiz sayılır ve tekrarlanır.
- **GPU paylaşımı:** Enerji oturumunun (enerji-90) onayı alınmadan başlatılmaz.

## Kapılar (koşudan sonra, analizden önce)
1. Tohum 42'nin S0 bölmesi Faz A ile aynı olmalı (modülün kendi assert'i).
2. Her tohumda test pencereleri ((oturum, frame_id) kümesi) ve `y_true`, A2.2'nin GTX OOF dosyalarındakilerle birebir aynı olmalı.
3. `device` alanı `cuda` olmalı ve iş günlüğünde "RTX 3090" görünmeli.

Bir kapı geçmezse analiz yapılmaz ve durum raporlanır.

## Analiz (D.2) — `faz_d/02_leakage_3090.py`
- A2.2 sızıntı tablosu (`faz_a2/02_leakage.py` ile aynı metrikler ve aynı Welch yöntemi) yeniden hesaplanır.
  - B1/B2 satırları A2.2'den değişmeden alınır; bunlar CPU'da deterministik.
  - CNN-GRU S0 = tohum 42 (Faz A, 3090) + yeni 3090 tohumları 1–4.
  - CNN-GRU S1 = Faz A katları (3090).
- Çıktılar A2.2 ile aynı biçimde: `reports/faz_d/02_leakage_per_run.csv`, `02_leakage_summary.csv`, `02_leakage_3090.md`.
- **Okuma kuralı (koşudan önce sabitlendi):**
  - Makalenin sızıntı başlık değeri, PyTorch portu için bu tamamen-3090 beş tohumlu Welch tahmini olur: doğruluk ve macro-F1, %95 GA ile.
  - Keras'ın tek koşuluk tahmini (3090, S0 tohum 42 − S1 kat ortalaması, A4.3b'den değişmeden) yanında verilir.
  - GTX 1650 koşuları (A2.2 tohum 1–4, A4.3a S1) makalenin tablo ve şekillerinde kullanılmaz. Ek dosyadaki plan bölümüne, iki GPU'lu ilk karşılaştırmanın bu fazla yenilendiğini söyleyen tek bir cümle eklenir.
  - **A4 kuralından sapma:** A4'ün "başlık değeri iki aynı-donanım tahmininden küçük olanı" kuralı, iki ayrı donanımdan gelen iki tahmin için yazılmıştı. Tek donanımda tek bir beş tohumlu tahmin kalıyor, bu yüzden o kural bu fazda uygulanmaz. Bu sapma kullanıcı kararından kaynaklanıyor ve sonuçlar görülmeden yazıldı.
- **Önceden yazılan beklentiler (yalnız betimsel; karar kuralına girmez):**
  - (a) 3090'daki S0 tohum 1–4 doğrulukları %89,5–93,0 aralığında kalacak. GTX'te %91,4–92,2, 3090 tohum 42'de %90,1 idi.
  - (b) Tamamen-3090 beş tohumlu doğruluk etkisi +2 ile +6 puan arasında olacak. A2.2'nin karışık tahmini +4,4 idi.

## Makalede değişecek yerler (sonuçtan sonra)
- Özet ve katkı 3'teki sızıntı sayıları.
- Bulgular metni: §6.2 (sızıntı paragrafları) ve §Discussion.
- Tablo 3: `make_tables_a4.py` `tab_leak`; GTX satırı çıkar, PyTorch S0 sütunu beş tohum olur.
- Tablo S-keras folds: GTX sütunu çıkar.
- Şekil 3 (`fig3_ranking.py`, faz_d çıktılarıyla) ve başlığındaki 4,4 / 5,6.
- Ek dosyanın sızıntı ayrıntıları ve ortam beyanı (cuDNN satırı).
- Kapak mektubundaki sızıntı cümlesi.

## Kapsam dışı
- Keras S0 tohum 1–4: Keras zaten tümüyle 3090'da; tek GPU için gerekmiyor.
- CNN-only ve anomali hattı: bunlar zaten 3090'da ya da CPU'da.

## Sapmalar ve uygulama notları
- 2026-11-05 ~18:15 (koşudan önce):
  - `tools/remote_start.sh` ve `tools/remote_run.sh` içindeki gönderim listesine `faz_c` ve `faz_d` eklendi (tek satır). Aksi halde Faz D betikleri uzak kopyaya gitmiyordu.
  - D.1 sarmalayıcısı yerel GTX 1650 SUPER'de `--smoke --seeds 1` (1 epoch) ile denendi. Çıktılar yalnız `CACHE_DIR/smoke_d1`'e yazıldı; A2 dosyaları değişmedi.
  - D.2'nin yöntem kapısı K3 (`--source a2`) geçti: max |fark| 8,9e-16.
  - Koşu, enerji-90'ın G1-MT koşusu bitene kadar bekliyor (yaklaşık TR 00:30–01:30). O oturumun isteği: o zamana kadar 3090'a iş gönderilmeyecek.
- 2026-11-05 23:52 TR (20:52 UTC): enerji-90, G1-MT koşusunun 20:49 UTC'de bittiğini ve GPU'nun boş olduğunu bildirdi (`nvidia-smi`: 1 MiB, süreç yok; doğrulandı). D.1 başlatıldı: `job-20261105-235202-37553-14034` (`--seeds 1,2,3,4 --require-gpu 3090`). Plan ve okuma kuralı değişmedi.
- 2026-11-06 00:03 TR: D.1 bitti (çıkış 0; tohum başına 142–175 s; env: torch 2.6.0+cu124, CUDA 12.4, cuDNN 90100, RTX 3090). D.2 `--source d` koşuldu: K1, K2, K4 geçti (K3 önceden geçmişti). Sonuç: doğruluk +4,04 [+2,44, +5,63], macro-F1 +5,21 [+3,34, +7,07]; beklenti (a) ve (b) tuttu. GPU enerji-90'a geri bildirildi.
- 2026-11-06: Makale okuma kuralına göre güncellendi (`make_tables_a4.py` tab_leak + tab_keras_folds, `fig3_ranking.py` LEAK_DIR=faz_d, main/supplementary/kapak mektubu metinleri). Ek dosyada GTX karşılaştırmalarının sonuç paragrafı ("Matched hardware (Phase A4)") ve tanım satırı çıkarıldı; yerine Faz D paragrafı ve A4.3a kuralının geçersiz kaldığını söyleyen cümle kondu. Ön-düzenleme kopyaları: `paper/archive/*_2026-11-06_preD*`.

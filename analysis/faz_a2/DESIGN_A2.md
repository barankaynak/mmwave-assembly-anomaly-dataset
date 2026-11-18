# Faz A2 — İkincil analizler (ÖN-KAYIT, 2026-11-04)

## Statü (değiştirilemez)
- Faz A sonuçları **görüldü**. Bu yüzden A2'deki her şey Faz A'ya göre **sonradan**dır.
- **Makalenin birincil sonuçları Faz A'nın ön-kayıtlı sayılarıdır** (`reports/faz_a/FAZ_A_RESULTS.md`). A2 bunların yerine geçmez; makale A2 sayılarıyla açılmaz. A2 sonuçları daha iyi çıksa da durum değişmez.
- A2, aşağıda sabitlenen **ikincil** analizlerdir. Her biri adı konmuş bir Faz A gözlemine cevap verir ve makalede tek bir cümleyi destekler.
- **Hiçbir A2 sonucu ana model seçimini değiştirmez** (B2 + D3). Tanımlanan bütün sayılar raporlanır; seçici raporlama yok.
- Yeni veri toplanmaz. Faz A dosyalarının (OOF, fold CSV'leri, raporlar) **üzerine yazılmaz**; A2 çıktıları `reports/faz_a2/` altına gider.
- Sapmalar en altta tarihli olarak tutulur.

## Ortam
- Koşular **yerel makinede**: AMD Ryzen 7 3700X (8 çekirdek / 16 iş parçacığı), 32 GB RAM, GTX 1650 SUPER 4 GB.
- Venv `<proje venv>`: Python 3.12, `requirements.txt` ile aynı sürümler, torch 2.6.0+cu124.
- Bütün betikler `tools/local_run.sh` ile koşar: CACHE_DIR veri klasörünün dışında, OMP_NUM_THREADS ≤ 6.
- Veri: klasör 4'teki yerel Zenodo kopyası (uzak kopyayla md5 aynı). Faz A ise uzak makinede (RTX 3090) koşmuştu.

## A2.0 — Yerel = uzak boru hattı denkliği (önce bu)
- Yerelde heatmap önbelleğini yeniden kur. Orijinal `build_heatmap` döngüsüyle karşılaştır: 600 rastgele karede max |fark| = 0 olmalı.
- Kare / pencere / döngü sayıları Faz A ile aynı olmalı: 16.272 kare / 16.248 pencere; fold CSV'leri bayt bayt aynı.
- **B2 S1 kat 0 ve kat 1'i yerelde yeniden eğit.** Tahminler `reports/faz_a/oof/b2__S1__f0/f1` ile karşılaştırılır.
  - Beklenen: özdeş (RF sabit tohumlu ve sürümler aynı). Fark çıkarsa oran raporlanır ve A2 yine sürer.
- Çıktı: `reports/faz_a2/00_equivalence.md`. **Başarısızsa A2 durur ve kullanıcıya bildirilir.**

## A2.1 — Conformal eşik: eşitlikler ve kalibrasyon çözünürlüğü
**Gerekçe (Faz A gözlemi):**
- D1 skorlarında eşitlik çok; bu yüzden FAR = 0 çıkıyor (muhafazakâr).
- n_cal ≈ 20'de en küçük p ≈ 0,048. D3 α = 0,10'da yalnız uç döngülerde, α = 0,05'te hiç alarm veremiyor.
- Sıra tespiti %25–48'de kalıyor.

**Sabit kurallar:**
- **Kombinasyon değişmiyor.** Birincil kural D3 = min(1, 2·min(p1, p2)), yani Bonferroni: alarm = "D1 ya da D2, her biri α/2'de".
  - D1 ve D2 ayrıca tek tek raporlanır (ikincil).
  - Betimleyici olarak "D1 ya da D2, her biri α'da" FAR'ı da verilir (nominal ≤ 2α). Bu bir çalışma noktası olarak sunulmaz.
- **Düzeltme 1 — düzgünleştirilmiş (randomize) p:** p = (#{cal > s} + U·(#{cal = s} + 1)) / (n_cal + 1), U ~ Unif(0, 1). Sabit tohumla 200 çekiliş; FAR ve tespit için beklenen değer ve %2,5–97,5 aralığı raporlanır.
- **Düzeltme 2 — cross-conformal (s02 normal döngüleri):**
  - Test katı f, Faz A'daki `folds_anom.csv` katlarıyla aynı.
  - Kalan 4 katın her biri k için: parametreler diğer 3 kattan (~31 döngü) öğrenilir. k katı skorlanır (kat dışı kalibrasyon skoru), test ve anomali döngüleri de model k ile skorlanır.
  - p = (1 + Σ_k #{cal_k ≥ s^(k)}) / (n_cal + 1), n_cal ≈ 41. Eşitliklerde düzeltme 1 de uygulanır.
  - Bilinen uyarı: teorik sınır 2α, ampirik olarak ≈ α. Gözlenen FAR raporlanır.
  - Not: parametre öğrenme seti Faz A'dakinden büyük (~31 vs ~20); bu bir karıştırıcıdır, açıkça yazılır.
- **Hücreler:** B2 × pred (dağıtılabilir, ana), B2 × tof, ToF oracle × tof, CNN-GRU × pred. Yalnız mevcut S1 OOF tahminleri kullanılır, yeniden eğitim yok.
- **Metrikler:** α ∈ {0,10; 0,05}'te FAR (SE ile) ve tür başına tespit; AUROC (Faz A ile aynı skorlar; değişmesi beklenmez).
- Çıktı: `reports/faz_a2/01_conformal.md`.
- **Desteklediği makale cümlesi (ikincil):** "Eşitliğe duyarlı cross-conformal kalibrasyonla (~40 normal döngü) iki kanallı dedektör, döngü düzeyinde %Z yanlış alarmda sıra sapmalarının %X'ini, tempo sapmalarının %Y'sini yakalıyor."

## A2.2 — Sızıntı bir örüntü mü? (port etkisinden arınmış)
**Gerekçe:** S0 tek koşu. Sızıntı etkisinin yalnız CNN-GRU'ya mı özgü olduğu bilinmiyor.
- S0 = sınıf-tabakalı rastgele %80/20 pencere bölmesi (Faz A tanımı), **5 tohum**: 42 (Faz A) ve 1, 2, 3, 4.
- Modeller: B1, B2 (5 tohum), CNN-GRU (tohum 42 Faz A'dan, 1–4 yeni; yerel GPU).
- OOF adı: `{model}__S0s{seed}__f0.csv.gz`. Faz A dosyası `cnngru__S0__f0` aynen kalır.
- **Metrik:** model başına sızıntı etkisi = ortalama(S0) − ortalama(S1 kat). Tohumlar ve katlar üzerinden %95 GA, accuracy ve macro-F1 için.
- Not: CNN-GRU'nun S1 sayıları uzak 3090'da, yeni S0 tohumları yerel 1650S'te koştu; GPU belirlenimsizliği not edilir.
- Çıktı: `reports/faz_a2/02_leakage.md`.
- **Desteklediği cümle:** "Rastgele pencere bölmesi hem derin hem sığ modelde doğruluğu X ± Y puan şişiriyor."

## A2.3 — Etiket duyarlılığı (s05'te silinen Box5)
**Gerekçe:** ±3 kare mod yumuşatması, 49 hızlı döngünün 40'ında Box5'i siliyor.
- **Yeniden eğitim yok.** Mevcut S1 OOF tahminleri (B2, CNN-GRU, CNN-only, B1) alternatif doğrulara karşı yeniden skorlanır. Pencere etiketi = son karenin etiketi.
  - L0: Faz A `y_true` (±3 mod);
  - L1: ham 10 Hz etiket (yumuşatma yok);
  - L2: ±1 kare mod;
  - L3: 30 Hz ham ToF'tan "dokunuş" etiketi. Karenin (t − 100 ms, t] aralığındaki ToF örneklerinde mesafe < 140 mm olan kutu; birden fazlaysa en çok örnek alan, eşitlikte en son.
- Oturum başına accuracy / macro-F1; oturum × etiket başına "en az bir Box5 karesi olan döngü" oranı.
- Uyarı: eğitim etiketleri de aynı silmeyi taşıyordu. Bu analiz yalnız **değerlendirme tarafındaki** artefaktı ölçer.
- Çıktı: `reports/faz_a2/03_labels.md`.
- **Desteklediği cümle:** "s05'teki düşük skor kısmen etiket artefaktı: 30 Hz dokunuş etiketiyle B2'nin s05 macro-F1'i A'dan B'ye çıkıyor (ya da çıkmıyor)."

## A2.4 — Devreye alma maliyeti
**(a) Anomali:**
- Her tekrarda (R = 50, sabit tohum) s02'nin 51 normal döngüsü rastgele dağıtılır: test 11, kalibrasyon 20, eğitim havuzu 20.
- Eğitim için havuzdan ilk n döngü alınır, n ∈ {5, 10, 15, 20}.
- Kaynak: B2 × pred ve ToF oracle × tof. Mevcut S1 OOF tahminleri kullanılır.
- **Metrikler:** D3 AUROC (tür başına ve tümü) ve α = 0,10'da FAR; tekrarlar üzerinden ortalama ± sd.
- Not: döngüler zaman içinde bağımsız değil (sürüklenme). Rastgele dağıtım bu yüzden iyimser olabilir; yazılır.

**(b) Durum tahmini:**
- S1 katlarında, eğitim havuzundan **oturum başına k etiketli döngü** alınır, k ∈ {1, 2, 5, 10, 20, hepsi}; R = 10 tekrar.
- B2 ve B1 eğitilir; kat f'de test edilir.
- **Metrik:** macro-F1 ortalama ± sd. k, kayıt süresine çevrilir (k × ortalama döngü süresi). ToF etiketi bedava olduğu için maliyet = kayıt dakikası.

- Çıktı: `reports/faz_a2/04_commissioning.md`.
- **Desteklediği cümle:** "Yeni bir istasyonu devreye almak için N normal döngü (~M dakika) ve k etiketli döngü yetiyor."

## A2.5 — CPU gecikmesi
- Tek iş parçacığı (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`, RF `n_jobs=1`), yerel Ryzen 7 3700X.
- Ölçülenler (pencere başına; 50 ısınma + 1000 ölçüm; medyan ve IQR):
  - B1;
  - B2 (özellik çıkarımı + RF tahmini);
  - CNN-GRU (PyTorch, batch 1);
  - ayrıca kare başına heatmap kurulumu.
- Eski makalenin 12,6 ms'si farklı bir makinede (Ryzen 7 5800X) ve Keras ile ölçülmüştü; açıkça yazılır.
- Çıktı: `reports/faz_a2/05_latency.md`.
- **Desteklediği cümle:** "Masaüstü CPU'da pencere başına X ms; 100 ms kare aralığının çok altında."

## Kapsam dışı
Sentetik anomali enjeksiyonu (zaman kalırsa ayrı ön-kayıtla), yeni veri, ana modelin değiştirilmesi, ZDA/SigPath.

## Sapmalar (tarihli)

- 2026-11-04 (A2.4b): B1 icin tau, Faz A gibi grup bazli ic dogrulamayla secilir; oturum basina k=1 dongude ic dogrulama kumesi bos kaldigindan bu durumda tau egitim pencerelerinin kendisinde (resubstitution) secilir. Ayrica A2.4a pred hucresinde "ilk n dongu" = havuzdaki ToF dongulerine dusen tahmin dongulerinin ilk n tanesi (n dongu sayisi; dagitim ToF dongusu bazinda). Sebep: brif k=1 icin tau protokolunu ve pred hucresinde dongu/ToF-dongusu eslesmesini tanimlamiyor.
- 2026-11-04 (A2.3): Brief'e ek, ikincil betimleyici kontroller eklendi (sonuc secimini degistirmez): L3b (frame_id tabanli ToF atamasi), macro-F1'in sinif-5-harici versiyonu, dongu-bazli paired bootstrap GA, s05-disi oturumlar (EX5) ve fark-farki (DiD). Brief'teki L0-L3 tanimlari aynen uygulandi; L3 'en cok ornek' kutu basina mesafe<140 mm ornek sayisi olarak yorumlandi.
- 2026-11-04 (A2.2): Brif tohum disindaki RNG'leri belirtmiyordu; sabitlendi: S0 ic dogrulamasi = egitim pencerelerinden sinif-tabakali rastgele %10 (random_state = bolme tohumu; CNN S0 dali gibi, B1 tau icin de), model tohumu (torch/RF) = 20261104 + tohum. CNN-GRU tohum 42 Faz A dosyasindan aynen (model tohumu 20261104). B1/B2 S0 Faz A'da kosulmamisti; bes tohum da yeni kosuldu. Yerel GPU'da OOM olmadi (CPU'ya dusulmedi).
- 2026-11-04 (A2.1 duzeltme turu): 01_conformal.md Yorum madde 2'de "B2 x tof -0,002" -> "-0,003" (dogrusu 0.94161 - 0.94423 = -0.00263). Yorum bolumu elle yazilir (script MARK sonrasini korur); script ve sayisal tablolar degismedi, yeniden kosu gerekmedi.

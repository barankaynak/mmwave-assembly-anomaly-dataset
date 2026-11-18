# Faz A — Değerlendirme protokolü + taban çizgisi kıyası (ÖN-KAYIT, 2026-11-04)

Bu dosya, sonuçlar görülmeden önce sabitlenen tasarımdır. Değişiklik gerekirse aşağıdaki "Sapmalar" bölümüne tarih ve gerekçeyle yazılır; buradaki metin geriye dönük olarak değiştirilmez.

## Amaç
1. **Durum tahmini** (kare → {none, Box1..5}): CNN-GRU'yu **sızıntısız** ölçmek ve aynı bölmelerde basit taban çizgileriyle kıyaslamak.
2. **Döngü düzeyi anomali tespiti:** üç yaklaşımı aynı protokolde karşılaştırmak:
   - eski boru hattı: CNN-GRU → TFT |Δ| (Denk. 9) + Z-score, |Z| > 1;
   - düzeltilmiş skorlar: Markov −log p, adım bazlı süre, conformal eşik;
   - en basit taban: döngü süresi.
3. Bu sayılar yeni makalenin Sonuçlar bölümünün omurgası olacak.

## Ortak tanımlar
- **Etiket:** ToF < 140 mm (= `box*_active`, %100 aynı) + **±3 kare mod yumuşatma** (`run_all_pt.py derive_labels` ile aynı, oturum içinde). Ham etiket de saklanır.
- **Nokta kaynağı:** `merged_df.csv` (v ≠ 0), eski boru hattıyla aynı.
- **Heatmap:** 48×48×2; X ∈ [−0,5; 0,5], Y ∈ [0,1; 1,0]. Kanal 0 = hücredeki en büyük SNR, kanal 1 = ortalama hız (`run_all_pt.py build_heatmap` ile birebir). Önbellek: `$CACHE_DIR`.
- **Pencere:** L = 5 kare (0,5 s); etiket = son karenin etiketi; pencere oturum sınırını aşmaz → 16.248 pencere (16.272 − 6×4).
- **Grup (döngü):** `(session, loop_id)`. loop 0 parçaları kendi grubunu oluşturur.

## Bölmeler
- **S1 (ana): 5 katlı `StratifiedGroupKFold`.** 6 oturumun havuzunda; grup = döngü, strata = oturum (her katta her oturumdan döngü olur); seed 20261104.
  - Her model için OOF (out-of-fold) tahmin üretilir: her pencere tam bir kez testte.
  - Model seçimi ve erken durdurma, eğitim katlarının içinden **döngü bazlı %10 iç doğrulama** ile yapılır. Test katına hiçbir şekilde bakılmaz.
- **S0 (sızıntı gösterimi):** eski makaledeki gibi rastgele, sınıfa göre tabakalı %80/20 **pencere** bölmesi (random_state 42). Yalnız CNN-GRU, tek koşu. Eski %93,66 ile kıyas için.
- **S2 (ikincil): leave-one-session-out (6 kat).** Durum tahmininde oturumlar arası genelleme (Box2 kayması). CNN-GRU + tabanlar; hesaplama kısıtlıysa CNN-only S2'de atlanır.
- **Maruz kalma notu:** havuzlu S1 tahmincisi s03–s06'nın **karelerini** (yalnız durum etiketleri) görür. Anomali dedektörü anomali döngülerini **hiç** görmez. "Anomali verisini hiç görmedi" sorusunu S2 (LOSO) cevaplar.
- **Anomali bölmesi: s02 normal döngülerinde 5 katlı çapraz uyum (cross-fitting).**
  - Döngü bazlı `GroupKFold` (karıştırılmış, aralıklı; Q3'te görülen oturum içi sürüklenme yüzünden ardışık bloklar kullanılmaz).
  - Her f için: test = kat f, **kalibrasyon = katlar (f+1) ve (f+2) mod 5 (n_cal ≈ 20)**, eğitim = kalan 2 kat.
    - Gerekçe: n_cal ≈ 10 olsaydı en küçük p = 1/11 ≈ 0,09 olurdu ve α = 0,05'te hiçbir döngü alarm veremezdi.
  - s03–s06'nın tüm döngüleri her f'de test olur.
  - **FAR ve tespit oranı:** 5 f'nin test döngüleri havuzlanarak hesaplanır (her normal döngü tam bir kez test; ≈ 50 döngü).
  - **AUROC/AUPRC:** f başına hesaplanıp ortalama ± sd verilir.

## Modeller — durum tahmini (hepsi S1 ve S2'de aynı katlarla)
- **M1 CNN-GRU:** eski mimari, PyTorch portu (`run_all_pt.py CNNGRU`).
  - Eğitim: Adam lr 5e-4, ExponentialDecay (8000 adım, 0,92); batch 32; en fazla 120 epoch; val_loss'ta sabır 25, en iyi ağırlığa dönüş; eğitim katında class_weight = balanced.
  - Keras'taki GRU dropout 0,3 / recurrent 0,2 → PyTorch'ta GRU çıkışına dropout 0,3 (fark not edilir).
- **M2 CNN-only:** ablasyon; yalnız son kare.
- **B1 En-yakın-merkez** (derin öğrenme yok):
  - Son karenin SNR-ağırlıklı merkezi (cx, cy) kullanılır; eğitim katında her kutu için bir merkez hesaplanır.
  - En yakın kutu merkezine uzaklık > τ ise ya da karede nokta yoksa → none.
  - τ, iç doğrulamada macro-F1'i en büyük yapan değer seçilir.
- **B2 El yapımı özellik + RandomForest:**
  - Pencerenin 5 karesinden özellikler: cx, cy, n_points, snr_sum, ortalama |v|, x ve y std; son kare değerleri + pencere ortalaması + eğimi.
  - RF 300 ağaç, class_weight = balanced, sabit hiperparametre (ayar yok).

## Metrikler — durum tahmini
- OOF tahmin üzerinden: accuracy, macro-F1, sınıf başına F1; katlar arası ortalama ± sd.
- **Alt kümeler:** (a) kutu kareleri, (b) none kareleri, (c) geçiş kareleri (etiket değişimine ±2 kare), (d) oturum başına, özellikle **s01** (kutular çözünürlük altında).
- **Sızıntı etkisi = S0_pt − S1_pt** (ikisi de PyTorch portu). Keras %93,66 yanında raporlanır, port farkları listelenir: Conv→BN→ReLU sırası, GRU dropout yeri, kayıtlı 89,4 vs 93,7 eşit-epoch farkı.
- Parametre sayısı ve CPU'da pencere başına çıkarım süresi.

## Anomali tespiti — döngü düzeyi
- **Girdi:** her durum tahmincisinin (M1, B1, B2) OOF kare dizisi + **ToF oracle** (üst sınır).
- **Ziyaret dizisi:** run-length; debounce: **kutu run'ı ≥ 2 kare (0,2 s)**, aynı kutu arasındaki ≤ 1 karelik none birleştirilir → ziyaret dizisi + süreler. (Q2: Fast'te bile medyan kalış ≥ 0,29 s.)
- **Döngü segmentasyonu — 2×2 tasarım:** segmentasyon ∈ {**tahmin edilen Box1 girişleri** (dağıtım sayısı; ToF yok), **ToF döngü sınırları** (durum tahmini hatasını yalıtır)} × durum kaynağı ∈ {M1, B1, B2, ToF}. Her hücre için döngü sayıları tabloda raporlanır; hücreler arasında büyük fark varsa bu da bir bulgudur.
  - **Kat eşlemesi:** tahmini bir döngü, Box1-giriş karesini içeren ToF döngüsünün katına aittir (eğitim/kalibrasyon/test).
  - Her tahmini döngü kendi oturumunun etiketini alır: s02 normal, s03/s04 sıra anomalisi, s05/s06 tempo anomalisi.
  - İlk ve son kısmi döngüler atılır.
  - s02'de ToF dizisi `1-2-5` olmayan doğal sapmalar işaretlenir; FAR bunlar dahil **ve** hariç raporlanır.
- **Skorlar** (parametreler yalnız s02 eğitim döngülerinden):
  - **D0 (en basit taban, tempo):** log döngü süresinin |z|'si.
  - **D1 (sıra, 1. derece Markov):** durumlar {1..5, END}, Laplace α = 1; döngü skoru = max −log p(s_t | s_{t−1}).
  - **D2 (adım bazlı zamanlama):** adım = kutu kalışı (i) ve geçiş (i→j); her adım için log d ~ N(μ, σ). Döngü skoru = max |z|. Referansı olmayan adım atlanır; onu sıra kanalı yakalar.
  - **D3 (birleşik):** D1 ve D2'nin conformal p-değerlerinden Bonferroni min-p (2 × min p).
  - **OLD (eski boru hattı):**
    - TFT: eski hiperparametreler; regresyon hedefi = kutu numarası; encoder 24, hidden 32, 2 head, dropout 0,1, lr 1e-3, en fazla 50 epoch, sabır 10. s02 eğitim döngülerinin durum-değişim dizisiyle eğitilir.
    - **Erken durdurma yalnız eğitim döngülerinin iç bölmesiyle** yapılır (eski val = test hatasının tam düzeltmesi).
    - Veri gerçeği: ~20 eğitim döngüsü ≈ 60 geçiş − 24'lük encoder ≈ 36 eğitim örneği. TFT veri açlığı çekecek; bu bir hata değil, raporlanacak bir sonuçtur.
    - Çıkarımda encoder geçmişinin komşu katların döngülerine taşması **gözlenen bağlamdır, etiket sızıntısı değildir**.
    - Geçiş skoru = |gerçek − tahmin| (Denk. 9); döngü skoru = max.
    - Süre: `prev_box` başına Z (eğitim istatistikleri); döngü skoru = max |Z|.
    - **Eski sabit eşikler (0,5 / 1,0) ile** ve **conformal eşikle** ayrı ayrı raporlanır.
- **Eşik (conformal):** p = (1 + #{cal ≥ s}) / (n_cal + 1); alarm p ≤ α. **Birincil çalışma noktası α = 0,10**, ikincil α = 0,05 (n_cal ≈ 20'de en küçük p ≈ 0,048).
- **Metrikler** (normal test döngüleri vs her anomali türü, döngü düzeyi):
  - AUROC ve AUPRC;
  - α'da gözlenen FAR (normal test);
  - tür başına tespit oranı;
  - nominal α vs gözlenen FAR.
  - Ayrıca eski "% geçiş eşik üstü" metriği aynı veride yeniden hesaplanır (29,3 ile kıyas).

## Önceden sabitlenen karar ölçütleri
- **"Basit taban yeterli" hükmü:** iki şart birlikte sağlanırsa CNN-GRU makalenin ana modeli olarak gerekçesizdir; ana model basit / yapı-bilinçli taban olur.
  1. B1 veya B2'nin S1 macro-F1'i CNN-GRU'nunkine **2 puan** yakın (kat sd'si yanında raporlanır);
  2. aynı tabanla elde edilen D3 döngü-AUROC'u ile CNN-GRU tabanlı D3 arasındaki **kat bazlı eşli farkın %95 GA'sı 0'ı içeriyor ya da |fark| ≤ 0,05** (her anomali türünde; ~50/50 döngüde AUROC SE ≈ 0,03–0,05 olduğu için 0,02 çözülemez).
- Aksi halde CNN-GRU'nun kazandığı alt küme (s01 / none / geçiş) raporlanır ve CNN-GRU ana model kalır.
- **Conformal geçerlilik:** normal testte gözlenen FAR, α ± 2·SE içinde olmalı. SE = √(α(1−α)/n_test), 5 f üzerinden.

## Kapsam dışı (Faz A2)
Sentetik anomali enjeksiyonu, sağlamlık simülasyonu, tam ZDA, SigPath, ESP32 portu.

## Sapmalar (tarihli)
- 2026-11-04, **hiçbir koşudan önce** gözden geçirildi: conformal parçalılık (kal = 2 kat, α = 0,10 birincil, havuzlu FAR); 2×2 segmentasyon + kat eşlemesi; AUROC payı 0,05 / eşli GA; S0_pt referansı; debounce ≥ 2 kare; TFT iç bölmesi ve veri notu; maruz kalma notu.
- 2026-11-04, 02 ilk kosusunda B1 tau'nun izgara tabaninda (0,02 m) secildigi goruldu (kenar optimumu); izgara 0,002-0,018 m ince adimlarla genisletildi ve yeniden kosuldu. Secim yine yalniz ic dogrulama macro-F1'ine dayanir; ilk kosunun test sayilari karar ya da secimde kullanilmadi.
- 2026-11-04, 05 (OLD boru hatti) **kosu sonrasi not (sonuc gorulmeden degil, uygulama ayrintisi)**: (i) pytorch_optimizer uzakta kurulu degil -> TFT optimizer'i notebook'un varsayilan 'ranger'i yerine Adam (port farki); (ii) state-change dizisi notebook'taki gibi ham kare etiketlerinden ve none(0) durumu DAHIL kuruldu -> egitim ornegi ~36 degil ~66-78 (DESIGN 3 gecis/dongu varsaymisti), TFT yine veri-aclik cekiyor; (iii) ic dogrulama orani TFT icin dongu bazli %20 (DESIGN yalniz 'ic bolme' diyordu; encoder 24 nedeniyle %10 ~2 dongu = <25 satir ornek vermez); (iv) tum dizilerde tek sabit group_id. Duyarlilik: --normalizer none, --seq deb (05_old_none.md, 05_old_deb.md).
- 2026-11-04, **koşu sonrası, doğrulayıcıların işaret ettiği kayıt boşlukları** (uygulama ayrıntısı; hiçbiri sonuca bakılarak seçilmedi):
  - (i) Heatmap kanal 1 orijinal `build_heatmap` ile birebir aynı: hücre başına hız **toplamı**, ortalama değil (ikili göstergeye bölünüyor). DESIGN "ortalama" demişti.
  - (ii) D1'e bir START bağlamı eklendi: ilk ziyaretin olasılığı da puanlanıyor.
  - (iii) D2 = log(d + 0,05 s); σ tabanı 0,05; referans için en az 3 eğitim gözlemi; referanslı adımı olmayan döngünün D2'si 0. D0'da σ tabanı 0,05.
  - (iv) "Geçiş" alt kümesi = etiket değişiminin ±2 karesi (sınır başına 4 kare). 5 karelik okuma 9.370 pencere verirdi, koşulan 7.727.
  - (v) PyTorch GRU'da recurrent dropout yok: GRU çıkışında Dropout 0,3, ardından mevcut Dropout 0,4 (üst üste).
  - (vi) S0 iç doğrulaması sınıf-tabakalı rastgele %10 (eski protokolü taklit eder); S1/S2 için oturum başına grup bazlı %10. CNN ile tabanlar farklı iç-doğrulama tohumları kullanıyor.
- 2026-11-04, **karar ölçütü 2'nin yorumu (sonuç görüldükten sonra; açıkça kaydedilir).**
  - Ölçüt bir eşdeğerlik testi olarak yazıldı: "taban CNN-GRU'dan kötü değil". Tabanın CNN-GRU'yu anlamlı biçimde geçmesi öngörülmedi.
  - B2 − CNN-GRU D3 AUROC farkı 8 hücrenin 7'sinde şartı sağlıyor. pred × fast hücresinde sağlamıyor: +0,058, %95 GA [+0,006, +0,110]. Bunun tek nedeni B2'nin **anlamlı olarak daha iyi** olması.
  - "Aksi halde" dalı, CNN-GRU'nun kazandığı bir alt küme olduğunu varsayıyor. Böyle bir alt küme yok:
    - kutu: 0,894 vs 0,933;
    - geçiş: 0,818 vs 0,834;
    - s01 mF1: 0,774 vs 0,784;
    - none: 0,852 vs 0,850, yani eşit.
  - Hüküm: "basit taban yeterli". Ölçüt ruhen sağlanıyor; harfi harfine sonuç (7/8) sonuç belgesinde ayrıca veriliyor.

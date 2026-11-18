# Faz A4 — Hakem itirazları M1–M4 (ANALİZ PLANI, 2026-11-05)

## Statü (A2/A3 ile aynı; değiştirilemez)
- Bu analizler, makale taslağına yazılan iç hakem raporunun (`paper/review/referee.md`, M1–M4) istediği **ikincil** analizlerdir. Faz A, A2 ve A3 sonuçları görüldükten **sonra** planlandı.
- Faz A'nın sayıları birincil olarak kalır. A4'te tanımlanan bütün sayılar raporlanır; seçici raporlama yok.
- Yeni veri toplanmaz.
- Faz A / A2 / A3 dosyalarının üzerine yazılmaz; kod `faz_a4/`, çıktılar `reports/faz_a4/` altına gider.
- Yeniden kullanılan modüller: `faz_a2/a2_cycles.py`, `faz_a/fa_common.py`.
- **Yasak:** s02'deki doğal sapma döngülerinin alarm durumuna bakmak.
- Sapmalar en altta tarihli olarak tutulur.

## A4.1 (M1) — Anomali türü oturumla karışık mı?
**(a) Başlangıçtan önce verilen alarmlar:**
- Kapsam: A3.3'teki B2 × pred alarmlarından seq135'te 17/71, seq145'te 17/114 alarm.
- Her biri için tetikleyen olay (dwell / geçiş / boşluk) çıkarılır. Ayrıca alarm anındaki **B2 tahmini durum** kaydedilir: radar yanlış kutuyu ToF eşiğinden önce mi gördü, yoksa alarm Box1 kalışından ya da Box1→ geçişinden mi geldi?
- İki açıklama ayrılır:
  - (i) radar yaklaşmayı erken görüyor: alarm anında tahmin Box3/Box4;
  - (ii) oturuma özgü zamanlama: alarm Box1 dwell ya da Box1→X boşluğundan geliyor.
- Sayılar ve oranlar raporlanır.

**(b) Karşı kanal = negatif kontrol** (B2 × pred, Faz A standart p, α = 0,10 ve 0,05):
- D1 (sıra) alarm oranı tempo oturumlarında (s05, s06) ve normal s02'de.
- D2 (süre) alarm oranı sıra oturumlarında (s03, s04) ve normal s02'de.
- Wilson %95 GA ve döngü-kümeli bootstrap GA verilir.
- Önceden yazılmış okuma: karşı kanal oranı normal FAR'ın GA'sı içindeyse "oturum etkisi bu kanalda saptanamadı"; dışındaysa "oturum etkisi var" denir ve büyüklüğü raporlanır.
- s05'teki D1 oranının Box5 silinmesiyle kirlendiği (A4.2) not edilir.

**(c) Oturum başına adım süreleri (betimleyici):**
- 30 Hz ToF segmentlerinden, s02–s06 için Box1 kalışı, Box1→sonraki kutu geçişi, ikinci kutu kalışı, →Box5 geçişi, Box5 kalışı ve döngü süresi.
- Medyan ve IQR verilir.

- Çıktı: `reports/faz_a4/01_session_confound.md`.

## A4.2 (M2) — Tempo tespitinin kanal dökümü ve oracle'ın davranışı
- B2 × pred ve ToF × tof için, D3 alarmı alan fast ve slow döngüler kanallara ayrılır (α = 0,10, eşik min(p1, p2) ≤ 0,05):
  - yalnız D1;
  - yalnız D2;
  - ikisi birden.
- Fast döngülerde, kare düzeyi dizisinde Box5 olmayanların (kısalmış dizi) payı ve bunların D1 ile alarm alma oranı raporlanır.
- **Önceden yazılmış ikincil tempo tanımı:** D2 tek başına, α = 0,10'da. Bu "tempo kanalı" tespiti olarak ayrıca verilir; makalede "tempo deviations detected" bu sayıyla nitelenir.
- Oracle'ın slow'da %60'ta kalmasının dökümü: D1 / D2 / D3 ayrı ayrı ve p dağılımı. Açıklama betimleyicidir.
- Çıktı: `reports/faz_a4/02_tempo_channels.md`.

## A4.3 (M3) — Sızıntı etkisi aynı donanımda; orijinal Keras modeli
**(a) Aynı donanım:**
- PyTorch CNN-GRU'nun S1 5 katı yerel GTX 1650 SUPER'da, `faz_a2/12_cnn_s0_seeds.py` ile aynı kod ve tohum kurallarıyla yeniden eğitilir.
- Sızıntı etkisi = yerel S0 (tohum 1–4) − yerel S1 (5 kat), Welch %95 GA.
- Uzak donanımda aynı-donanım karşılaştırması (S0 tohum 42 − uzak S1 = +3,2) yanında raporlanır.
- **Önceden yazılmış okuma:** makaledeki başlık-düzeyi iddia, iki aynı-donanım tahmininden **küçük olanı** kullanır. Beş tohumluk karışık tahmin ikincil kalır.

**(b) Orijinal Keras CNN-GRU:**
- Kaynak: klasör 4 `dataset/notebooks/01_mmwave_cnn_lstm_model.ipynb`. Mimari, eğitim ayarları, sınıf ağırlıkları ve dropout'lar defterle aynı. Optimizasyon: Adam 5e-4 + ExponentialDecay(8000, 0,92), batch 32, en fazla 120 epoch, EarlyStopping(val_loss, patience 25, restore_best).
- Eğitim: S1 5 katı ve S0 tohum 42 (Faz A bölmesi). Aynı heatmap girdileri; iç doğrulama Faz A kuralıyla (S1 grup bazlı %10, S0 tabakalı %10).
- Ortam: TensorFlow 2.18 + Keras 3.9.2, ayrı venv, CPU. Windows'ta TF GPU yok; CPU'da tek seferde ~1 saat bekleniyor.
- Raporlanan: S1 accuracy / macro-F1, kat ortalaması ± sd; S0 tohum 42; eşli kat farkı (Keras − B2).
- **Önceden yazılmış karar kuralı (Faz A ölçütü 1'in aynısı):**
  - B2'nin S1 macro-F1'i Keras'ınkine 2 puan yakın ya da üstündeyse, "basit taban yeterli" hükmü Keras için de geçerlidir.
  - Keras, B2'yi 2 puandan fazla geçerse hüküm **değişir**: makale bunu açıkça yazar, ana model iddiası Keras CNN-GRU lehine düzeltilir ve "sıralama ters dönüyor" iddiası Keras ile yeniden sınanır (S0 tohum 42).
- Keras modeli anomali boru hattına bu fazda **sokulmaz**; gerekirse ayrı planlanır.

- Çıktı: `reports/faz_a4/03_leakage_matched_keras.md`.

## A4.4 (M4) — Anomali oturumlarını hiç görmemiş durum tahmincisi
Faz A anomali protokolü (roller, segmentasyon, conformal) değiştirilmeden, B2 durumları iki yeni kaynaktan alınır:

- **(i) S2:** Mevcut `b2__S2__f*` OOF tahminleri (oturum dışarıda bırakılarak). Her oturum, kendisini görmemiş bir modelle tahmin edilmiştir.
- **(ii) "Devreye alma gerçekçi":** B2 yalnız s01 + s02 ile eğitilir.
  - s02'nin anomali katı k'deki döngüleri, s01 + s02'nin k dışındaki katlarıyla eğitilen modelle tahmin edilir.
  - s03–s06, s01 + s02'nin tamamıyla eğitilen tek modelle tahmin edilir.
  - Özellikler, RF ayarları ve tohum B2 ile aynı.

- **Raporlanan** (pred segmentasyonu, D3, α = 0,10 ve 0,05):
  - FAR (Wilson GA), tür başına tespit, AUROC.
  - D1 / D2 tek başına.
  - Faz A Tablo 3 ile yan yana.
  - Ayrıca bu kaynakların durum macro-F1'i (s02–s06).
- **Önceden yazılmış okuma:**
  - Tümü-AUROC düşüşü ≤ 0,05 ve tespit düşüşü ≤ 0,10 ise "Faz A sonuçları anomali oturumlarının görülmesine bağlı değil".
  - Değilse makale (ii)'yi **dağıtım gerçekçi tahmin** olarak öne alır ve Faz A sayılarını iyimser diye niteler.
- Devreye alma dakikaları, kullanılan eğitim setine göre yeniden etiketlenir.
- Çıktı: `reports/faz_a4/04_unseen_sessions.md`.

## Kapsam dışı
- Keras modelinin anomali boru hattına sokulması.
- Zaman sıralı kalibrasyon.
- D0 + D1 kombinasyonu.
- Eksik kare analizi.

Bunlar makalede sınırlama ya da gelecek iş olarak yazılır.

## Sapmalar (tarihli)
- 2026-11-05 (A4.3a): faz_a4/13_cnn_s1_local.py - S1 ic dogrulama ve torch tohumu 01 ile birebir (m01.fold_indices, tohum SEED_S1+1000+k); OOF/runs yalniz reports/faz_a4/ altina yazilir (cnngru__S1local__f{k}); cihaz CUDA (GTX 1650 SUPER), torch.set_num_threads(3).
- 2026-11-05 (A4.2): Birim = (dongu, f) cifti; GA = dongu-kumeli bootstrap (4000 tekrar), normal FAR icin Wilson; kanal esigi D1 = p1 <= 0.05, D2 = p2 <= 0.05 (D3 alpha = 0.10 ile ayni). Brief'e ek olarak (raporlanan, secici degil): ToF x pred ve B2 x tof hucreleri; kisalmis dizi payi slow/normal icin de; D2'yi belirleyen adim (en buyuk |z|); kat bazinda cal D2 uclari ve cal-maksimum sahibi dongunun kimligi (yalniz cal-rolu skoru; natural_dev / alarm durumuna bakilmadi). Bos ziyaret dizisi (visit_seq NaN) 'Box5 yok' sayildi. **DUZELTME (2026-11-05, hakem turu B1):** yukaridaki 'natural_dev / alarm durumuna bakilmadi' ifadesi eksikti. Ilk kosuda `02_tempo_channels.py` bolum 3'te s02 normal test ciftlerini `natural_dev` ile ayirip `FARexNat_*` (dogal sapma haric FAR; D1/D2/D3) hesaplamis ve `02_tempo_secondary_D2alone.csv`, md bolum 3 tablosu ('FAR D2 (dogal sapma haric)') ve stdout'a yazmisti; bu sutun s02 natural_dev dongulerinin alarm durumunu dolayli gosterebilirdi (yasagin ihlali). Kod, sutunlar, md kolonu ve stdout satiri silindi; 02 ciktilari yeniden uretildi (yalniz `FARexNat_*` sutunlari/kolonu kayboldu, diger tum sayilar ayni). Ihlal gecici olarak okunmus bir degerdir: ilk kosunun degerleri yeniden raporlanmaz ve hicbir analizde kullanilmaz; A4.2'nin diger sonuclari (D1/D2/D3 kanal dokumu, oracle) bu sutuna bagli degildi.
- 2026-11-05 (A4.1): (a) A3.3 tetik/onset mantigi 03_online.py'den ice aktarildi (17/71 ve 17/114 birebir yeniden uretildi); kategori kurali: (i) alarm karesinde B2 durumu = anomali kutusu (seq135->Box3, seq145->Box4), (ii) degilse ve tetik Box1 dwell ya da Box1->X, (iii) diger; BIRINCIL durum = alarm mantiginin gordugu debounce'lu B2 etiketi (ilk kosuda ham y_pred kullanilmisti: 18/16; debounce'luya gecis ilk sonuc gorulduktan sonra yapildi, iki tanim da raporda yan yana). (b) 'icinde' = karsi kanal nokta tahmini, ayni kanalin s02 normal FAR Wilson GA'sinda; bootstrap GA'larinin ortusmesi ikincil; Wilson havuzlu (dongu,f) ciftleri uzerinde, bootstrap kume = oturum#dongu (B=10000, tohum 20261105); s02 normal natural_dev ayrilmadan butun alindi; brief'e ek: D3 referansi, tek-oturum satirlari, normal FAR ile fark + kume-bootstrap GA, ALTINDA/USTUNDE yonu. (c) adim sureleri tof_segments (30 Hz; debounce 0.1/0.15 s) ile, yalniz ToF dizisi beklenen diziyle birebir eslesen donguler icin; dongu suresi Faz A ToF dongu siniri (tum donguler); ek: s02'ye gore medyan orani ve (a)'daki (ii) alarmlarini veren donguler icin Box1 dwell baglantisi.

- 2026-11-05 (A4.3b, Keras): Mimari/ayarlar defterden BIREBIR (GRU(128, dropout .3, recurrent_dropout .2) -> Dense(6 softmax); Dense64/ek Dropout(0.4) YOK; Conv(relu)->BN sirasi; Adam 5e-4+ExpDecay(8000,.92), batch 32, <=120 epoch, EarlyStopping(val_loss,25,restore_best), balanced class_weight, girdi olcekleme yok). Defterden farklar: (i) defter validation_data=TEST kullaniyordu -> burada Faz A ic dogrulamasi (01_state_cnn fold_indices ile ayni indeksler, md5 dogrulandi); (ii) ModelCheckpoint(val_accuracy) kullanilmadi (yalniz EarlyStopping restore_best); (iii) heatmap Faz A onbellegi (kanal 1 = hiz TOPLAMI; defterde hucre ORTALAMASI) ve etiket = fa_common y_true; (iv) tohum = 20261104+1000*split_id+fold (defterde tohum yok); (v) ortam: .venv-tf = Python 3.12 + tensorflow-cpu 2.18.0 + keras 3.9.2 + numpy 2.0.2 + pandas 3.0.6 + sklearn 1.9.1, CPU, TF intra 4 / inter 2 is parcacigi. Betikler: faz_a4/14_keras_train.py (+ ana venv ince baslatici 14_keras_launch.py).
- 2026-11-05 (A4.4): Brief s02'nin tek parça döngüsünü (loop 0, fold -1, 1 pencere) tanımlamıyor; hiçbir eğitim kümesine alınmadı, k=0 modeliyle tahmin edildi. Ayrıca: B2 özellik/RF kodu 02_state_baselines.py'den kopyalandı (modül import edilmez; çalıştırıldığında OOF yazar) ve kopya, Faz A S1 f0 ile S2 f1 OOF'unu birebir yeniden üretti; RF n_jobs=3 (sonucu etkilemez); (ii) OOF'ları reports/faz_a4/oof/b2_s0102__f0..f5 (f0-f4 = s02 anomali katı k, f5 = s03-s06 tek model); labels_from_oof(strict=False) yalnız tamlık doğrulamalarından (i: 16.248 pencere/6 kat, ii: 11.569 pencere) sonra kullanıldı; devreye alma dakikası = eğitim penceresi sayısı/600 ('fiili' tanım, A2.4b ile aynı); okuma kuralındaki 'tespit düşüşü' = B2 x pred, D3, 'tüm anomali', α=0,10 (tür başına düşüşler ek olarak raporlandı); ek tof segmentasyonu betimleyici olarak raporlandı; s02 doğal-sapma döngülerinin alarm durumu ve far_excl_nat raporlanmadı. Sebep: brif bu ayrıntıları belirtmiyor.
- 2026-11-05 00:15, **çalıştırma (sonuç görülmeden):**
  - İlk PyTorch S1 işi 7. epoch'ta sessizce öldü. Sebep: başlatan ajan bitince Windows iş nesnesi alt süreçleri kapattı. `tools/local_job.py`'ye CREATE_BREAKAWAY_FROM_JOB eklendi ve iş baştan başlatıldı.
  - Keras işi 4 iş parçacığıyla epoch başına 114 s sürüyordu (~14 saat). Makine geceleyin bu işe ayrıldığı için durduruldu. Aynı ayarlarla, iki paralel süreç halinde yeniden başlatıldı: S1:0–2 ve S1:3–4 + S0:0, her biri TF intra 7 / inter 1.
  - İş parçacığı sayısı yalnız hızı ve CPU'daki sayısal belirlenimsizliği etkiler; model, veri ve ayarlar değişmedi.
- 2026-11-05 00:31, **çalıştırma (sonuç görülmeden):** Yerel CPU'da Keras iki süreçte de epoch başına ~210 s sürdü (öğlene yetişmezdi) ve durduruldu.
  - Keras artık uzak makinede (RTX 3090) koşuyor: TensorFlow 2.18.0 GPU sürümü; Keras 3.9.2, numpy 2.0.2, pandas 3.0.6 ve diğer sürümler yerel `.venv-tf` ile aynı.
  - Başlatıcı `faz_a4/14_keras_remote.py`; GPU paylaşıldığı için bellek büyümesi açık. Altı eğitim tek işte sırayla koşuyor; epoch başına ~6,5 s.
  - Böylece Keras sonuçları Faz A'nın uzak CNN-GRU koşularıyla aynı GPU'da. CPU/GPU sayısal farkı not edilir.
- 2026-11-05 02:40, **eksik kaydın sonradan tamamlanması** (verify_b madde 7). Kayıt sonuçlar görüldükten sonra yazıldı; değişikliklerin kendisi 00:31'de, sonuç görülmeden yapılmıştı.
  - Değişen dosyalar: `tools/remote_run.sh` ve `tools/remote_start.sh`. Uzağa gönderilen klasör listesine `faz_a4` ve `faz_b` eklendi; uzak `.venv-tf` ile çalıştırma desteği geldi.
  - Bunlar yalnız başlatıcı altyapısı. Model, veri, ayar ya da metrik değişmedi.
  - `14_keras_runs.csv`'deki `device` sütunu her satırda "cpu" diyor; bu `14_keras_train.py:206`'da sabit yazılmış bir etiket. Gerçekte Keras koşuları uzak RTX 3090'da, epoch başına ~6,6 s sürdü (uzak günlük: `reports/_logs/job-20261105-003141-626-10306.log`).

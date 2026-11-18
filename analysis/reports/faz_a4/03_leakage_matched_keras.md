# Faz A4 / A4.3 — Sızıntı etkisi aynı donanımda; orijinal Keras CNN-GRU (M3)

Üreten: `faz_a4/03_leakage_matched_keras.py`. Önceden kayıtlı plan: `faz_a4/DESIGN_A4.md` §A4.3 (okuma kuralı ve karar kuralı sonuç görülmeden yazılmıştır). Bu bölüm **ikincil** analizdir; Faz A'nın sayıları birincil kalır. Tüm sayılar OOF dosyalarından `fa_common.state_metrics` ile yeniden hesaplandı (accuracy; macro-F1 = test kümesinde bulunan sınıflar üzerinden). Puan = yüzde puan.

## Koşu durumu ve çapraz kontroller
- PyTorch yerel S1 işi (`job-20261105-001436-6884`): 5 kat tamam (`13_cnn_s1_local_runs.csv`), betik `faz_a4/13_cnn_s1_local.py`.
- Keras uzak işi (`job-20261105-003141-626-10306`): S1 5 kat + S0 tohum 42 tamam (`14_keras_runs.csv`); TF 2.18.0 GPU + Keras 3.9.2, RTX 3090 (günlükte `Created device ... RTX 3090`). `14_keras_runs.csv`'deki `device=cpu` sütunu betikte sabit yazılmış bir etikettir (14_keras_train.py), gerçek cihaz GPU'dur (epoch başına ~6,6 s; yerel CPU'da ~210 s idi).
- S1: cnngru_remote / cnngru_local / keras OOF dosyalarının test pencereleri (oturum, kare) ve y_true, B2 ile her katta birebir aynı (assert).
- S0 tohum 42: Keras, uzak CNN-GRU ve B2 OOF test pencereleri (n=3.250) birebir aynı (assert).
- `13_cnn_s1_local_runs.csv` (5 koşu) ile OOF'tan yeniden hesaplanan accuracy/macro-F1: hepsi eşleşti (|fark| < 6e-6).
- `14_keras_runs.csv` (6 koşu) ile OOF'tan yeniden hesaplanan accuracy/macro-F1: hepsi eşleşti (|fark| < 6e-6).
- Uzak S1 CNN-GRU kat ortalamaları (acc/macro-F1) `faz_a2/02_leakage_summary.csv` içindeki S1 ortalamasıyla birebir aynı.
- Eksik dosya yok.

## Koşu başına sonuçlar (test pencereleri)

| model | bölme | koşu (kat / tohum) | n | accuracy | macro-F1 | kaynak |
|---|---|---|---|---|---|---|
| CNN-GRU (Keras, uzak 3090) | S1 | 0 | 3300 | 82.06 | 79.06 | A4.3b, uzak RTX 3090 |
| CNN-GRU (Keras, uzak 3090) | S1 | 1 | 3224 | 85.17 | 82.47 | A4.3b, uzak RTX 3090 |
| CNN-GRU (Keras, uzak 3090) | S1 | 2 | 3223 | 87.03 | 84.96 | A4.3b, uzak RTX 3090 |
| CNN-GRU (Keras, uzak 3090) | S1 | 3 | 3237 | 86.07 | 82.90 | A4.3b, uzak RTX 3090 |
| CNN-GRU (Keras, uzak 3090) | S1 | 4 | 3264 | 84.68 | 80.72 | A4.3b, uzak RTX 3090 |
| CNN-GRU (Keras, uzak 3090) | S0 | 42 | 3250 | 90.34 | 88.41 | A4.3b, uzak RTX 3090 |
| CNN-GRU (PyTorch, uzak 3090) | S1 | 0 | 3300 | 85.45 | 82.75 | Faz A, uzak RTX 3090 |
| CNN-GRU (PyTorch, uzak 3090) | S1 | 1 | 3224 | 88.59 | 85.83 | Faz A, uzak RTX 3090 |
| CNN-GRU (PyTorch, uzak 3090) | S1 | 2 | 3223 | 87.65 | 85.33 | Faz A, uzak RTX 3090 |
| CNN-GRU (PyTorch, uzak 3090) | S1 | 3 | 3237 | 87.15 | 83.67 | Faz A, uzak RTX 3090 |
| CNN-GRU (PyTorch, uzak 3090) | S1 | 4 | 3264 | 85.81 | 82.82 | Faz A, uzak RTX 3090 |
| CNN-GRU (PyTorch, uzak 3090) | S0 | 42 | 3250 | 90.09 | 87.81 | Faz A, uzak RTX 3090 |
| CNN-GRU (PyTorch, yerel 1650S) | S1 | 0 | 3300 | 85.03 | 81.97 | A4.3a, yerel GTX 1650S |
| CNN-GRU (PyTorch, yerel 1650S) | S1 | 1 | 3224 | 86.14 | 83.16 | A4.3a, yerel GTX 1650S |
| CNN-GRU (PyTorch, yerel 1650S) | S1 | 2 | 3223 | 87.43 | 85.23 | A4.3a, yerel GTX 1650S |
| CNN-GRU (PyTorch, yerel 1650S) | S1 | 3 | 3237 | 88.14 | 84.48 | A4.3a, yerel GTX 1650S |
| CNN-GRU (PyTorch, yerel 1650S) | S1 | 4 | 3264 | 84.65 | 81.82 | A4.3a, yerel GTX 1650S |
| CNN-GRU (PyTorch, yerel 1650S) | S0 | 1 | 3250 | 91.38 | 89.63 | A2.2, yerel GTX 1650S |
| CNN-GRU (PyTorch, yerel 1650S) | S0 | 2 | 3250 | 91.51 | 90.09 | A2.2, yerel GTX 1650S |
| CNN-GRU (PyTorch, yerel 1650S) | S0 | 3 | 3250 | 91.51 | 89.93 | A2.2, yerel GTX 1650S |
| CNN-GRU (PyTorch, yerel 1650S) | S0 | 4 | 3250 | 92.25 | 90.91 | A2.2, yerel GTX 1650S |
| B2 (el yapımı özellik + RF) | S1 | 0 | 3300 | 86.88 | 84.55 | Faz A, CPU |
| B2 (el yapımı özellik + RF) | S1 | 1 | 3224 | 89.80 | 87.01 | Faz A, CPU |
| B2 (el yapımı özellik + RF) | S1 | 2 | 3223 | 88.64 | 86.33 | Faz A, CPU |
| B2 (el yapımı özellik + RF) | S1 | 3 | 3237 | 89.09 | 86.15 | Faz A, CPU |
| B2 (el yapımı özellik + RF) | S1 | 4 | 3264 | 87.62 | 84.18 | Faz A, CPU |
| B2 (el yapımı özellik + RF) | S0 | 42 | 3250 | 89.08 | 86.19 | A2.2, CPU |

## (a) Aynı donanımda sızıntı etkisi (S0 rastgele pencere bölmesi − S1 grup-bazlı katlar), CNN-GRU PyTorch

| tahmin | metrik | S0 | S1 | etki (puan) | %95 GA (puan) | not |
|---|---|---|---|---|---|---|
| yerel-eslesmis (yerel S0 tohum 1-4 - yerel S1) | accuracy | 91.66 ± 0.39 | 86.28 ± 1.50 | **+5.38** | [+3.55, +7.22] | S0: yerel tohum 1–4 (n=4), S1: yerel 5 kat (n=5); Welch t |
| CAPRAZ-donanim (yerel S0 tohum 1-4 - uzak S1) [karisik] | accuracy | 91.66 ± 0.39 | 86.93 ± 1.30 | **+4.73** | [+3.15, +6.32] | yalnız bilgi: farklı donanım |
| yerel-eslesmis (yerel S0 tohum 1-4 - yerel S1) | macro-F1 | 90.14 ± 0.54 | 83.33 ± 1.51 | **+6.81** | [+4.96, +8.65] | S0: yerel tohum 1–4 (n=4), S1: yerel 5 kat (n=5); Welch t |
| CAPRAZ-donanim (yerel S0 tohum 1-4 - uzak S1) [karisik] | macro-F1 | 90.14 ± 0.54 | 84.08 ± 1.43 | **+6.06** | [+4.31, +7.81] | yalnız bilgi: farklı donanım |
| uzak-eslesmis (S0 tohum 42 - uzak S1) [tek S0 kosusu, GA yok] | accuracy | 90.09 | 86.93 ± 1.30 | **+3.16** | — | S0: uzak tohum 42 (n=1), S1: uzak 5 kat; tek S0 koşusu, GA yok (Faz A2'deki +3,16/+3,73) |
| uzak-eslesmis (S0 tohum 42 - uzak S1) [tek S0 kosusu, GA yok] | macro-F1 | 87.81 | 84.08 ± 1.43 | **+3.73** | — | S0: uzak tohum 42 (n=1), S1: uzak 5 kat; tek S0 koşusu, GA yok (Faz A2'deki +3,16/+3,73) |
| 5-tohum karisik (S0: uzak 42 + yerel 1-4; S1: uzak) [A2.2] | accuracy | 91.35 | 86.93 | **+4.42** | [+2.79, +6.04] | S0: uzak 42 + yerel 1–4 (n=5), S1: uzak 5 kat (A2.2); donanım karışık |
| 5-tohum karisik (S0: uzak 42 + yerel 1-4; S1: uzak) [A2.2] | macro-F1 | 89.67 | 84.08 | **+5.59** | [+3.69, +7.50] | S0: uzak 42 + yerel 1–4 (n=5), S1: uzak 5 kat (A2.2); donanım karışık |

**Yerel S1 − uzak S1 (aynı katlar, aynı tohumlar, farklı GPU/PyTorch yapısı):**

| metrik | yerel S1 ort ± sd | uzak S1 ort ± sd | fark (yerel − uzak, puan) | eşli kat farkı %95 GA | Welch %95 GA |
|---|---|---|---|---|---|
| accuracy | 86.28 ± 1.50 | 86.93 ± 1.30 | **-0.65** | [-2.23, +0.92] (kat farkı sd 1.27, 1/5 katta yerel > uzak) | [-2.71, +1.40] |
| macro-F1 | 83.33 ± 1.51 | 84.08 ± 1.43 | **-0.74** | [-2.34, +0.85] (kat farkı sd 1.28, 1/5 katta yerel > uzak) | [-2.89, +1.40] |

(Aynı kat bölmesi ve aynı model tohumu kullanıldığından kat bazlı eşli fark, GPU/cuDNN/PyTorch yapısı kaynaklı belirlenimsizliği ve donanım farkını ölçer; iki yapının bit düzeyinde aynı olması beklenmez.)

### Önceden yazılmış okuma kuralının uygulanması
Kural (DESIGN_A4 §A4.3a): makaledeki başlık-düzeyi iddia, iki aynı-donanım tahmininden **küçük olanı** kullanır; beş tohumluk karışık tahmin ikincil kalır.

- **accuracy:** yerel-eşleşmiş +5.38 puan (GA [+3.55, +7.22]) ve uzak-eşleşmiş +3.16 puan → küçük olan **uzak-eşleşmiş**: **+3.16 puan** başlık değeridir. Karışık 5-tohum tahmin +4.42 (ikincil).
- **macro-F1:** yerel-eşleşmiş +6.81 puan (GA [+4.96, +8.65]) ve uzak-eşleşmiş +3.73 puan → küçük olan **uzak-eşleşmiş**: **+3.73 puan** başlık değeridir. Karışık 5-tohum tahmin +5.59 (ikincil).

Okuma notları (veriyle sınırlı):
- İki eşleşmiş tahmin birbirinden 2.2 puan (accuracy) / 3.1 puan (macro-F1) ayrılıyor. Bu ayrım donanım, model tohumu ve bölme-tohumu etkilerinden hangisinden geldiği bu verilerle ayrıştırılamaz (yerel S0 tohum 1–4, uzak S0 tohum 42'den farklı bölmeler de içerir). Her iki tahmin de pozitif ve S1 kat sd'sinden (CNN-GRU için ≈1,3–1,5 puan) büyük; yön (rastgele bölme S1'den yüksek) hiçbir tahminde değişmiyor.
- accuracy: uzak-eşleşmiş nokta tahmin (+3.16), yerel-eşleşmiş %95 GA'nın [+3.55, +7.22] **dışında** (altında).
- macro-F1: uzak-eşleşmiş nokta tahmin (+3.73), yerel-eşleşmiş %95 GA'nın [+4.96, +8.65] **dışında** (altında).
- Kural gereği başlık değeri **tek S0 koşusuna** dayanan uzak-eşleşmiş tahmindir (GA hesaplanamaz); makale metninde başlık sayısı yazılırken bunun tek koşu olduğu ve yerel-eşleşmiş değerin (daha büyük, GA'lı) birlikte verilmesi gerekir.
- Yerel S0 koşuları (91,4–92,2 accuracy) uzak S0 tohum 42'den (90,1) yüksek; yerel S1 ise uzak S1'den biraz düşük (yukarıdaki yerel − uzak tablosu: −0,65 / −0,74 puan, eşli GA 0'ı içeriyor). Yani yerel donanımda S0 daha yüksek, S1 daha düşük çıkmış; bu, yerel-eşleşmiş farkı büyütüyor. S0 tarafı farklı tohumlar da içerdiği için bunun donanım mı tohum mu olduğu ayrıştırılmadı.

## (b) Orijinal Keras CNN-GRU (klasör 4 defteri `01_mmwave_cnn_lstm_model.ipynb`'in portu)

**Port sadakati (kayıt: DESIGN_A4 Sapmalar, 2026-11-05 A4.3b).** Defterden birebir: GRU(128, dropout 0,3, recurrent_dropout 0,2) → Dense(6, softmax) (Dense64 ve ek Dropout(0,4) yok); Conv(relu) → BatchNorm sırası; Adam 5e-4 + ExponentialDecay(8000; 0,92), batch 32, ≤ 120 epoch, EarlyStopping(val_loss, patience 25, restore_best), dengeli class_weight, girdi ölçekleme yok (n_param = 1.914.022, tüm koşularda aynı). **Defterden farklar:** (i) defter `validation_data=TEST` kullanıyordu → burada Faz A iç doğrulaması (S1: grup bazlı %10; S0: sınıf-tabakalı %10; indeksler `01_state_cnn.py` ile aynı, md5 doğrulandı); (ii) `ModelCheckpoint(val_accuracy)` kullanılmadı (yalnız EarlyStopping restore_best); (iii) heatmap Faz A önbelleği (kanal 1 = hız TOPLAMI; defterde hücre ortalaması) ve etiket = `fa_common` y_true; (iv) tohum = 20261104 + 1000·split_id + kat (defterde tohum yok); (v) ortam: TF 2.18.0 + Keras 3.9.2 (defteri kaydeden sürüm), GPU (RTX 3090). **İş parçacığı/donanım sapması (kayıtlı):** yerel CPU koşusu epoch başına ~210 s sürdüğü için iş uzak GPU'ya taşındı; model, veri ve ayarlar aynı kaldı. Her koşuda kayıtlı `restore check` (geri yüklenen ağırlıkla val_loss = en iyi val_loss) günlükte eşleşiyor.

### S1 (grup-bazlı 5 kat): Keras ve B2, kat bazında

| kat | Keras acc | Keras macro-F1 | B2 acc | B2 macro-F1 | PyTorch uzak macro-F1 | PyTorch yerel macro-F1 | Keras − B2 macro-F1 (puan) | epoch (en iyi) |
|---|---|---|---|---|---|---|---|---|
| 0 | 82.06 | 79.06 | 86.88 | 84.55 | 82.75 | 81.97 | -5.49 | 57 (32) |
| 1 | 85.17 | 82.47 | 89.80 | 87.01 | 85.83 | 83.16 | -4.55 | 42 (17) |
| 2 | 87.03 | 84.96 | 88.64 | 86.33 | 85.33 | 85.23 | -1.37 | 48 (23) |
| 3 | 86.07 | 82.90 | 89.09 | 86.15 | 83.67 | 84.48 | -3.24 | 90 (65) |
| 4 | 84.68 | 80.72 | 87.62 | 84.18 | 82.82 | 81.82 | -3.45 | 45 (20) |

### S1 özet (kat ortalaması ± sd, n = 5 kat)

| model | accuracy | macro-F1 | havuzlanmış (5 katın OOF'u birleşik) acc / macro-F1 |
|---|---|---|---|
| CNN-GRU (Keras, uzak 3090) | 85.00 ± 1.87 | 82.02 ± 2.24 | 84.99 / 82.14 |
| B2 (el yapımı özellik + RF) | 88.41 ± 1.16 | 85.64 ± 1.22 | 88.40 / 85.78 |
| CNN-GRU (PyTorch, uzak 3090) | 86.93 ± 1.30 | 84.08 ± 1.43 | 86.92 / 84.10 |
| CNN-GRU (PyTorch, yerel 1650S) | 86.28 ± 1.50 | 83.33 ± 1.51 | 86.27 / 83.39 |

### Eşli kat farkı: Keras − B2 (S1)

| metrik | Keras ort | B2 ort | fark (puan) | %95 GA (t, df=4) | kat sd | Keras > B2 olan kat | t | p |
|---|---|---|---|---|---|---|---|---|
| macro-F1 | 82.02 | 85.64 | **-3.62** | [-5.54, -1.70] | 1.55 | 0/5 | -5.23 | 0.006 |
| accuracy | 85.00 | 88.41 | **-3.40** | [-5.05, -1.76] | 1.33 | 0/5 | -5.74 | 0.005 |

### Önceden yazılmış karar kuralının LİTERAL uygulanması
Kural (DESIGN_A4 §A4.3b; Faz A ölçütü 1): B2'nin S1 macro-F1'i Keras'ınkine 2 puan yakın **ya da üstündeyse** "basit taban yeterli" hükmü Keras için de geçerlidir; Keras B2'yi **2 puandan fazla** geçerse hüküm değişir. Karşılaştırma ölçüsü: S1 macro-F1'in kat ortalamaları (Faz A ölçütünün kullandığı büyüklük); Keras − B2 = 82.02 − 85.64 = **-3.62 puan**.

- **KURAL TETİKLENMEDİ.** Keras − B2 = -3.62 puan; eşik +2,00. B2, Keras'ın önünde (B2 3.62 puan önde). "Basit taban yeterli" hükmü Keras için de geçerli (bu ölçütün macro-F1 şartı bakımından).
- Duyarlılık (aynı sonuç mu?): kat bazlı eşli GA [-5.54, -1.70], Keras'ın B2'yi geçtiği kat sayısı 0/5; havuzlanmış macro-F1 farkı -3.64 puan. Kural eşiği (2 puan) aşılmıyor: GA üst sınırı -1.70.
- Kapsam: Faz A ölçütü 1'in yalnız birinci şartı (durum macro-F1) Keras için sınanabilir; ikinci şart (D3 döngü-AUROC farkı) Keras anomali boru hattına sokulmadığı için (DESIGN_A4 "Kapsam dışı") değerlendirilmedi.

### Keras: S0 tohum 42, sızıntı etkisi ve eski makalenin %93,66'sı

| model | S0 tohum 42 acc | S0 tohum 42 macro-F1 |
|---|---|---|
| CNN-GRU (Keras, uzak 3090) | 90.34 | 88.41 |
| CNN-GRU (PyTorch, uzak 3090) | 90.09 | 87.81 |
| B2 (el yapımı özellik + RF) | 89.08 | 86.19 |
| eski makale (Keras defteri, rastgele bölme; değer hakem raporunda yazarların notu olarak anılıyor, burada yeniden üretilmedi) | 93,66 | — |

- **Keras sızıntı etkisi (accuracy):** S0(42) 90.34 − S1 ort. 85.00 = **+5.34 puan** (tek S0 koşusu; S1 kat sd 1.87; GA yok). Karşılaştırma: PyTorch CNN-GRU uzak-eşleşmiş +3.16, yerel-eşleşmiş +5.38.
- **Keras sızıntı etkisi (macro-F1):** S0(42) 88.41 − S1 ort. 82.02 = **+6.39 puan** (tek S0 koşusu; S1 kat sd 2.24; GA yok). Karşılaştırma: PyTorch CNN-GRU uzak-eşleşmiş +3.73, yerel-eşleşmiş +6.81.
- **S0 tohum 42'de sıralama (Keras vs B2):** Keras 90.34 / 88.41 (acc / macro-F1), B2 89.08 / 86.19 → Keras − B2 = +1.26 / +2.22 puan. Rastgele bölmede Keras önde, grup-bazlı S1'de B2 önde → "sıralama ters dönüyor" bulgusu Keras ile de görülüyor (tek S0 koşusu; yön olarak).
- **Keras portu ile PyTorch portu aynı S0 bölmesinde:** Keras 90.34 vs PyTorch 90.09 → Keras − PyTorch = +0.25 puan accuracy (+0.60 macro-F1).
- **Eski makalenin %93,66'sı ile karşılaştırma:** Keras portu S0 tohum 42'de 90.34 → eski değerin **3.32 puan altında**; PyTorch portu 90.09 idi (hakem raporunda belirtilen ~3,6 puanlık açık). Yani Keras'a geçmek açığın yalnız 0.25 puanını kapatıyor; katman sırası / dropout yerleşimi açıklaması (yazarların varsayımı) bu deneyle **desteklenmiyor**. Kalan açık bu deneyde izole edilmedi; eski defterin test kümesini doğrulama olarak kullanması ve `ModelCheckpoint(val_accuracy)` ile en iyi test epoch'unu seçmesi (iyimser bir seçim) ile heatmap kanal 1 tanımı, bölme ve etiket farkları adaydır, ama hiçbiri burada ayrı ayrı sınanmadı (aday açıklamalardır, bulgu değil).

## Sınırlar ve dürüstlük notları
- Yerel-eşleşmiş tahmin n=4 (S0) vs n=5 (S1) ile Welch aralığıdır; S0 tohumlarının test kümeleri aynı 16.248 pencereden çekildiği için örtüşür, S1 katlarının eğitim kümeleri paylaşılır; aralıklar yaklaşıktır (A2.2'deki not geçerli).
- Uzak-eşleşmiş tahmin tek S0 koşusudur; makaledeki başlık sayısı bu kurala göre seçilirse bunun belirsizliği bilinmeyen bir tek-koşu belirsizliğidir. Her iki eşleşmiş tahmin ve GA'sı birlikte verilmelidir.
- Keras: S0 için tek koşu (tohum 42), S1 için 5 kat; eşli GA df=4 ile geniştir. Karar kuralı nokta tahmin (kat ortalaması farkı) üzerinden literal uygulandı; GA ve kat sayıları duyarlılık olarak verildi, kuralın yerine geçmez.
- Keras GPU koşusu cuDNN belirlenimsizliği içerir; yerel CPU ile bit düzeyinde aynı sonuç beklenmez. Keras modeli anomali boru hattına sokulmadı (kapsam dışı).
- Seçici raporlama yok: A4.3'te tanımlanan tüm sayılar yukarıdadır; eksik koşu yoktur.

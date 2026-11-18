# Faz A2 / A2.2 — Sızıntı bir örüntü mü? (S0 rastgele pencere bölmesi vs S1 grup-bazlı katlar)

Üreten: `faz_a2/02_leakage.py` (B1/B2 S0: `11_baselines_s0_seeds.py`; CNN-GRU S0 tohum 1–4: `12_cnn_s0_seeds.py`). Önceden kayıtlı: `faz_a2/DESIGN_A2.md` §A2.2. Bu bir **ikincil** analizdir; ana model seçimini (B2 + D3) değiştirmez.

## Protokol
- **S0** = sınıf-tabakalı rastgele %80/20 pencere bölmesi (`train_test_split(test_size=0.2, stratify=y_true, random_state=tohum)`), tohumlar **42 (Faz A) ve 1, 2, 3, 4**; her tohumda 12.998 eğitim / 3.250 test penceresi. Tohum 42 bölmesi Faz A `split_s0.csv` ile birebir aynı (betikte assert).
- **S1** = Faz A grup-bazlı 5 kat (`StratifiedGroupKFold`, grup = (oturum, döngü)); S1 değerleri `reports/faz_a/oof/` içindeki Faz A OOF dosyalarından **kat başına** yeniden hesaplandı (havuzlanmadı). Kat başına n = 3.223–3.300.
- Her bölmede B1/B2/CNN-GRU aynı hiperparametre ve aynı etiketlerle çalışır. Metrikler `fa_common.state_metrics` (accuracy; macro-F1 = test kümesinde bulunan sınıflar üzerinden).
- **Tohum şeması** (DESIGN_A2'de açık bırakılan ayrıntı, burada sabitlendi ve üç modele aynı uygulandı): bölme `random_state` = tohum; iç doğrulama (CNN erken durdurma ve B1 τ seçimi) = eğitim pencerelerinden sınıf-tabakalı rastgele %10, `random_state` = tohum (01_state_cnn.py'nin S0 dalıyla aynı); model tohumu (torch başlangıç / RF `random_state`) = 20261104 + tohum. **İstisna:** CNN-GRU tohum 42, Faz A'dan olduğu gibi alındı (model tohumu 20261104; yeniden eğitilmedi). B1/B2'nin S0 sonuçları Faz A'da yoktu; 5 tohumun tümü bu aşamada yeni koşuldu (`02_state_baselines.py` kodu birebir, yalnız bölme farklı; B2 RF 300 ağaç, `class_weight=balanced`, n_jobs=3; B1 τ'yu iç doğrulama macro-F1'iyle seçer).
- **Etki** = ortalama(S0, 5 tohum) − ortalama(S1, 5 kat), accuracy ve macro-F1 için; %95 GA = Welch t aralığı: d ± t(0,975; df)·√(s₀²/5 + s₁²/5), df Welch–Satterthwaite. Puan = yüzde puan.
- **Donanım notu:** S1 CNN-GRU eğitimi uzak RTX 3090'da; yeni S0 tohumları (CNN-GRU 1–4) yerel GTX 1650 SUPER'de koştu; CNN-GRU S0 tohum 42 Faz A'dan (uzak 3090). GPU/cuDNN belirlenimsizliği ve PyTorch sürüm farkı tam tekrarlanabilirliği engeller; B1/B2 CPU'da deterministiktir.

## Çapraz kontroller
- CNN-GRU S0 tohum 42 yeniden hesap: acc 0.90092 / macro-F1 0.87807 = Faz A cnn_runs.csv (0.90092 / 0.87807) -> ESLESTI
- S0 OOF dosyalarının test pencereleri (oturum, kare) ilgili tohumun bölmesiyle birebir aynı (her dosya için assert).
- Faz A OOF dosyalari + cnn_runs.csv + split_s0.csv: 41 dosyanin md5'i is oncesi anlik goruntuyle AYNI (hicbiri degismedi).
- Eksik S0 koşuları: yok (3 model × 5 tohum = 15 OOF dosyası mevcut)

## Özet: S0 vs S1 ve sızıntı etkisi

| model | metrik | S0 ort ± sd (n=5 tohum) | S1 ort ± sd (n=5 kat) | etki (puan) | %95 GA (puan) | Welch df | t | p |
|---|---|---|---|---|---|---|---|---|
| B1 en-yakin-merkez | accuracy | 61.64 ± 0.67 | 61.30 ± 1.23 | **+0.34** | [-1.18, +1.87] | 6.2 | 0.55 | 0.6 |
| B1 en-yakin-merkez | macro-F1 | 53.46 ± 0.59 | 52.65 ± 1.36 | **+0.81** | [-0.85, +2.47] | 5.4 | 1.22 | 0.27 |
| B2 el-yapimi ozellik + RF | accuracy | 88.86 ± 0.28 | 88.41 ± 1.16 | **+0.45** | [-0.98, +1.87] | 4.5 | 0.84 | 0.44 |
| B2 el-yapimi ozellik + RF | macro-F1 | 86.29 ± 0.41 | 85.64 ± 1.22 | **+0.65** | [-0.84, +2.14] | 4.9 | 1.12 | 0.31 |
| CNN-GRU | accuracy | 91.35 ± 0.78 | 86.93 ± 1.30 | **+4.42** | [+2.79, +6.04] | 6.6 | 6.52 | 0.00043 |
| CNN-GRU | macro-F1 | 89.67 ± 1.15 | 84.08 ± 1.43 | **+5.59** | [+3.69, +7.50] | 7.6 | 6.83 | 0.00017 |

Makale cümlesi için (DESIGN_A2'ye göre; hesaplanan sayılarla; etki işareti ve GA'ya göre ifade seçilmelidir):
- B1 en-yakin-merkez, accuracy: rastgele pencere bölmesi − grup-bazlı bölme = **+0.34** puan (Welch SE 0.63; %95 GA [-1.18, +1.87]; GA 0'ı içeriyor (etki belirgin değil)).
- B1 en-yakin-merkez, macro-F1: rastgele pencere bölmesi − grup-bazlı bölme = **+0.81** puan (Welch SE 0.66; %95 GA [-0.85, +2.47]; GA 0'ı içeriyor (etki belirgin değil)).
- B2 el-yapimi ozellik + RF, accuracy: rastgele pencere bölmesi − grup-bazlı bölme = **+0.45** puan (Welch SE 0.53; %95 GA [-0.98, +1.87]; GA 0'ı içeriyor (etki belirgin değil)).
- B2 el-yapimi ozellik + RF, macro-F1: rastgele pencere bölmesi − grup-bazlı bölme = **+0.65** puan (Welch SE 0.58; %95 GA [-0.84, +2.14]; GA 0'ı içeriyor (etki belirgin değil)).
- CNN-GRU, accuracy: rastgele pencere bölmesi − grup-bazlı bölme = **+4.42** puan (Welch SE 0.68; %95 GA [+2.79, +6.04]; GA 0'ı dışlıyor).
- CNN-GRU, macro-F1: rastgele pencere bölmesi − grup-bazlı bölme = **+5.59** puan (Welch SE 0.82; %95 GA [+3.69, +7.50]; GA 0'ı dışlıyor).

**Yorum notları (veriyle sınırlı):**
- Aralıklar yaklaşıktır: S0 tohumlarının test kümeleri aynı 16.248 pencereden çekilir (örtüşür) ve S1 katlarının eğitim kümeleri birbiriyle paylaşılır; tohumlar/katlar bağımsız örnekler değildir. S0 sd'si bölme-tohumu + model-tohumu varyansını, S1 sd'si ise farklı grupların (döngülerin) zorluk farkını da içerir; ikisi aynı nicelik değildir.
- n=5 vs n=5 ile df küçüktür; GA'lar t dağılımıyla (normal değil) hesaplandı.
- CNN-GRU karıştırıcısı: S0 tohum 42 (uzak 3090, Faz A) accuracy 90.09; yerel 1650S'teki tohum 1–4 91.38–92.25 (ort 91.66). Tek bir 3090 koşusuyla donanım/model-tohumu etkisi ayrıştırılamaz; 5-tohum ortalaması 4 yerel + 1 uzak koşunun karışımıdır. Tohum 42-yalnız (aynı donanım) etki en alttaki ekte ayrıca verilmiştir.
- B1/B2 için fark küçük ve GA 0'ı içeriyor; CNN-GRU için GA 0'ı dışlıyor. Bu, gözlenen sızıntı farkının bu veri ve bu üç model için derin modelde göründüğünü, düşük kapasiteli/elle çıkarılmış özellikli modellerde (B1: tek kare centroid; B2: 21 özellik + RF) ayırt edilemediğini söyler; nedeni (kapasite, girdi türü) bu deneyle ayrıştırılmadı.

## Tohum başına S0 sonuçları

| model | tohum | n_test | accuracy | macro-F1 | kaynak |
|---|---|---|---|---|---|
| B1 en-yakin-merkez | 42 | 3250 | 0.6234 | 0.5444 | yerel CPU, bu aşama |
| B1 en-yakin-merkez | 1 | 3250 | 0.6126 | 0.5356 | yerel CPU, bu aşama |
| B1 en-yakin-merkez | 2 | 3250 | 0.6120 | 0.5312 | yerel CPU, bu aşama |
| B1 en-yakin-merkez | 3 | 3250 | 0.6240 | 0.5324 | yerel CPU, bu aşama |
| B1 en-yakin-merkez | 4 | 3250 | 0.6102 | 0.5296 | yerel CPU, bu aşama |
| B2 el-yapimi ozellik + RF | 42 | 3250 | 0.8908 | 0.8619 | yerel CPU, bu aşama |
| B2 el-yapimi ozellik + RF | 1 | 3250 | 0.8855 | 0.8576 | yerel CPU, bu aşama |
| B2 el-yapimi ozellik + RF | 2 | 3250 | 0.8889 | 0.8641 | yerel CPU, bu aşama |
| B2 el-yapimi ozellik + RF | 3 | 3250 | 0.8858 | 0.8621 | yerel CPU, bu aşama |
| B2 el-yapimi ozellik + RF | 4 | 3250 | 0.8917 | 0.8689 | yerel CPU, bu aşama |
| CNN-GRU | 42 | 3250 | 0.9009 | 0.8781 | Faz A (`cnngru__S0__f0`, uzak 3090) |
| CNN-GRU | 1 | 3250 | 0.9138 | 0.8963 | yerel GTX 1650S, bu aşama |
| CNN-GRU | 2 | 3250 | 0.9151 | 0.9009 | yerel GTX 1650S, bu aşama |
| CNN-GRU | 3 | 3250 | 0.9151 | 0.8993 | yerel GTX 1650S, bu aşama |
| CNN-GRU | 4 | 3250 | 0.9225 | 0.9091 | yerel GTX 1650S, bu aşama |

## Kat başına S1 sonuçları (Faz A OOF, kat başına yeniden hesap)

| model | kat | n_test | accuracy | macro-F1 |
|---|---|---|---|---|
| B1 en-yakin-merkez | 0 | 3300 | 0.6000 | 0.5125 |
| B1 en-yakin-merkez | 1 | 3224 | 0.6331 | 0.5447 |
| B1 en-yakin-merkez | 2 | 3223 | 0.6075 | 0.5194 |
| B1 en-yakin-merkez | 3 | 3237 | 0.6108 | 0.5193 |
| B1 en-yakin-merkez | 4 | 3264 | 0.6137 | 0.5368 |
| B2 el-yapimi ozellik + RF | 0 | 3300 | 0.8688 | 0.8455 |
| B2 el-yapimi ozellik + RF | 1 | 3224 | 0.8980 | 0.8701 |
| B2 el-yapimi ozellik + RF | 2 | 3223 | 0.8864 | 0.8633 |
| B2 el-yapimi ozellik + RF | 3 | 3237 | 0.8909 | 0.8615 |
| B2 el-yapimi ozellik + RF | 4 | 3264 | 0.8762 | 0.8418 |
| CNN-GRU | 0 | 3300 | 0.8545 | 0.8275 |
| CNN-GRU | 1 | 3224 | 0.8859 | 0.8583 |
| CNN-GRU | 2 | 3223 | 0.8765 | 0.8533 |
| CNN-GRU | 3 | 3237 | 0.8715 | 0.8367 |
| CNN-GRU | 4 | 3264 | 0.8581 | 0.8282 |

## CNN-GRU S0 eğitim kayıtları (cihaz / epoch)

| koşu | cihaz | epoch (best) | best val loss | süre (s) | n_train / n_val / n_test | model tohumu |
|---|---|---|---|---|---|---|
| S0 tohum 42 (Faz A) | cuda (uzak RTX 3090) | 79 (54) | 0.28043 | 396 | 11698 / 1300 / 3250 | 20261104 |
| S0 tohum 1 (A2.2) | cuda (yerel GTX 1650 SUPER) | 83 (58) | 0.24867 | 719 | 11698 / 1300 / 3250 | 20261105 |
| S0 tohum 2 (A2.2) | cuda (yerel GTX 1650 SUPER) | 83 (58) | 0.28275 | 718 | 11698 / 1300 / 3250 | 20261106 |
| S0 tohum 3 (A2.2) | cuda (yerel GTX 1650 SUPER) | 90 (65) | 0.24597 | 777 | 11698 / 1300 / 3250 | 20261107 |
| S0 tohum 4 (A2.2) | cuda (yerel GTX 1650 SUPER) | 81 (56) | 0.25469 | 698 | 11698 / 1300 / 3250 | 20261108 |

CPU'ya düşen tohum: yok (tümü CUDA).

## Sınırlamalar
- CNN-GRU S1 sayıları uzak 3090'da, S0 tohum 1–4 yerel 1650S'te üretildi; donanım/belirlenimsizlik farkı etkiye karışabilir (tohum 42 S0 ile S1 aynı donanımdadır; aşağıda tohum 42-yalnız etki de verilir).
- S0 ve S1 aynı pencere kümesini paylaşır ama test kümeleri farklıdır (S0 rastgele %20; S1 katları ayrık, grup-bazlı). S0'da ardışık pencereler (4/5 kare örtüşmesi) eğitim ve testte birlikte bulunur — sızıntı mekanizması budur; etki bu mekanizmanın toplam ölçüsüdür, ayrıştırılmış bir nedensel kestirim değildir.

### Ek: tohum 42 S0 ile S1 katları (CNN-GRU için aynı donanım: uzak 3090)
- CNN-GRU accuracy: S0(42) 90.09 − S1 ortalama 86.93 = **+3.16** puan (tek koşu; S1 kat sd 1.30).
- CNN-GRU macro-F1: S0(42) 87.81 − S1 ortalama 84.08 = **+3.73** puan (tek koşu; S1 kat sd 1.43).


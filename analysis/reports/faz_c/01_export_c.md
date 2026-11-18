# Faz C / C.1 - Kucuk ormanin C'ye aktarimi ve derlenmis gercek boyutu

Brief: `faz_c/DESIGN_C.md` C.1 (calistirmadan once yazildi). Betik: `faz_c/01_export_c.py`. Uretilen C: `faz_c/c/` (tam veri modelleri `forest_m10_full_{a,b,c}.c`, `forest_m25_full_{a,b,c}.c`, `driver.c`, `forest_api.h`). Tum sayilar stdout'ta ve `01_export_c*.csv` dosyalarinda. Cipte CALISTIRILMADI; x86-64 mingw gcc -Os ile derlendi.

## Ozet

- **Kapi 0 (Faz B yeniden uretimi):** GECTI (tum satirlarda |fark| < 1e-9; ayrinti asagida).
- **Esdegerlik kapisi (kodlama a, double toplama, C == sklearn predict):** GECTI (%100) - secilen n=10 modeli: gecti; toplam uyusmayan pencere (n10+n25, a): 0.
- C cikarimi ile bagimsiz numpy yeniden uygulamasi (ayni dizilerden) tum model/kodlama/birimlerde birebir ayni toplam ve tahmin.
- **Karar kurali (harfiyen):** secilen n=10/None modeli, kodlama (c) tam veri modelinde **189.9 KB** sabit veri (olculen `.rdata`), dF1(B2'ye gore) = -0.0103 >= -0.02 -> **192 KB MSS veri RAM** icine sigar.

## 1. Kapi 0 - Faz B yeniden uretimi

| model | olcu | bu betik | Faz B / Faz A | |fark| |
|---|---|---|---|---|
| tf_n10_dN | macro_f1_mean | 0.846121 | 0.846121 | 0.00e+00 |
| tf_n10_dN | nodes_fold_mean | 16759.600000 | 16759.600000 | 0.00e+00 |
| tf_n10_dN | nodes_full | 20460.000000 | 20460.000000 | 0.00e+00 |
| tf_n10_dN | auroc_all | 0.937032 | 0.937032 | 0.00e+00 |
| tf_n10_dN | far_a10 | 0.039216 | 0.039216 | 0.00e+00 |
| tf_n10_dN | det_a10_all | 0.534653 | 0.534653 | 0.00e+00 |
| tf_n25_dN | macro_f1_mean | 0.851755 | 0.851755 | 0.00e+00 |
| tf_n25_dN | nodes_fold_mean | 41955.000000 | 41955.000000 | 0.00e+00 |
| tf_n25_dN | nodes_full | 50479.000000 | 50479.000000 | 0.00e+00 |
| tf_n25_dN | auroc_all | 0.939554 | 0.939554 | 0.00e+00 |
| tf_n25_dN | far_a10 | 0.057692 | 0.057692 | 9.71e-17 |
| tf_n25_dN | det_a10_all | 0.452941 | 0.452941 | 5.55e-17 |

Faz A B2 macro-F1 (5 kat ortalamasi) = 0.8564 (referans 0.8564).

`tree_.value` sklearn 1.9.1'de ihrac sirasinda kontrol edildi (varsayilmadi): tum agaclarda max |satir toplami-1| = 2.220e-16 (dugum basina normalize edilmis oranlar). Ihracta yine sklearn predict_proba gibi agac basina normalize edildi (value / value.sum).

Yaprak saflik (ek bilgi, `_leaf_purity.csv`): n=10 tam veri modelinde 10235 yapraktan 0 tanesi saf degil (0.00000); agaclar sinirsiz derinlikte oldugundan yapraklar neredeyse saftir; (b) nicemleme ve (c) sert oy bu nedenle (a) ile ayni tahmini verir.

## 2. Uygulama

- Struct-of-arrays, `static const`: `FEAT` uint8 (yaprak = 255), `THR` float32, `LEFT`/`RIGHT` int16 (agac-yerel indeks; agac basina dugum <= 32767), yaprak dugumunde `LEFT` = agac-yerel yaprak indeksi; `NOFF`/`LOFF` agac basina dugum/yaprak taban ofseti (uint16, toplam dugum <= 65535). Yaprak tablosu ayri: (a) 6 x float32, (b) 6 x uint8, (c) 1 x uint8. Yaprak dugumunde `THR`/`RIGHT` kullanilmaz (yer harcanir; optimize edilmedi).
- Esik: float64 esigi asmayan en buyuk float32 (`nextafter` ile asagi yuvarlama); ihracta tum bolunmelerde dogrulandi.
- (b) nicemleme: floor(p*255 + 0.5); toplam uint16; (c) yaprak argmax (esitlikte kucuk sinif), oylar uint8; esitlikte kucuk sinif.
- Kapi: sklearn'in gordugu float32 matris ham ikili dosyadan C surucusune; ozellikler C'de yeniden hesaplanmadi; toplama double.

## 3. Esdegerlik kapisi (kodlama a)

| model | birim | pencere | uyusmayan (double) | uyusma (double) | uyusmayan (gomulu float32 toplama) | uyusma (float32) |
|---|---|---|---|---|---|---|
| n10_None | f0 | 3300 | 0 | 1.000000 | 0 | 1.000000 |
| n10_None | f1 | 3224 | 0 | 1.000000 | 0 | 1.000000 |
| n10_None | f2 | 3223 | 0 | 1.000000 | 0 | 1.000000 |
| n10_None | f3 | 3237 | 0 | 1.000000 | 0 | 1.000000 |
| n10_None | f4 | 3264 | 0 | 1.000000 | 0 | 1.000000 |
| n10_None | full | 16248 | 0 | 1.000000 | 0 | 1.000000 |
| n25_None | f0 | 3300 | 0 | 1.000000 | 0 | 1.000000 |
| n25_None | f1 | 3224 | 0 | 1.000000 | 0 | 1.000000 |
| n25_None | f2 | 3223 | 0 | 1.000000 | 0 | 1.000000 |
| n25_None | f3 | 3237 | 0 | 1.000000 | 0 | 1.000000 |
| n25_None | f4 | 3264 | 0 | 1.000000 | 0 | 1.000000 |
| n25_None | full | 16248 | 0 | 1.000000 | 0 | 1.000000 |

Uyusmayan pencere yok (tum kodlama/birimler).

(b)/(c) icin sklearn predict ile toplam uyusmayan pencere (5 kat test + tam veri modeli toplami):

| model | kodlama | pencere | uyusmayan |
|---|---|---|---|
| n10_None | b | 32496 | 0 |
| n10_None | c | 32496 | 0 |
| n25_None | b | 32496 | 0 |
| n25_None | c | 32496 | 0 |

## 4. Kodlama sonuclari (S1, 5 kat)

dF1 = C kodlamasinin kat macro-F1'i eksi referans (B2 = Faz A 300 agac; ya da ayni modelin sklearn float cikarimi), eslesmis 5 kat, %95 t-GA (df=4).

| model | kodlama | uyusma (sklearn) | macro-F1 (ort ± sd) | dF1 vs B2 [GA] | dF1 vs float [GA] | AUROC (tum anomali, D3) | FAR@0.10 | tespit@0.10 | AUROC seq | AUROC tempo |
|---|---|---|---|---|---|---|---|---|---|---|
| n10_None | a | 1.000000 | 0.8461 ± 0.0146 | -0.0103 [-0.0144, -0.0062] | +0.0000 [+0.0000, +0.0000] | 0.9370 ± 0.0425 | 0.0392 | 0.5347 | 0.9189 | 0.9559 |
| n10_None | b | 1.000000 | 0.8461 ± 0.0146 | -0.0103 [-0.0144, -0.0062] | +0.0000 [+0.0000, +0.0000] | 0.9370 ± 0.0425 | 0.0392 | 0.5347 | 0.9189 | 0.9559 |
| n10_None | c | 1.000000 | 0.8461 ± 0.0146 | -0.0103 [-0.0144, -0.0062] | +0.0000 [+0.0000, +0.0000] | 0.9370 ± 0.0425 | 0.0392 | 0.5347 | 0.9189 | 0.9559 |
| n25_None | a | 1.000000 | 0.8518 ± 0.0110 | -0.0047 [-0.0080, -0.0014] | +0.0000 [+0.0000, +0.0000] | 0.9396 ± 0.0374 | 0.0577 | 0.4529 | 0.9206 | 0.9596 |
| n25_None | b | 1.000000 | 0.8518 ± 0.0110 | -0.0047 [-0.0080, -0.0014] | +0.0000 [+0.0000, +0.0000] | 0.9396 ± 0.0374 | 0.0577 | 0.4529 | 0.9206 | 0.9596 |
| n25_None | c | 1.000000 | 0.8518 ± 0.0110 | -0.0047 [-0.0080, -0.0014] | +0.0000 [+0.0000, +0.0000] | 0.9396 ± 0.0374 | 0.0577 | 0.4529 | 0.9206 | 0.9596 |

Referans (sklearn float cikarimi, ayni model): n10_None: macro-F1 0.8461, AUROC 0.9370, FAR@0.10 0.0392, tespit@0.10 0.5347; n25_None: macro-F1 0.8518, AUROC 0.9396, FAR@0.10 0.0577, tespit@0.10 0.4529. Faz A B2 x pred D3: bkz. Faz B CSV (300 x None: AUROC 0.9237, FAR 0.0784, tespit 0.6335).

## 5. Gercek derlenmis boyut (gcc -Os -c, mingw64 x86-64)

`.rdata` = model sabitleri (dolgu dahil), `.text` = yalniz `forest_predict` (test fonksiyonu `forest_sum` derlenmedi). Aritmetik = dizi baytlari toplami (dugum x (1+4+2*2) + yaprak x yaprak-baytlari + agac ofsetleri). '8 B/dugum' B.1 varsayimidir. x86 `.text` ARM Cortex-R4F'yi temsil ETMEZ; yalniz sabit veri mimariden bagimsizdir.

| model | kodlama | dugum (fold-ort / tam) | yaprak (tam) | aritmetik KB (fold-ort / tam) | olculen .rdata KB (fold-ort / tam) | .rdata - aritmetik B (tam) | olculen / 8B-varsayim (fold-ort / tam) | .text B (fold-ort / tam) |
|---|---|---|---|---|---|---|---|---|
| n10_None | a | 16759.6 / 20460 | 10235 | 343.9 / 419.7 | 344.0 / 419.8 | 100 | 2.63x / 2.63x | 240 / 240.0 |
| n10_None | b | 16759.6 / 20460 | 10235 | 196.5 / 239.8 | 196.6 / 239.9 | 106 | 1.50x / 1.50x | 240 / 240.0 |
| n10_None | c | 16759.6 / 20460 | 10235 | 155.5 / 189.9 | 155.6 / 189.9 | 81 | 1.19x / 1.19x | 208 / 208.0 |
| n25_None | a | 41955.0 / 50479 | 25252 | 860.8 / 1035.6 | 860.9 / 1035.7 | 53 | 2.63x / 2.63x | 240 / 240.0 |
| n25_None | b | 41955.0 / 50479 | 25252 | 491.8 / 591.7 | 491.9 / 591.8 | 61 | 1.50x / 1.50x | 240 / 240.0 |
| n25_None | c | 41955.0 / 50479 | 25252 | 389.3 / 468.4 | 389.5 / 468.5 | 81 | 1.19x / 1.19x | 208 / 208.0 |
| n300_None(B2) | a | 504200.8 / 612946 | 306623 | 10345.9 / 12576.0 | derlenmedi | - | - | - |
| n300_None(B2) | b | 504200.8 / 612946 | 306623 | 5911.8 / 7186.2 | derlenmedi | - | - | - |
| n300_None(B2) | c | 504200.8 / 612946 | 306623 | 4680.1 / 5689.0 | derlenmedi | - | - | - |

B.1 varsayimi (8 B/dugum): n=10 fold-ort 130.9 KB, tam veri 159.8 KB.

## 6. Calisma bellegi (ikincil)

Pencere tamponu 5x7x4 = 140 B; ozellik vektoru 21x4 = 84 B; olasilik toplayici 6x4 = 24 B; toplam 248 B (+ tek satir sonuc). Gerceklesen C kodunda toplayicilar (b) uint16, (c) uint8 oldugundan fiili bellek bundan kucuktur; ozellik hesaplama kodu ve yigit dahil degildir.

## 7. Onceden yazilmis okuma ve karar kurali (harfiyen)

- **Okuma (a):** olculen sabit veri 8 B/dugum hesabindan buyuk: **evet** (oran fold-ort 2.63x, tam veri 2.63x). Tahmin ~1.5-3x: **tutuyor**.
- **Okuma (c) en kucuk:** **tutuyor** (tam veri .rdata: a 419.8 KB, b 239.9 KB, c 189.9 KB).

Karar tablosu (secilen n=10/None, tam veri modeli; esikler 192 KB = 196608 B, 256 KB, 512 KB; dF1 tolerans -0.02):

| kodlama | .rdata B | KB | dF1 vs B2 | dF1 vs float | <=192 KB | <=256 KB | <=512 KB | dF1(B2) >= -0.02 | dF1(float) >= -0.02 |
|---|---|---|---|---|---|---|---|---|---|
| a | 429920 | 419.8 | -0.0103 | +0.0000 | False | False | True | True | True |
| b | 245696 | 239.9 | -0.0103 | +0.0000 | False | True | True | True | True |
| c | 194496 | 189.9 | -0.0103 | +0.0000 | True | True | True | True | True |

Kural: herhangi bir kodlama <= 192 KB ve dF1 >= -0.02 ise -> "X KB'lik derlenmis model, MSS veri RAM'inin (192 KB) icine sigar"; yoksa 256 KB DSP L2 / 512 KB MSS program RAM'i; yoksa "sigmaz". dF1 olarak B.1'deki referansa uygun sekilde B2'ye gore dF1 kullanildi; float modele gore dF1 de ayni sonucu verip vermedigi son iki sutunda.

**Sonuc:** 192 KB MSS veri RAM icin gecerli. Makale icin onerilen cumle: "189.9 KB'lik derlenmis model (kodlama c, model sabitleri), MSS veri RAM'inin (192 KB) icine sigar" [`.rdata` olculen 194496 B; dF1(B2) -0.0103]. Pay: 196608 - 194496 = 2112 B (%1.1); sigma payi dar ve yalniz model sabitleri icin gecerli (calisma bellegi 248 B ayri; yazilimin kendisi/yigit/tamponlar sayilmadi). 192 KB'a sigmayan kodlamalar: (a) 419.8 KB, (b) 239.9 KB.

Her durumda: **cipte calistirilmadi; bellek aygit yazilimiyla paylasilir** (veri RAM'inin yazilim, yigit, radar ham verisi tamponlari ile paylasildigi dikkate alinmali; olculen boyut yalniz model sabitleridir ve x86-64 .rdata'dir).

## 8. Dosyalar
`faz_c/01_export_c.py`, `faz_c/c/*` (forest_m{10,25}_full_{a,b,c}.c, driver.c, forest_api.h), `faz_c/oof/` (C kodlamalarinin OOF'u), `reports/faz_c/01_export_c.csv` (boyut ozeti + sonuclar), `_sizes.csv` (her birim), `_sizes_summary.csv`, `_results.csv`, `_gate.csv`, `_folds.csv`, `_mismatch.csv`, `_ref_gate.csv`, `_decision.csv`.
# A2.5 - CPU gecikmesi (tek is parcacigi)

On-kayit: `faz_a2/DESIGN_A2.md` A2.5. Betik: `faz_a2/05_latency.py`. Her sayi stdout'a da basilmistir.

## Ortam

- CPU: AMD Ryzen 7 3700X (8 cekirdek / 16 is parcacigi), yerel Windows 11; Python 3.12.13, torch 2.6.0+cu124, sklearn 1.9.1, numpy 2.5.4.
- Tek is parcacigi: OMP/MKL/OPENBLAS_NUM_THREADS=1, torch.set_num_threads(1) (olculen torch is parcacigi = 1), RF n_jobs=1.
- Olcum: yontem basina 50 isinma + 1000 olculen pencere (S1 kat 0 test pencerelerinden sabit tohumlu rastgele secim, 3 kosuda ayni pencereler); `time.perf_counter_ns`; tek cagri = tek pencere (batch 1). Her kosu ayri bir surec.
- B1/B2 S1 kat 0 egitim verisiyle (B2: RF 300 agac, class_weight=balanced, tohum 20261104, 21 ozellik; B1: sinif centroidleri + Faz A tau). CNN-GRU **rastgele baslatilmis agirliklarla** (faz_a/01_state_cnn.py'deki birebir mimari, 1,921,446 parametre, eval modu, no_grad); gecikme agirlik degerlerine bagli degildir, egitim yapilmadi.
- Makine yuku (olcum oncesi): Toplam CPU yuku (5x2 s ortalama) olcum oncesi: %22,54; kosan baska proje python isi yok (local_job list: hepsi bitmis, exit 0); arka planda file-sync/remote-desktop vb. normal kullanici sureclari var.

## Sonuc: her kosu icin medyan [IQR] (p95), ms

| Yontem | Kosu 1 | Kosu 2 | Kosu 3 | Kosu medyanlari araligi |
|---|---|---|---|---|
| B1 | 0.0090 [0.0002] (0.0092) | 0.0090 [0.0001] (0.0092) | 0.0091 [0.0001] (0.0094) | 0.0090 - 0.0091 (yayilim %1.1) |
| B2 | 20.1520 [1.2423] (24.5516) | 20.1478 [1.0305] (24.1614) | 19.8466 [0.8849] (23.6635) | 19.8466 - 20.1520 (yayilim %1.5) |
| B2_features_only | 0.5430 [0.0344] (0.7794) | 0.5487 [0.0376] (0.8181) | 0.5414 [0.0181] (0.6267) | 0.5414 - 0.5487 (yayilim %1.4) |
| CNN-GRU (5 frames, full window) | 7.1422 [0.7890] (9.2503) | 7.5578 [1.0353] (10.5169) | 6.9132 [0.5549] (8.4158) | 6.9132 - 7.5578 (yayilim %8.9) |
| CNN-GRU streaming (1 new CNN frame + GRU over 5 cached) | 2.7376 [0.2636] (3.3657) | 2.8061 [0.2499] (3.5970) | 2.7264 [0.3580] (3.2879) | 2.7264 - 2.8061 (yayilim %2.9) |
| Heatmap build per frame (2x48x48) | 0.0426 [0.0007] (0.0481) | 0.0427 [0.0010] (0.0481) | 0.0418 [0.0011] (0.0483) | 0.0418 - 0.0427 (yayilim %2.1) |
| Frame stats from points (7 feats, approx.) | 0.0487 [0.0010] (0.0572) | 0.0490 [0.0119] (0.0797) | 0.0474 [0.0006] (0.0497) | 0.0474 - 0.0490 (yayilim %3.3) |

## Ozet (3 kosunun ortalamasi)

| Yontem | Medyan ms | IQR ms | p95 ms | 100 ms kare araligina orani (medyan) |
|---|---|---|---|---|
| B1 | 0.0090 | 0.0001 | 0.0093 | %0.009 |
| B2 | 20.0488 | 1.0526 | 24.1255 | %20.049 |
| B2_features_only | 0.5444 | 0.0300 | 0.7414 | %0.544 |
| CNN-GRU (5 frames, full window) | 7.2044 | 0.7931 | 9.3943 | %7.204 |
| CNN-GRU streaming (1 new CNN frame + GRU over 5 cached) | 2.7567 | 0.2905 | 3.4169 | %2.757 |
| Heatmap build per frame (2x48x48) | 0.0424 | 0.0009 | 0.0482 | %0.042 |
| Frame stats from points (7 feats, approx.) | 0.0484 | 0.0045 | 0.0622 | %0.048 |

## Notlar

- Tek-kare heatmap yapimi, onbellekteki tam boru hatti heatmap'ine karsi dogrulandi (olculen karelerin her 10'uncusu): max |fark| = 0.0. Karede ortalama 7.1 nokta.
- 'B2' = 21 ozelligin 5 karelik, onceden hesaplanmis kare istatistiklerinden cikarimi + RF.predict (300 agac). Kare istatistiklerinin ham noktalardan hesabi ayri satirdadir (yaklasik: SNR-agirlikli centroid, std, |v| ortalamasi; Faz A kare tablosunun v!=0 filtresini birebir yapmaz, yalniz maliyet mertebesi icindir).
- 'CNN-GRU (5 frames, full window)': 5 karenin hepsi CNN'den gecer (Faz A degerlendirmesi gibi; heatmap girdisi hazir, zamanlamaya dahil degil). 'CNN-GRU streaming (1 new CNN frame + GRU over 5 cached)' ek, betimleyici bir senaryodur: akista yalniz yeni karenin CNN ozelligi hesaplanir, onceki 4 ozellik onbellekten gelir (burada yer tutucu ozelliklerle taklit edildi; sayi yalniz gecikme icindir).
- Uctan uca pencere karari (yeni kare icin heatmap + model): CNN-GRU tam pencere + heatmap = 7.247 ms; CNN-GRU streaming + heatmap = 2.799 ms; B2 + kare istatistigi = 20.097 ms (medyanlarin toplami).
- **Eski makale karsilastirmasi:** eski makaledeki 12,6 ms CPU gecikmesi **Keras** ile ve **Ryzen 7 5800X** uzerinde olculmustu (farkli makine, farkli cerceve). Buradaki degerler PyTorch + Ryzen 7 3700X + tek is parcacigi ile olculmustur; dogrudan es-degerlik iddiasi yoktur.
- Faz A'daki eski B2 olcumu (`reports/faz_a/baseline_runs.json`: 13.32 ms, 300 ornegin ortalamasi) ile bu raporun B2 medyani (20.05 ms) karsilastirilabilir.

## Desteklenen cumle (ikincil)

"Masaustu CPU'da (Ryzen 7 3700X, tek is parcacigi) pencere basina B2 ~20.0 ms, CNN-GRU ~7.2 ms; her ikisi de 100 ms kare araliginin altinda (CNN-GRU araligin %7.2'i)."

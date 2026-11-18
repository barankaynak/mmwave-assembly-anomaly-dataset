# B.1 — Küçük ormanlar: B2 modeli radar çipinin belleğine sığar mı? (tahmin; port yok)

Üretim: `faz_b/01_tiny_forests.py` (ızgara + OOF + anomali hattı) ve `faz_b/01b_tiny_forests_report.py` (bu rapor, şekil). Ham sayılar: `01_tiny_forests.csv`; her yapılandırmanın tam S1 OOF'u `reports/faz_b/oof/tf_n*_d*__S1__f*.csv.gz`. **Statü: keşif amaçlı, plan koşudan önce yazıldı (`DESIGN_B.md` B.1); makaleye girmez.** Çip üzerinde hiçbir şey çalıştırılmadı; yalnız model boyutu ve işlem sayısı hesaplandı.

## Yöntem (DESIGN_B.md'ye göre)
- Özellikler, tohum (20261104), sınıf ağırlığı (`balanced`) ve S1 5 katı Faz A B2 ile aynı: 7 kare istatistiği × (son, ortalama, eğim) = 21 özellik, L = 5. Yalnız `n_estimators` ve `max_depth` değişir (30 yapılandırma × 5 kat = 150 model).
- **Düğüm sayısı** = 5 kat modelinin toplam düğüm (iç + yaprak) sayısı ortalaması (`nodes_mean`); ek olarak tüm 16.248 pencereyle eğitilmiş modelin düğüm sayısı (`nodes_full`, dağıtılacak modele yakın; kurala girmez).
- **Bayt** = düğüm × 8 B ve düğüm × 12 B (brief'teki varsayım; aşağıdaki uyarıya bakın).
- **Pencere başına karşılaştırma** = test pencerelerinde her ağaçta kökten yaprağa geçilen iç düğüm sayısı, ağaçlar üzerinden toplanıp pencereler üzerinden ortalanır (= ortalama yol uzunluğu × ağaç sayısı). En kötü durum (ağaç `max_depth` toplamı) ayrıca verilir.
- **Anomali hattı** her yapılandırmanın tam OOF `y_pred` çıktısıyla Faz A protokolünde koşuldu: ±3 kare debounce (MIN_BOX 2, MAX_GAP 1), `pred` segmentasyonu (tahmin edilen Box1 girişinden sonrakine), D1 + D2, D3 = min(1, 2 min(p1, p2)), standart conformal, Faz A rolleri (eğitim f+3,f+4; kal. f+1,f+2; test f). AUROC = kat bazlı AUROC'un ortalaması (± sd), `tüm anomali` grubu (seq135 + seq145 + fast + slow); FAR = α = 0,10'da 5 katın havuzlanmış normal test döngüleri (n ≈ 50, SE ≈ 0,042).

## Eşdeğerlik kapısı (300 ağaç, derinlik sınırsız = Faz A B2)
- macro-F1: 0.8564 ± 0.0122 (Faz A B2: 0.8564); B2'ye göre eşli kat farkı ortalaması +0.00e+00 (sd 0.00e+00).
- B2 × pred D3 AUROC (tüm anomali): 0.923697 ± 0.0352 (Faz A: 0.923697 ± 0,0352); FAR α = 0,10: 0.0784 (Faz A: 0,0784); tespit α = 0,10: 0.6335 (Faz A: 0,6335).
- **Sonuç: kapı geçti (Faz A B2 yeniden üretildi).**

## Çip belleği (kaynak)
- **IWR6843**, Texas Instruments veri sayfası **SWRS219F** (Ekim 2018, Nisan 2025 revizyonu), *Features → Internal memory with ECC* ve bellek tablosu: 1,75 MB = MSS program RAM (TCM-A) 512 KB + MSS veri RAM (TCM-B) 192 KB + DSP L1 64 KB (L1P 32 + L1D 32) + DSP L2 256 KB + L3 radar veri küpü RAM 768 KB.
- URL: https://www.ti.com/lit/ds/symlink/iwr6843.pdf (ürün sayfası https://www.ti.com/product/IWR6843).
- Çizgiler: 128 / 256 / 512 KB (brief) ve çip: MSS veri RAM 192 KB, DSP L2 256 KB, L3 768 KB, toplam 1792 KB. **Uyarı:** bu bellekler paylaşılır (L3 radar küpünü tutar; program RAM üretici yazılım + uygulama kodunu tutar; DSP L2 DSP kodu/verisi + ara tamponlar). Model sabitleri için fiilen kullanılabilir pay bu toplamdan küçüktür; aşağıdaki "sığar" yalnız model sabitlerinin tek başına bu kovaya girdiği anlamına gelir.

## Tüm ızgara (S1, 5 kat)
`nodes` = kat ortalaması; `full` = tüm veriyle eğitilmiş modelin düğüm sayısı; B8/B12 = KB (1 KB = 1024 B), kat ortalaması düğüm × 8 / 12 B; `kars.` = pencere başına karşılaştırma (ortalama; en kötü durum); `Δ B2` = macro-F1 eşli kat farkı (ortalama ± sd); AUROC = D3, tüm anomali, kat ortalaması ± sd; tespit/FAR α = 0,10.

| ağaç | derinlik | macro-F1 | Δ B2 | düğüm | full | B8 KB | B12 KB | kars. (en kötü) | yol uz. | AUROC D3 | seq | tempo | tespit | FAR (SE) | F1 ≥ 0,836 | AUROC ≥ 0,894 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 4 | 0.6320 ± 0.0155 | -0.2244 ± 0.0093 | 308 | 310 | 2.4 | 3.6 | 40.0 (40) | 4.00 | 0.8452 ± 0.1351 | 0.864 | 0.826 | 0.409 | 0.160 (0.042) | hayır | hayır |
| 10 | 6 | 0.7386 ± 0.0208 | -0.1178 ± 0.0100 | 1,140 | 1,174 | 8.9 | 13.4 | 59.8 (60) | 5.98 | 0.8709 ± 0.0837 | 0.843 | 0.900 | 0.152 | 0.020 (0.042) | hayır | hayır |
| 10 | 8 | 0.7795 ± 0.0149 | -0.0770 ± 0.0102 | 3,218 | 3,528 | 25.1 | 37.7 | 78.1 (80) | 7.81 | 0.8690 ± 0.0556 | 0.837 | 0.902 | 0.155 | 0.078 (0.042) | hayır | hayır |
| 10 | 10 | 0.8069 ± 0.0177 | -0.0496 ± 0.0082 | 6,356 | 7,022 | 49.7 | 74.5 | 92.6 (100) | 9.26 | 0.8870 ± 0.1009 | 0.855 | 0.922 | 0.589 | 0.137 (0.042) | hayır | hayır |
| 10 | 12 | 0.8226 ± 0.0111 | -0.0338 ± 0.0078 | 9,936 | 11,638 | 77.6 | 116.4 | 104.0 (120) | 10.40 | 0.9017 ± 0.0514 | 0.908 | 0.895 | 0.200 | 0.038 (0.042) | hayır | evet |
| 10 | None | 0.8461 ± 0.0146 | -0.0103 ± 0.0033 | 16,760 | 20,460 | 130.9 | 196.4 | 121.1 (260) | 12.11 | 0.9370 ± 0.0425 | 0.919 | 0.956 | 0.535 | 0.039 (0.042) | evet | evet |
| 25 | 4 | 0.6550 ± 0.0200 | -0.2015 ± 0.0129 | 771 | 771 | 6.0 | 9.0 | 100.0 (100) | 4.00 | 0.8543 ± 0.0856 | 0.903 | 0.803 | 0.464 | 0.040 (0.042) | hayır | hayır |
| 25 | 6 | 0.7463 ± 0.0220 | -0.1101 ± 0.0105 | 2,855 | 2,933 | 22.3 | 33.5 | 149.4 (150) | 5.98 | 0.9151 ± 0.0669 | 0.927 | 0.903 | 0.267 | 0.059 (0.042) | hayır | evet |
| 25 | 8 | 0.7871 ± 0.0184 | -0.0693 ± 0.0084 | 8,006 | 8,727 | 62.5 | 93.8 | 194.7 (200) | 7.79 | 0.8864 ± 0.1351 | 0.875 | 0.898 | 0.309 | 0.137 (0.042) | hayır | hayır |
| 25 | 10 | 0.8105 ± 0.0202 | -0.0459 ± 0.0104 | 15,976 | 17,585 | 124.8 | 187.2 | 231.4 (250) | 9.26 | 0.9163 ± 0.0484 | 0.896 | 0.938 | 0.570 | 0.098 (0.042) | hayır | evet |
| 25 | 12 | 0.8283 ± 0.0140 | -0.0281 ± 0.0056 | 24,578 | 28,813 | 192.0 | 288.0 | 258.6 (300) | 10.34 | 0.9373 ± 0.0429 | 0.932 | 0.943 | 0.396 | 0.058 (0.042) | hayır | evet |
| 25 | None | 0.8518 ± 0.0110 | -0.0047 ± 0.0027 | 41,955 | 50,479 | 327.8 | 491.7 | 300.0 (651) | 12.00 | 0.9396 ± 0.0374 | 0.921 | 0.960 | 0.453 | 0.058 (0.042) | evet | evet |
| 50 | 4 | 0.6640 ± 0.0198 | -0.1924 ± 0.0133 | 1,542 | 1,542 | 12.0 | 18.1 | 200.0 (200) | 4.00 | 0.9100 ± 0.0505 | 0.924 | 0.896 | 0.488 | 0.060 (0.042) | hayır | evet |
| 50 | 6 | 0.7458 ± 0.0164 | -0.1107 ± 0.0057 | 5,775 | 5,888 | 45.1 | 67.7 | 298.9 (300) | 5.98 | 0.8755 ± 0.1122 | 0.831 | 0.921 | 0.358 | 0.059 (0.042) | hayır | hayır |
| 50 | 8 | 0.7922 ± 0.0165 | -0.0642 ± 0.0061 | 16,541 | 17,508 | 129.2 | 193.8 | 389.7 (400) | 7.79 | 0.9019 ± 0.0966 | 0.917 | 0.886 | 0.379 | 0.118 (0.042) | hayır | evet |
| 50 | 10 | 0.8162 ± 0.0171 | -0.0403 ± 0.0079 | 32,943 | 36,250 | 257.4 | 386.0 | 462.9 (500) | 9.26 | 0.9221 ± 0.0651 | 0.902 | 0.943 | 0.399 | 0.078 (0.042) | hayır | evet |
| 50 | 12 | 0.8331 ± 0.0142 | -0.0234 ± 0.0063 | 50,463 | 57,706 | 394.2 | 591.4 | 516.1 (600) | 10.32 | 0.9245 ± 0.0775 | 0.925 | 0.923 | 0.550 | 0.137 (0.042) | hayır | evet |
| 50 | None | 0.8565 ± 0.0117 | +0.0000 ± 0.0035 | 84,037 | 101,108 | 656.5 | 984.8 | 595.1 (1292) | 11.90 | 0.9259 ± 0.0484 | 0.918 | 0.934 | 0.659 | 0.098 (0.042) | evet | evet |
| 100 | 4 | 0.6639 ± 0.0199 | -0.1925 ± 0.0139 | 3,087 | 3,092 | 24.1 | 36.2 | 400.0 (400) | 4.00 | 0.8754 ± 0.0712 | 0.884 | 0.867 | 0.594 | 0.120 (0.042) | hayır | hayır |
| 100 | 6 | 0.7467 ± 0.0167 | -0.1098 ± 0.0066 | 11,610 | 11,830 | 90.7 | 136.1 | 597.7 (600) | 5.98 | 0.8679 ± 0.1857 | 0.857 | 0.879 | 0.195 | 0.039 (0.042) | hayır | hayır |
| 100 | 8 | 0.7931 ± 0.0173 | -0.0633 ± 0.0078 | 33,269 | 35,236 | 259.9 | 389.9 | 779.5 (800) | 7.79 | 0.9252 ± 0.0862 | 0.924 | 0.926 | 0.432 | 0.098 (0.042) | hayır | evet |
| 100 | 10 | 0.8175 ± 0.0153 | -0.0389 ± 0.0066 | 66,640 | 74,482 | 520.6 | 780.9 | 926.1 (1000) | 9.26 | 0.9061 ± 0.0972 | 0.890 | 0.923 | 0.420 | 0.118 (0.042) | hayır | evet |
| 100 | 12 | 0.8346 ± 0.0125 | -0.0218 ± 0.0054 | 101,845 | 114,210 | 795.7 | 1193.5 | 1031.9 (1200) | 10.32 | 0.9218 ± 0.0685 | 0.926 | 0.918 | 0.536 | 0.137 (0.042) | hayır | evet |
| 100 | None | 0.8558 ± 0.0132 | -0.0006 ± 0.0029 | 168,586 | 202,940 | 1317.1 | 1975.6 | 1188.5 (2579) | 11.88 | 0.8980 ± 0.0621 | 0.854 | 0.944 | 0.596 | 0.096 (0.042) | evet | evet |
| 300 | 4 | 0.6702 ± 0.0194 | -0.1862 ± 0.0137 | 9,267 | 9,276 | 72.4 | 108.6 | 1199.9 (1200) | 4.00 | 0.9279 ± 0.0731 | 0.927 | 0.929 | 0.707 | 0.060 (0.042) | hayır | evet |
| 300 | 6 | 0.7445 ± 0.0165 | -0.1120 ± 0.0072 | 34,828 | 35,528 | 272.1 | 408.1 | 1792.9 (1800) | 5.98 | 0.8912 ± 0.0900 | 0.886 | 0.897 | 0.120 | 0.039 (0.042) | hayır | hayır |
| 300 | 8 | 0.7909 ± 0.0148 | -0.0655 ± 0.0073 | 99,427 | 104,826 | 776.8 | 1165.2 | 2336.7 (2400) | 7.79 | 0.9091 ± 0.1124 | 0.904 | 0.915 | 0.286 | 0.118 (0.042) | hayır | evet |
| 300 | 10 | 0.8188 ± 0.0134 | -0.0376 ± 0.0056 | 199,356 | 218,442 | 1557.5 | 2336.2 | 2777.7 (3000) | 9.26 | 0.8949 ± 0.1021 | 0.883 | 0.907 | 0.458 | 0.176 (0.042) | hayır | evet |
| 300 | 12 | 0.8349 ± 0.0117 | -0.0216 ± 0.0060 | 302,240 | 340,848 | 2361.3 | 3541.9 | 3098.0 (3600) | 10.33 | 0.9574 ± 0.0426 | 0.954 | 0.961 | 0.539 | 0.039 (0.042) | hayır | evet |
| 300 | None | 0.8564 ± 0.0122 | +0.0000 ± 0.0000 | 504,201 | 612,946 | 3939.1 | 5908.6 | 3571.6 (7737) | 11.91 | 0.9237 ± 0.0352 | 0.897 | 0.952 | 0.633 | 0.078 (0.042) | evet | evet |

Referans (Faz A B2 = 300 ağaç, sınırsız derinlik): macro-F1 0.8564 ± 0,0122, AUROC 0.9237 ± 0,0352, FAR 0,0784; düğüm 504,201 (kat ort.), 612,946 (tam veri) → 8 B/düğümle 3.85 MB, 12 B/düğümle 5.77 MB; pencere başına 3572 karşılaştırma.

## Önceden yazılmış seçim kuralı (kural sabitleri aynen)
Kural: macro-F1 ≥ 0,856 − 0,02 = **0.836** **ve** AUROC ≥ 0,924 − 0,03 = **0.894** olan en küçük (bayt) yapılandırma; yoksa "sığan yapılandırma yok". Bayt sırası 8 ve 12 B/düğümde aynıdır (düğüm sayısı); eşitlikte az karşılaştırma.
- Yalnız macro-F1 koşulunu sağlayan: 5 / 30; yalnız AUROC koşulunu sağlayan: 20 / 30; ikisini birden: **5 / 30**.
- Kuralı sağlayanlar (küçükten büyüğe):

| ağaç | derinlik | macro-F1 | AUROC D3 | düğüm (kat ort.) | B8 KB | B12 KB | kars. |
|---|---|---|---|---|---|---|---|
| 10 | None | 0.8461 | 0.9370 | 16,760 | 130.9 | 196.4 | 121.1 |
| 25 | None | 0.8518 | 0.9396 | 41,955 | 327.8 | 491.7 | 300.0 |
| 50 | None | 0.8565 | 0.9259 | 84,037 | 656.5 | 984.8 | 595.1 |
| 100 | None | 0.8558 | 0.8980 | 168,586 | 1317.1 | 1975.6 | 1188.5 |
| 300 | None | 0.8564 | 0.9237 | 504,201 | 3939.1 | 5908.6 | 3571.6 |

- **Seçilen: n=10, d=None** — macro-F1 0.8461 ± 0.0146 (B2'ye göre -0.0103 ± 0.0033), AUROC 0.9370 ± 0.0425 (B2: 0.9237), FAR 0.039, tespit 0.535 (B2: 0,078 / 0,633); 16,760 düğüm (kat ort.; tam veri 20,460) = 130.9 KB (8 B) / 196.4 KB (12 B) [tam veri: 159.8 / 239.8 KB]; pencere başına 121.1 karşılaştırma (en kötü 260).
- Seçilen yapılandırmanın sığdığı çizgiler: kat ortalaması 8 B → 256 KB, 512 KB, MSS veri RAM 192 KB, DSP L2 256 KB, L3 768 KB, toplam 1,75 MB; 12 B → 256 KB, 512 KB, DSP L2 256 KB, L3 768 KB, toplam 1,75 MB; tam-veri modeli 8 B → 256 KB, 512 KB, MSS veri RAM 192 KB, DSP L2 256 KB, L3 768 KB, toplam 1,75 MB; 12 B → 256 KB, 512 KB, DSP L2 256 KB, L3 768 KB, toplam 1,75 MB. **128 KB çizgisine sığmaz** (kat ortalaması 8 B bile 130.9 KB).
- B2 ile yan yana: B2 0.8564 / 0.9237 ile 3,939–5,909 KB; seçilen yapılandırma ≈ 30 kat daha az düğümle 0.8461 / 0.9370. **B2 birincil model olarak kalır.**

## Hangi bellek kovasına ne sığar (düğüm kat ortalaması; bayt = düğüm × B/düğüm)
Her kova için kovaya sığan yapılandırmalar arasında en yüksek macro-F1 ve en yüksek AUROC. Son sütun: kovaya sığan ve kuralın iki koşulunu da sağlayan yapılandırma sayısı.

| bellek çizgisi | B/düğüm | sığan yapılandırma sayısı | en yüksek macro-F1 (yapılandırma) | en yüksek AUROC (yapılandırma) | kural koşullarını sağlayan sığan sayısı |
|---|---|---|---|---|---|
| 128 KB | 8 | 14 | 0.8226 (n=10, d=12) | 0.9279 (n=300, d=4) | 0 |
| 128 KB | 12 | 12 | 0.8226 (n=10, d=12) | 0.9279 (n=300, d=4) | 0 |
| 256 KB | 8 | 17 | 0.8461 (n=10, d=None) | 0.9373 (n=25, d=12) | 1 |
| 256 KB | 12 | 16 | 0.8461 (n=10, d=None) | 0.9370 (n=10, d=None) | 1 |
| 512 KB | 8 | 22 | 0.8518 (n=25, d=None) | 0.9396 (n=25, d=None) | 2 |
| 512 KB | 12 | 21 | 0.8518 (n=25, d=None) | 0.9396 (n=25, d=None) | 2 |
| MSS veri RAM 192 KB | 8 | 16 | 0.8461 (n=10, d=None) | 0.9370 (n=10, d=None) | 1 |
| MSS veri RAM 192 KB | 12 | 14 | 0.8226 (n=10, d=12) | 0.9279 (n=300, d=4) | 0 |
| DSP L2 256 KB | 8 | 17 | 0.8461 (n=10, d=None) | 0.9373 (n=25, d=12) | 1 |
| DSP L2 256 KB | 12 | 16 | 0.8461 (n=10, d=None) | 0.9370 (n=10, d=None) | 1 |
| L3 768 KB | 8 | 24 | 0.8565 (n=50, d=None) | 0.9396 (n=25, d=None) | 3 |
| L3 768 KB | 12 | 22 | 0.8518 (n=25, d=None) | 0.9396 (n=25, d=None) | 2 |
| toplam 1,75 MB | 8 | 28 | 0.8565 (n=50, d=None) | 0.9396 (n=25, d=None) | 4 |
| toplam 1,75 MB | 12 | 26 | 0.8565 (n=50, d=None) | 0.9396 (n=25, d=None) | 3 |

## Dikkat edilecekler (koşudan önce yazılan + sonuçtan çıkan)
- **B/düğüm varsayımı:** 8 B = özellik indeksi + eşik + iki çocuk işaretçisi (yaprakta sınıf); 12 B = yaprakta sınıf olasılıklarını da taşıyan daha cömert bir düzen. Gerçek bir C uygulaması farklı olabilir (ör. yapraklarda 6 sınıf olasılığı için ağaç başına ek dizi; ağaç sayısı büyükse ofset tabloları). Sıralama (hangi yapılandırma daha küçük) bu varsayımlardan etkilenmez; mutlak bayt etkilenir.
- **Oy sayımı:** burada `predict_proba` ortalamasının argmax'ı kullanıldı (Faz A B2 ile aynı); çipte sınıf oylarının sayılması ya da yaprak olasılıklarının toplanması gerekir. Özellik çıkarımı (kare istatistikleri, 5 karelik pencere) ve nokta bulutu tamponları bu boyuta dahil değildir.
- **Anomali AUROC'u gürültülüdür:** kat sd'si 0,04–0,19; ~50 normal döngü ve 5 kat. Küçük AUROC farkları (≈ 0,03) kat sd'sinin içindedir; AUROC'un düğüm sayısıyla tekdüze artmaması (aşağıdaki şekil) buna uyar. FAR'ın SE'si ≈ 0,04.
- Çip belleği: bkz. yukarıdaki uyarı. Hiçbir çip üzerinde ölçüm, derleme ya da gecikme testi yapılmadı; "sığar" yalnız boyut hesabıdır.
- Kural sabitleri (0,856 / 0,924) Faz A B2'nin yuvarlanmış değerleridir (B2: 0,8564 / 0,9237); kural bu yuvarlanmış sabitlerle aynen uygulandı.

## Şekil
`01_tiny_forests.png`: üst sıra macro-F1, alt sıra D3 AUROC (tüm anomali); sol sütun 8 B/düğüm, sağ sütun 12 B/düğüm; x ekseni model baytı (log). Renk = ağaç sayısı, işaret = derinlik. Kesikli yatay: Faz A B2 (0,8564 / 0,9237); noktalı yatay: kural eşikleri (0,836 / 0,894). Siyah dikey: 128 / 256 / 512 KB; kırmızı dikey: çip (MSS veri RAM 192 KB, DSP L2 256 KB, L3 768 KB, toplam 1792 KB). Siyah 256 KB çizgisi ile kırmızı DSP L2 256 KB çizgisi aynı x'tedir (üst üste biner). Not: rapordaki hesaplanan sayılar noktalı, elle yazılan Faz A sabitleri virgüllü yazılmıştır; aynı nicelikler.
Bayt sırası kat ortalaması düğüm sayısıyla ve tam-veri modeli düğüm sayısıyla aynıdır (n=10/None 20.460 < n=25/None 50.479 < ...), yani seçim hangi bayt tanımı kullanılsa değişmez.

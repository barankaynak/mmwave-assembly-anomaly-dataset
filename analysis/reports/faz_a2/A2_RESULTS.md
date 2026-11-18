# Faz A2 sonuçları — ikincil analizler (2026-11-04)

Ön-kayıt: `faz_a2/DESIGN_A2.md`; sapmalar orada tarihli olarak kayıtlı.
- **Birincil sonuçlar Faz A'dır.** A2'nin her sayısı ikincildir ve bir Faz A gözlemine cevap verir.
- Koşular yerel makinede yapıldı (Ryzen 7 3700X, GTX 1650 SUPER).
- İki adversarial doğrulayıcı bütün analizleri yeniden üretti; çıktılar bayt bayt aynı (`verify_a.md`, `verify_b.md`). Tek engelleyici bulgu, rapordaki elle yazılmış bir yuvarlama hatasıydı (−0,002 → −0,003); düzeltildi ve yeniden doğrulandı.
- Faz A dosyalarının hiçbiri değişmedi (md5 kontrolü).

## A2.0 — Yerel = uzak: GEÇTİ
- Heatmap farkı 0 (600 kare); katlar bayt bayt aynı.
- B2 S1 kat 0 ve 1'i yeniden eğitildi; tahminler %100 aynı.
- Yeni modül `faz_a2/a2_cycles.py`, Faz A 04'ün bütün skorlarını 7e-15 farkla yeniden üretiyor.

## A2.2 — Sızıntı: yalnız derin modelde, ve model sıralamasını **ters çeviriyor** (en önemli A2 bulgusu)
Ölçüm: sızıntı etkisi = rastgele pencere bölmesi (S0, 5 tohum) − döngü bazlı bölme (S1, 5 kat), puan cinsinden, Welch %95 GA.

| model | S0 accuracy | S1 accuracy | etki (acc) | etki (macro-F1) |
|---|---|---|---|---|
| B1 en-yakın-merkez | 61,6 ± 0,7 | 61,3 ± 1,2 | +0,3 [−1,2; +1,9] | +0,8 [−0,9; +2,5] |
| **B2 özellik + RF** | 88,9 ± 0,3 | 88,4 ± 1,2 | **+0,5 [−1,0; +1,9]** | +0,7 [−0,8; +2,1] |
| **CNN-GRU** | 91,4 ± 0,8 | 86,9 ± 1,3 | **+4,4 [+2,8; +6,0]** | **+5,6 [+3,7; +7,5]** |

- Rastgele bölme **yalnız CNN-GRU'yu** şişiriyor. B1 ve B2'de ölçülebilir bir etki yok; GA'lar 0'ı içeriyor.
- **Bu yüzden sıralama tersine dönüyor:**
  - Rastgele bölmede CNN-GRU önde: 91,4 vs 88,9, +2,5 puan.
  - Sızıntısız bölmede B2 önde: 88,4 vs 86,9; macro-F1 eşli kat farkı +1,6, sd 0,6.
  - Yani eski protokol yalnız sayıları değil, **hangi modelin kazandığını** da değiştiriyor.
- Aynı donanımda kalan tek karşılaştırma S0 tohum 42 (uzak 3090). Orada etki +3,2 / +3,7 puan; sıralama tersine dönmesi orada da var (CNN-GRU S0 tohum 42 macro-F1 0,878 > B2 S0 5 tohum ortalaması 0,863).
- **Uyarılar:**
  - CNN-GRU'nun 5 tohumluk S0 ortalaması 1 uzak + 4 yerel GPU koşusunu karıştırıyor.
  - S0'ın iç doğrulaması da rastgele pencerelerle yapıldığı için mekanizmanın parçası; ölçülen etki "toplam" etkidir.
  - Etkinin nedeni (model kapasitesi mi, girdi türü mü) bu deneyle ayrılamıyor.
- **Makale cümlesi:** "Rastgele pencere bölmesi derin modelin doğruluğunu ~4 puan şişiriyor, sığ modeli etkilemiyor ve iki modelin sıralamasını tersine çeviriyor."

## A2.1 — Conformal eşik: eşitlikler ve kalibrasyon çözünürlüğü
B2 × pred, D3 (Bonferroni korunuyor), α = 0,10:

| yöntem | eğitim / kalibrasyon döngüsü | normal FAR | sıra tespiti (135 / 145) | tempo tespiti | tüm anomali | AUROC |
|---|---|---|---|---|---|---|
| Faz A standart (birincil) | ~20 / ~20 | 0,078 | 0,36 (0,25 / 0,48) | 0,93 | 0,63 | 0,924 |
| düzgünleştirilmiş p | ~20 / ~20 | 0,087 [0,078; 0,098] | 0,57 (0,45 / 0,72) | 0,93 | 0,75 | 0,924 |
| cross-conformal | ~30 / ~40 | 0,059 | 0,44 (0,26 / 0,65) | 0,93 | 0,68 | 0,944 |
| cross + düzgünleştirilmiş | ~30 / ~40 | 0,091 [0,059; 0,118] | **0,60 (0,38 / 0,86)** | **0,95** | **0,77** | **0,945** |

- α = 0,05'te Faz A standardı hiç alarm veremiyordu. Cross + düzgünleştirilmiş: FAR 0,041, sıra 0,40, tempo 0,85.
- **Karıştırıcı:** Cross-conformal'da hem eğitim (~30) hem kalibrasyon (~40) seti büyük, iki etki ayrılamıyor. Eğitim setini sabit tutan tek karşılaştırma standart → düzgünleştirilmiş.
- Diğer hücreler, cross + düzgünleştirilmiş, α = 0,10 (FAR / sıra / tempo):
  - B2 × tof: 0,093 / 0,64 / 0,89
  - ToF oracle: 0,086 / 1,00 / 0,84
  - CNN-GRU × pred: 0,088 / 0,58 / 0,66
- ToF oracle ve CNN-GRU'da yalnız düzgünleştirilmiş p'nin FAR'ı 0,12; α'nın üstünde ama 2SE içinde. Faz A'nın "geçerli" kuralı (|FAR − α| ≤ 2SE) bu örneklem büyüklüğünde gevşek.
- **Makale cümlesi (ikincil):** "Eşitliğe duyarlı cross-conformal kalibrasyonla (~40 normal döngü) iki kanallı dedektör, α = 0,10'da %9 döngü yanlış alarmıyla sıra sapmalarının %60'ını, tempo sapmalarının %95'ini yakalıyor."

## A2.3 — Etiket duyarlılığı: s05'teki düşük skorun bir kısmı etiket artefaktı
- s05'te ±3 kare yumuşatma (L0), 49 döngünün yalnız 9'unda Box5 bırakıyor. 30 Hz ToF dokunuş etiketinde (L3) 49'unun hepsinde Box5 var. Box5 kare sayısı 37 → 179.
- **B2, s05 macro-F1:** 0,758 (L0) → 0,833 (L3), +0,075 [+0,035; +0,120]. Box5 F1 0,35 → 0,69. Kazancın tamamı Box5'ten geliyor.
- Temiz kontrolle (s03/s04'te de kısmi silme var) fark-içinde-fark +0,058.
- Aynı yön diğer modellerde de görülüyor: CNN-GRU +0,090, CNN-only +0,144, B1 +0,134.
- Modeller, L0 etiketleriyle eğitildikleri halde s05'te L0'ın içerdiğinden çok daha fazla Box5 tahmin ediyor (B2: 110 kare, L0: 37).
- **Uyarılar:**
  - Kazancın büyüklüğü etiket tanımına bağlı. B2'de L1 / L2 0,879 / 0,881, yani L3'ten yüksek.
  - L1–L3 aynı ToF sensörlerinden geliyor; "bağımsız" yalnız "yumuşatılmamış" anlamında.
  - Eğitim etiketleri de aynı silmeyi taşıyordu.
- **Makale cümlesi:** "s05'in düşük skoru kısmen etiket artefaktıdır: ±3 kare yumuşatma kısa Box5 dokunuşlarını siliyor. 30 Hz dokunuş etiketiyle B2'nin s05 macro-F1'i 0,76'dan 0,83'e çıkıyor."

## A2.4 — Devreye alma maliyeti
**(a) Anomali, B2 × pred, D3 AUROC:**
- 5 / 10 / 15 / 20 normal eğitim döngüsü için 0,893 / 0,920 / 0,928 / 0,928. Bunlara ek olarak 20 kalibrasyon döngüsü kullanılıyor.
- Ortalama normal döngü ≈ 5 s. Yani ~10 eğitim döngüsü (≈ 0,8 dk) + 20 kalibrasyon döngüsü (≈ 1,7 dk) ≈ **2,5 dakika normal kayıt.**
- **Uyarılar:**
  - Bu "yeterlilik" ölçütü yalnız AUROC'a bakıyor. Standart p ile α = 0,10'da tespit zayıf (%32–44).
  - Tekrarlar aynı 51 döngüden yeniden örnekleme; yeni bir istasyona ya da operatöre genellemeyi ölçmüyor.
  - (a) ve (b) dakika hesapları farklı süre tanımlarına dayanıyor; makaleye yazılmadan önce tek tanıma indirilmeli (doğrulayıcı notu m2).

**(b) Durum tahmini, B2, S1 macro-F1:**

| oturum başına etiketli döngü | 1 | 2 | 5 | 10 | 20 | hepsi |
|---|---|---|---|---|---|---|
| macro-F1 | 0,713 | 0,770 | 0,810 | **0,829** | 0,842 | 0,856 |

- 10 döngü, tam verinin %95'ine ulaşıyor; bu oturum başına ~0,8 dk kayıt.
- ToF etiketi bedava olduğu için etiketleme maliyeti = kayıt süresi.

## A2.5 — CPU gecikmesi (Ryzen 7 3700X, tek iş parçacığı, 3 koşu)
| bileşen | medyan (ms) |
|---|---|
| B2 (özellik + RF-300) | **20,1** (özellik çıkarımı yalnız 0,54) |
| CNN-GRU, 5 karelik pencere | **7,2** (akışlı kullanımda 2,8) |
| Heatmap kurulumu, kare başına | 0,04 |
| B1 | 0,009 |

- İkisi de 100 ms'lik kare aralığının çok altında.
- B2'nin süresi neredeyse tamamen 300 ağaçlık RF'den geliyor. CNN-GRU tek iş parçacığında B2'den hızlı.
- Bu B2 sayısı (20 ms), Faz A'daki uzak ölçümden (13,3 ms, ortalama) yüksek. Makine farklı, arka plan yükü %22,5; neden ayrıca araştırılmadı.
- Eski makalenin 12,6 ms'si Keras ile, Ryzen 7 5800X'te ölçülmüştü.

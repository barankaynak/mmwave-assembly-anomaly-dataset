# Faz A sonuçları — sızıntısız protokol + basit taban kıyası (2026-11-04)

Tasarım `faz_a/DESIGN.md`'de sonuçlardan önce sabitlendi; sapmalar orada tarihli olarak kayıtlı. Ayrıntılı tablolar:
- `03_state.md`: durum tahmini;
- `04_anomaly.md`: döngü anomalisi;
- `05_old.md`: eski boru hattı.

İki adversarial doğrulayıcı (`verify_state.md`, `verify_anomaly.md`) bütün aşamaları yeniden koştu. Çıktılar bayt bayt aynı çıktı ve **engelleyici bulgu yok**.

## Hüküm
**Önceden yazılmış ölçüte göre "basit taban yeterli".** Makalenin ana modeli **B2** oluyor:
- 21 el yapımı nokta bulutu özelliği + Random Forest (300 ağaç).
- Özellikler: 7 kare istatistiği (SNR-ağırlıklı merkez x/y, nokta sayısı, SNR toplamı, ortalama |v|, x/y yayılımı) × 5 karelik pencerede son değer / ortalama / eğim.

Diğer modellerin yeni rolü:
- **CNN-GRU:** derin öğrenme karşılaştırması olarak kalıyor.
- **CNN-only:** ablasyon olarak kalıyor.
- **TFT:** çıkarılıyor; önemsiz bir Markov tahmincisini bile geçemiyor.

Anomali tarafında ana yöntem **D3**: Markov sıra skoru + adım bazlı süre skoru. Eşik conformal kalibrasyonla ve yalnız normal kalibrasyon döngülerinden belirleniyor.

## 1. Durum tahmini (kare → {none, Box1..5})

**S1 — 5 kat, döngü bazlı (ana)**

| model | accuracy | macro-F1 |
|---|---|---|
| **B2 özellik + RF** | **0,884 ± 0,012** | **0,856 ± 0,012** |
| CNN-GRU | 0,869 ± 0,013 | 0,841 ± 0,014 |
| CNN-only | 0,776 ± 0,037 | 0,732 ± 0,035 |
| B1 en-yakın-merkez | 0,613 ± 0,012 | 0,527 ± 0,014 |

**S2 — oturum dışarıda (LOSO, 6 kat; oturumlar arası genelleme)**

| model | accuracy | macro-F1 |
|---|---|---|
| **B2** | **0,868 ± 0,046** | **0,829 ± 0,094** |
| CNN-GRU | 0,842 ± 0,033 | 0,803 ± 0,078 |
| B1 | 0,615 ± 0,020 | 0,568 ± 0,076 |

- Eşli kat farkı (CNN-GRU − B2, macro-F1): S1'de −0,016 (sd 0,006), S2'de −0,025 (sd 0,023). Yani B2 her iki bölmede de önde.
- **Alt kümelerde (S1 accuracy) CNN-GRU'nun kazandığı yer yok:**

  | alt küme | CNN-GRU | B2 |
  |---|---|---|
  | kutu kareleri | 0,894 | 0,933 |
  | geçiş kareleri | 0,818 | 0,834 |
  | none kareleri | 0,852 | 0,850 (eşit) |
  | s01 (macro-F1) | 0,774 | 0,784 |
- Zamansal bilgi önemli. CNN-only'den CNN-GRU'ya geçince macro-F1 +0,109 artıyor. Ama bu bilgi B2'deki basit pencere istatistikleriyle (son değer / ortalama / eğim) de yakalanıyor.
- **Gecikme:** B2 tek iş parçacığında, özellik çıkarımı dahil pencere başına 13–15 ms CPU. Kare aralığı 100 ms, yani gerçek zamanlı çalışabilir. B2'nin parametresi yok; CNN-GRU 1,92 M parametre.

## 2. Sızıntı etkisi ve eski %93,66

| koşul | accuracy | macro-F1 |
|---|---|---|
| Keras, eski makale (S0 rastgele pencere, 3.250 test) | 0,9366 | — |
| PyTorch port, S0 (tek koşu) | 0,901 | 0,878 |
| PyTorch port, S1 (sızıntısız) | 0,869 ± 0,013 | 0,841 ± 0,014 |

- Yayımlanan %93,66, **rastgele pencere bölmesiyle** ölçülmüş bir sayı. Aynı mimarinin sızıntısız (döngü bazlı) değeri **%86,9 ± 1,3**.
- Fark iki ayrı kaynaktan geliyor ve ikisi ayrı verilmeli: sızıntı (S0_pt − S1_pt) +0,032 accuracy / +0,037 macro-F1; port farkı (Keras − S0_pt) +0,036 accuracy. İki fark aynı büyüklükte, bu yüzden "sızıntı 3,2 puan şişirdi" diye tek sayı vermek yanlış olur.
- S0 tek koşu (binom SE 0,005). S1'de döngü bootstrap %95 GA 0,859–0,880.

## 3. Döngü düzeyi anomali tespiti

Kurulum:
- Normal döngüler s02'den alındı: eğitim 2 kat, kalibrasyon 2 kat, test 1 kat (çapraz uydurma). Test edilen normal döngü sayısı ≈ 50.
- Anomaliler: s03 (sıra 1-3-5), s04 (sıra 1-4-5), s05 (hızlı), s06 (yavaş). Her biri ≈ 50 döngü ve her katta test ediliyor.
- Dağıtıma uygun hücre: **B2 × pred** (döngüler tahmin edilen Box1 girişlerinden kesiliyor; çalışma anında ToF yok).

**Ana sonuç: B2 × pred, D3, α = 0,10 (birincil çalışma noktası)**

| | normal (yanlış alarm) | seq135 | seq145 | fast | slow | tüm anomali |
|---|---|---|---|---|---|---|
| alarm oranı | **0,078** (SE 0,042, geçerli) | 0,254 | 0,475 | 0,921 | 0,929 | 0,633 |
| AUROC (kat ort.) | — | 0,879 | 0,918 | 0,947 | 0,956 | **0,924 ± 0,035** |

- **Döngü düzeyinde yanlış alarm oranı %7,8.** Eşik, ToF'suz çalışma anında kalibrasyon verisiyle seçiliyor; test verisine bakılmıyor.
  - Eski makalenin %29,3'ü ile **doğrudan kıyaslanmaz.** O sayı geçiş düzeyindeydi, sızıntılı tahminle ve eğitim = test normal veriyle ölçülmüştü; aynı tanımla yeniden hesaplanınca %67,1 çıkıyor.
  - Dürüst kıyas döngü düzeyinde yapılır: yeni dedektör AUROC 0,92, eski boru hattı 0,51–0,70.
- Tempo anomalileri güçlü yakalanıyor (%92–93).
- Sıra anomalilerinde AUROC yüksek (0,88–0,92) ama α = 0,10'daki tespit oranı düşük (%25–48). Neden yapısal:
  - D3 = 2·min(p1, p2) (Bonferroni) ve n_cal ≈ 20'de ulaşılabilecek en küçük p ≈ 0,048.
  - Bu yüzden D3 yalnız kalibrasyondaki **tüm** skorları aşan döngüye alarm verebiliyor.
  - D1 tek başına (B2 × pred) seq135'te %60, seq145'te %80 yakalıyor; yanlış alarm %5,9.
  - Düzeltme Faz A2'de yapılacak (daha büyük kalibrasyon seti ya da Bonferroni'siz bir birleştirme). Şimdi ayar yapılmadı.
- **α = 0,05'te D3 hiçbir hücrede alarm veremiyor** (eşik 0,025 < en küçük p). Bu açıkça yazılmalı.
- ToF sınırlarıyla (B2 × tof) D3: yanlış alarm 0,059, tespit 0,590, AUROC 0,944.

**CNN-GRU ile kıyas (aynı hücreler):**
- CNN-GRU × pred, D3: AUROC 0,879, yanlış alarm 0,082, tespit 0,285. seq135'te tespit 0,021, yani B2'den belirgin biçimde kötü.
- Tahmin edilen döngü kesimi B2'de daha temiz:
  - B2'nin oturum başına döngü sayıları referansa yakın: 51/56/48/48/51, referans 51/57/49/49/52. CNN-GRU s06'yı 59'a bölüyor.
  - Beklenen ziyaret dizisinin oranı:

    | oturum | s02 | s03 | s04 | s05 | s06 |
    |---|---|---|---|---|---|
    | B2 | 0,92 | 0,96 | 0,98 | 0,73 | 0,98 |
    | CNN-GRU | 0,88 | 0,95 | 0,91 | 0,59 | 0,73 |

**Eski boru hattı (TFT + Z-skoru) sızıntısız protokolde başarısız:**
- **TFT öğrenmiyor.** Normal test geçişlerinde Denk. 9 MAE'si 1,178. Eğitim ortalaması tahmincisi 1,444 alıyor, önemsiz Markov-mod tahmincisi 0,830. TFT 4 hücrenin hiçbirinde Markov-mod'u geçmiyor.
- Bunun nedeni veri azlığı: 10 uydurmanın 8'inde en iyi doğrulama kaybı sabit tahminin kaybına eşit.
- **Döngü düzeyi AUROC 0,51–0,70**, yani şans düzeyine yakın. Hızlı (fast) anomalide TFT skoru ters yönü gösteriyor (AUROC 0,13). Karşılaştırma için: aynı veride D3 0,88–0,97 (durum kaynağına ve segmentasyona göre).
- Eski "% geçiş eşik üstü" metriği aynı tanımla yeniden hesaplandı (ToF oracle, held-out normal): VEYA birleşiminde %67,1; yayımlanan değer %29,3.
  - Bu iki sayı bire bir karşılaştırılamaz: yayımlanan değer sızıntılı CNN-GRU tahminleriyle ve eğitim = doğrulama = test normal veriyle hesaplanmıştı.
  - Döngü düzeyinde sabit eşikle çıkan yanlış alarm (0,88–1,00) bir max-birleştirme artefaktı; rapor edilmemeli.

## 4. Önceden yazılmış karar ölçütleri — harfi harfine

| ölçüt | sonuç |
|---|---|
| 1. B1 veya B2'nin S1 macro-F1'i CNN-GRU'ya 2 puan yakın | **Sağlandı.** B2 1,6 puan **önde**. |
| 2. D3 AUROC eşli farkı (B2 − CNN-GRU): %95 GA 0'ı içeriyor ya da \|fark\| ≤ 0,05, her anomali türünde (8 hücre: 2 segmentasyon × 4 tür) | **8 hücrenin 7'sinde sağlandı.** pred × fast'ta sağlanmadı: +0,058, GA [+0,006, +0,110]. Bunun tek nedeni B2'nin **anlamlı biçimde daha iyi** olması. |
| "Aksi halde" dalı: CNN-GRU'nun kazandığı alt küme raporlanır, CNN-GRU ana model kalır | Uygulanamaz, çünkü CNN-GRU hiçbir alt kümede kazanmıyor. |
| Conformal geçerlilik: yanlış alarm oranı α ± 2·SE içinde | D0/D2/D3 bütün hücrelerde geçerli. D1 3 hücrede (ToF×tof, ToF×pred, CNN-GRU×pred) **muhafazakâr** (FAR 0): eşit skorlar p'yi büyütüyor. Protokol bunu öngörmüştü; hata değil. |

Ölçüt 2 bir eşdeğerlik testi olarak yazılmıştı ve tabanın CNN-GRU'yu açıkça geçmesini öngörmüyordu. Bu yorum DESIGN.md "Sapmalar"a tarihli olarak yazıldı.

## 5. Uyarılar (makalede açıkça yazılmalı)
1. **s05'te Box5 etiketten siliniyor.**
   - ±3 kare mod yumuşatması, 49 hızlı döngünün 40'ında kısa Box5 dokunuşunu siliyor. Bu bir etiket sorunu, bu yüzden s05 durum skorları kısmen hatalı referansla ölçülüyor: B2 mF1 0,758, CNN-GRU 0,726.
   - Oracle'ın D1-fast AUROC'u (0,93) bu yüzden bir artefakt; oracle tempo anomalileri için gerçek üst sınır değil.
   - B2 × pred s05 döngülerinin 35/48'inde 1-2-5 dizisini buluyor. Yani tahminci, etiketin kaybettiği Box5'i muhtemelen yakalıyor.
2. **Sızıntı tahmini port farkıyla karışıyor** (bkz. §2). S0 tek koşu.
3. **D3 / Bonferroni, n_cal ≈ 20'de yapısal olarak zayıf** (bkz. §3). Rapor α = 0,10'da verilmeli; α = 0,05 sonucu (sıfır tespit) açıkça yazılmalı.
4. **Eski boru hattı için doğru kıyas metrikleri:** geçiş düzeyi %67 vs %29,3 (bire bir değil) ve TFT MAE vs Markov-mod. Döngü düzeyi sabit eşik FAR'ı kullanılmamalı.
5. **S1'de ufak bir pencere örtüşmesi var.** Test pencerelerinin 2.091'i (%12,9) bir eğitim penceresiyle en az bir kare paylaşıyor (döngü sınırlarında). Bunlar dışlanınca fark ≤ 0,004 ve sıralama değişmiyor.
6. **AUROC çözünürlüğü kaba.** Katta ~10 normal döngü var ve D3'ün p değerleri ayrık; AUROC SE ≈ 0,03–0,05. 0,02'lik farklar anlamlı değil.
7. **Tek operatör, tek görev sınırlaması değişmedi.** S2 (LOSO) yalnız oturumlar arası genellemeyi gösteriyor, kişiler arası genellemeyi değil.

## 6. Makale için sonuçları
- Başlıktaki **"Deep Temporal Models" artık makaleyi tanımlamıyor.** Ana model derin değil, TFT de çıkıyor.
- Yeni çerçeve: **düşük maliyetli, gizliliği koruyan mmWave algılama + sızıntısız değerlendirme protokolü + kalibre edilmiş alarm eşiği + açık veri seti.** CNN-GRU, "derin model şart değil" bulgusunu destekleyen dürüst bir kıyas olarak kalıyor.
- Bu çerçeve, önceki retlerin gerekçelerine doğrudan cevap veriyor:
  - SOTA ve taban kıyası yok → artık var;
  - yüksek yanlış alarm → döngü düzeyinde %7,8, kalibrasyonla kontrol ediliyor. Eski %29,3 ile doğrudan kıyaslanmaz (bkz. §3);
  - abartılı iddialar → sızıntısız sayılar;
  - kategorik durumun ordinal kullanılması (Denk. 9) → TFT çıkıyor, Markov olasılığı geliyor.
- Kapanmayan tek gerekçe tek operatör sınırlaması.

## 7. Faz A2'ye kalanlar
- Sıra anomalisi tespit oranı: daha büyük kalibrasyon seti (ör. s02 + yeni normal kayıt) ya da Bonferroni'siz bir birleştirme (ör. Fisher ya da öğrenilmiş birleştirme), önceden kayıtla.
- Box5'i silmeyen etiket yumuşatması (30 Hz ToF'tan doğrudan pencere etiketi) ve s05 skorlarının yeniden hesabı.
- CNN-GRU'nun CPU gecikmesi (PyTorch port) ve B2'nin gecikme aralığı.
- Sentetik anomali enjeksiyonu, sağlamlık; mümkünse ek operatör kaydı.

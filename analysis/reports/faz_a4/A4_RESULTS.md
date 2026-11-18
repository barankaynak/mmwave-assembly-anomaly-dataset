# Faz A4 sonuçları — hakemin M1–M4 maddeleri (2026-11-05)

**Plan:** `faz_a4/DESIGN_A4.md`. Plan, sonuçlar görülmeden önce yazıldı; sapmalar aynı dosyanın sonundaki "Sapmalar" bölümünde.

**Doğrulama:** İki bağımsız doğrulayıcı vardı: `verify_a.md` (A4.1–A4.2) ve `verify_b.md` (A4.3–A4.4).
- Bütün betikler sıfırdan yeniden çalıştırıldı ve çıktılar bayt düzeyinde aynı çıktı.
- 25'ten fazla anahtar sayı bağımsız kodla yeniden hesaplandı.

**Sonuç:** Fable'ın ikinci tur incelemesine göre **dört madde de giderildi**. Dergi için öneri **minor revision** (`paper/review/referee_round2.md`).

## Tek satırlık sonuçlar
| Madde | Soru | Sonuç |
|---|---|---|
| **M1** | Anomali tipi oturumla karışıyor mu? | Ölçülebilir büyüklükte bir oturum etkisi yok. Süre kanalı kontrolü sıra oturumlarında %9,4 [3,8; 15,2], normalde %7,8 [3,1; 18,5]. Test ancak ~10 puanlık bir fazlalığı yakalayabilir. Onset öncesi 34 alarmın 20'sinde tahmin edilen durum zaten yanlış kutudaydı. Kalan 14 alarm 3 döngüden geliyor ve yalnız birinde oturuma özgü zamanlama dışlanamıyor. |
| **M2** | Etiket düzleştirme hızlı tempo tespitini şişiriyor mu? | Hızlı tempoda 221 D3 alarmının 169'u yalnız süre kanalından, 16'sı her iki kanaldan, 36'sı yalnız sıra kanalından geliyor. Hızlı döngü çiftlerinin %27'sinde Box 5 hiç görülmüyor (13 döngü); oysa 30 Hz ToF kaydı Box 5'i 49 hızlı döngünün 47'sinde gösteriyor. Yalnız süre kanalıyla tempo tespiti %89,9 [84,0; 94,7], FAR %7,8 (ikincil analiz). Oracle'ın yavaş tempo tespitinin %60'ta kalması, iki katta kalibrasyona düşen tek bir 25,0 s'lik döngüden kaynaklanıyor (betimsel, test edilmedi). |
| **M3** | Sızıntı etkisi aynı donanımda ne kadar? Keras orijinali yeniden eğitildi mi? | Önceden yazılan kurala göre başlık değeri iki aynı-donanım tahmininden küçük olanı: +3,2 doğruluk / +3,7 macro-F1 puanı (RTX 3090, tek S0 koşusu). Diğer aynı-donanım tahminleri: GTX 1650'de +5,4 [3,5; 7,2] / +6,8 [5,0; 8,7]; Keras'ta +5,3 / +6,4 (tek koşu). Makalede üç tahmin "3,2–5,4" aralığı olarak veriliyor. **Keras yeniden eğitildi:** S1'de macro-F1 0,820, B2'nin 3,6 puan altında (GA [−5,5; −1,7]; 5 katın 0'ında B2'den iyi). **Keras kuralı tetiklenmedi.** |
| **M4** | Durum tahmincisi anomali oturumlarını görmüştü; görmeseydi ne olurdu? | Önceden yazılan okuma kuralı sağlandı. Yalnız s01+s02 ile eğitilen tahminci (ii): FAR 3/50 = %6,0 [2,1; 16,2], tespit %65,3 (Faz A %63,3), AUROC 0,938 (Faz A 0,924). Oturum dışarıda bırakmalı (i): FAR 6/51 = %11,8, tespit %60,1, AUROC 0,928; ama hızlı tempo %64,6'ya, yavaş tempo %55,2'ye düşüyor. Gereken etiketli kayıt: (ii) için 11,6 dk, havuzlanmış S1 için 21,7 dk (kare tabanlı; duvar saatiyle ≈ 13 ve ≈ 25,4 dk, makalede kullanılan bu — bkz. `faz_c/C_RESULTS.md` C.2, düzeltme 2026-11-05). |

## Dikkat edilmesi gerekenler
1. **Yasak bakış ihlali (düzeltildi ve kayda geçti).** A4.2 betiği ilk koşusunda, s02'deki doğal sapma döngüleri dışlanarak hesaplanan bir FAR sütunu üretti. Bu, o döngülerin alarm durumunu dolaylı olarak gösteriyordu.
   - Doğrulayıcı bunu yakaladı; ilgili kod ve sütun silindi, çıktılar yeniden üretildi.
   - `DESIGN_A4.md` Sapmalar bölümüne bir DÜZELTME notu eklendi.
   - Ancak doğrulayıcı ve düzeltme ajanı bu değerleri gördü; görülen bir şey geri alınamaz. Değerler hiçbir yerde raporlanmayacak ve kullanılmayacak.
   - Diğer sayıların hiçbiri bu değerlere bağlı değil.
2. **Keras karşılaştırmasının kapsamı:** Bu, eski defterdeki *mimarinin* Faz A protokolüyle yeniden eğitilmesi; yayımlanmış modelin kendisi değil. Doğrulama seti, kontrol noktası kuralı, etiketler ve hız kanalı farklı.
   - Keras'ın S0 (tohum 42) doğruluğu %90,34. Eski %93,66 ile arasındaki 3,3 puan açıklanamadı.
3. **(ii) tahmincisinde ToF sınırlarıyla tespit düşüyor** (%32,1'e karşı %59,0). Bunun nedeni D3 eşiğinin en büyük kalibrasyon skoruna takılması; tek kanallar sağlam (süre kanalı yalnız: hızlı %81,6, yavaş %93,1). Makalede açıklandı.
4. **Kayıt düzeltmeleri:**
   - `tools/` altındaki gece düzenlemeleri (`remote_run.sh`, `remote_start.sh`) Sapmalar'a sonradan eklendi.
   - `14_keras_runs.csv`'deki "cpu" etiketi yanlış, gerçek cihaz RTX 3090. Bu da kayda geçti.

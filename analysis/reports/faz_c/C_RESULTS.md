# Faz C sonuçları — gece ek denemeleri (2026-11-05)

**Plan:** `faz_c/DESIGN_C.md`. Plan çalıştırmadan önce yazıldı ve hiç sapma olmadı.

**Doğrulama:** İki bağımsız doğrulayıcı betikleri sıfırdan yeniden çalıştırdı ve çıktıları bayt düzeyinde aynı buldu. Anahtar sayıları kendi kodlarıyla yeniden hesapladılar. Engelleyici bulgu çıkmadı. Raporlar: `reports/faz_c/verify_C1.md`, `verify_C2.md`.

**Kural:** Bu sonuçlar sen görmeden `main.tex`'e girmeyecek.

## C.1 — Küçük ormanın derlenmiş C kodundaki gerçek boyutu
**Eşdeğerlik:** Ormanı C'ye aktaran üreticiyi kendimiz yazdık; gcc ile derlendi.
- C kodunun tahminleri, 16.248 test penceresinin hepsinde (5 kat) ve tam veri modelinde sklearn ile %100 aynı.
- Bağımsız bir Python yeniden uygulaması da birebir tuttu.
- Eşikler float32'ye güvenli yuvarlandı; hiçbir düğümde karşılaştırma değişmiyor.

**Ölçülen sabit veri boyutu** (`gcc -Os`, `.rdata`; 10 ağaç, tam veri modeli; 1 KiB = 1024 B):

| Yaprak kodlaması | Boyut | B.1'deki 8 B/düğüm varsayımına (159,8 KiB) oranı |
|---|---|---|
| (a) float32 olasılık | 419,8 KiB | 2,63× |
| (b) uint8 olasılık | 239,9 KiB | 1,50× |
| (c) sert oy (yaprakta yalnız sınıf) | **189,9 KiB** | 1,19× |

- **Kayıp yok:** Ağaçlar tam derinliğe büyütüldüğü için 10.235 yaprağın hepsi saf. Bu yüzden sert oy ve uint8 bu veride hiçbir şey kaybettirmiyor: F1 0,846, AUROC 0,937, FAR 0,039, tespit 0,535; float modelle aynı.
- **Çalışma belleği:** 248 B. Pencere tamponu 140 B, öznitelik vektörü 84 B, toplayıcı 24 B.
- **Ölçek için:** 25 ağaçlı model (c) ile 468,5 KiB. B2'nin 300 ağacı yalnız aritmetik hesapla (c) 5,7 MB.
- **Önceden yazılmış okuma:**
  - (a) varsayımdan 1,5–3 kat büyük çıkacaktı: **tuttu** (2,63×).
  - Sıralama c < b < a olacaktı: **tuttu**.
- **Karar kuralı:** (c) 189,9 KiB ≤ 192 KiB MSS veri RAM'i, ΔF1 −0,010 ≥ −0,02. Öngörülen cümle: "sığar".

**Sınırlar (cümleye mutlaka girmeli):**
- **Pay çok dar:** 2.112 B (%1,1). "192 KB" ondalık okunursa (192.000 B) sığmaz.
- Yerleşim optimize edilmedi. Örneğin LEFT dizisi atılabilir, ön-sıra düzeninde LEFT = i+1; bu payı büyütür ama ölçülmedi.
- Yalnız sınıflandırıcının sabitleri ölçüldü. Aygıt yazılımı, yığın, nokta bulutu tamponları ve anomali katmanı hesapta yok.
- "Veri RAM'ine sığar" demek, sabit verinin RAM'e yüklendiğini varsayıyor. Hiçbir şey çipte çalıştırılmadı.

**Önerdiğim makale cümlesi:** "Exported to C, the 10-tree forest needs 190 KiB of constant data with class-only leaves (lossless here because every leaf is pure) and 248 B of working memory, which is within the size of the IWR6843's on-chip memory banks (192 KiB data RAM, 512 KiB program RAM); we did not run it on the device, and that memory is shared with the radar firmware."

**Dosyalar:**
- Rapor: `reports/faz_c/01_export_c.md` ve `.csv`.
- Üretilen C kodu: `faz_c/c/`.

## C.2 — Kayıttaki zaman boşlukları nereden geliyor?
**Sayılar:**
- 696 boşluk var; 2.801 radar karesi kayıtta yok. Bu, radar periyotlarının %14,7'si.
- Boşlukların %99,3'ü 0,1 s'nin tam katı (±5 ms). Yani eksik olanlar bütün radar kareleri.

**İki hipotezin ayrımı:**
- Boşluğa komşu karelerde medyan **2** nokta var; tüm karelerde 7.
- Boşluk komşularının %97,5'i "kutu yok" anında; tüm karelerde bu oran %56,9.
- Boşluk süresince ToF'ta kutu aktifliği %0,9; aynı uzunluktaki rastgele aralıklarda %36,1.
- Kare kaybetme olasılığı, kutu yokken kutu aktifken olduğundan ~100 kat yüksek.
- ToF ve video boşluk sırasında kesintisiz kaydetmeye devam ediyor.

**Karar** (önceden yazılmış kurala göre): **"Boş kareler kaydedilmedi" hipotezi destekleniyor.**
- Kanıt dolaylı. Kayıt kodu bu diskte ve uzak makinede yok. `experiment_info.txt`, kaydedicinin bir Linux makinesindeki `<REMOTE_HOME>/Github/mmwave/mmwave_sensor` yolunda olduğunu söylüyor.
- Kesin kanıt ancak o kodla mümkün.

**Makale için sonucu — süre hesaplarında tutarsızlık:**
- Toplam kayıt süresi duvar saatine göre 31,8 dk; kare sayısı × 0,1 s ile hesaplanınca 27,1 dk (oran 1,17).
- Makaledeki **devreye alma dakikaları kare sayısından** hesaplanmış (pencere sayısı / 600). Kayıt süresi olarak okunursa düşük kalıyor:

| Makaledeki değer | Duvar saatiyle |
|---|---|
| 11,6 dk (s01 + s02, ii) | ~13,0 dk |
| 21,7 dk (havuzlanmış S1 katı) | ~25,4 dk |
| 22,6 dk (oturum dışarıda bırakmalı, ortalama) | ~26,5 dk |

- Konumları: `main.tex` (~510 ve ~560; satırlar değişmiş olabilir), `supplementary.tex` ~306 (orada tanım açıkça yazılı), `reports/faz_a4/04_unseen_minutes.csv`.
- Makalenin diğer süre cümleleri zaman damgasından hesaplanmış; onlar doğru.
- Ayrıca `reports/paper_prep/05_gaps.md`'de "yalnız en az bir noktası olan kareler kaydedilir" ifadesi bir kesinlik olarak yazılmış. "Verilerle uyumlu" diye yumuşatılmalı.

**Önerdiğim veri bölümü cümlesi:** "Frames in which the radar reported no points appear not to have been logged: 2,801 radar periods (14.7%) are missing, almost all while no box was touched. A 5-frame window therefore spans more than 0.5 s in 13.0% of windows."

**Dosyalar:** `reports/faz_c/02_time_gaps.md` ve `.csv`'ler. Disk araması: `faz_c/02_time_gaps_search.txt`.

## Senin kararın gereken noktalar
1. **C.1 cümlesi makaleye girsin mi?** B.1 ile birlikte düşünülmeli.
   - Önerim: evet, yukarıdaki temkinli cümleyle.
   - derginin okuyucusu için en somut gömülü sistem argümanı bu.
2. **Devreye alma dakikaları:** Önerim duvar saatine geçmek (13,0 / 25,4 dk). Alternatif, metinde "dakikalık kayıtlı kare" diye adlandırmak. Bu bir hata düzeltmesi; onayınla hemen yaparım.
3. **Veri bölümüne boşluk cümlesi girsin mi?** Önerim: evet, B_RESULTS (f)'deki doğruluk notuyla birlikte.
4. **Kayıt kodu (isteğe bağlı):** `mmwave_sensor` kaydedicisi elindeyse, boş kareyi atlayan satırı görmek nedeni kesinleştirir.

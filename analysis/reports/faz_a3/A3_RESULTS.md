# Faz A3 sonuçları — "Ne kadarı yeter, sınırlar nerede?" (2026-11-04)

Ön-kayıt: `faz_a3/DESIGN_A3.md`; sapmalar orada tarihli olarak kayıtlı.
- İkincil analizler. Birincil sonuçlar Faz A'dır.
- İki adversarial doğrulayıcı üç analizi de bayt bayt yeniden üretti; engelleyici bulgu yok (`verify_a.md`, `verify_b.md`).
- **Önceden yazılan tahminlerin bir kısmı tutmadı.** Bunlar aşağıda olduğu gibi raporlanıyor.

## A3.3 — Döngü içi alarm: garanti birebir korunuyor (en temiz sonuç)
- **Sert kontrol GEÇTİ.** Döngü içi (çevrimiçi) alarm kümesi, Faz A'nın çevrimdışı D3 alarm kümesiyle her katta ve her iki hücrede birebir aynı. Yani yanlış alarm garantisi (%7,8) hiç değişmeden döngü içine taşınıyor.
- B2 × pred'de alarmın verildiği an, döngünün geçmiş oranı olarak (medyan [IQR]):

| tür | alarm oranı | geçmiş döngü oranı | not |
|---|---|---|---|
| seq135 | 0,25 | **0,29** [0,26; 0,35] | gerçek sapmadan medyan 0,1 s sonra |
| seq145 | 0,48 | **0,29** [0,26; 0,33] | medyan 0,1 s; bu, ziyaret onay gecikmesinin kendisi |
| yavaş (slow) | 0,93 | **0,32** [0,28; 0,36] | 237 alarmın 190'ı Box1→Box2 boşluğundan |
| hızlı (fast) | 0,92 | **1,03** [1,02; 1,03] | %80'i son olayda, döngü kapanınca |

- **Sıra sapmaları, yanlış geçiş olur olmaz yakalanıyor** (onay gecikmesi kadar sonra, ~0,1 s).
- **Tahminlerin durumu:**
  - seq135 ve seq145 < 0,5 → tuttu.
  - fast > 0,8 → tuttu.
  - **slow > 0,8 → TUTMADI.** Yavaş tempo erken yakalanıyor (~%32), çünkü ilk taşıma boşluğu zaten uzun. Doğru anlatım "tempo geç yakalanır" değil, "**hızlı geç, yavaş erken**".
- **Makale cümlesi:** "Aynı yanlış alarm garantisiyle sıra sapmaları sapmadan ~0,1 s sonra (döngünün ~%29'unda), yavaş tempo döngünün ~%32'sinde, hızlı tempo ise döngü kapanırken yakalanıyor."

## A3.1 — Hata yayılımı: hataların SAYISINDAN çok YERİ önemli
**Yapay gürültü (kol b):** ToF etiketleri bozulup sistemin tamamı yeniden koşturuldu: döngü kesimi, uydurma ve kalibrasyon.

| bozulma oranı r | iid AUROC | patlamalı AUROC |
|---|---|---|
| 0,02 | 0,962 | 0,895 |
| 0,05 | 0,925 | 0,847 |
| 0,10 | 0,855 | 0,765 |
| 0,20 | 0,725 | 0,691 |

- Yanlış alarm kalibrasyon sayesinde her koşulda α'nın altında kaldı (≤ 0,07).
- "Yeterli doğruluk" bandı (AUROC ≥ 0,90): iid'de r ≈ %5 [%2; %10]; patlamalıda %2'nin altında.
- **Tahmin 1 (patlamalı gürültü daha erken bozar): TUTTU.**
- **Tahmin 2 (gerçek CNN-GRU ve CNN-only, B2'den uydurulan patlamalı eğrinin 0,03 yakınında olur): TUTMADI.** Gerçek modeller yapay gürültüden çok daha iyi:
  - B2 %9,8 kare hatasında AUROC 0,924 alıyor; aynı orandaki yapay patlamalı gürültüde değer 0,77.
  - CNN-GRU: gerçek 0,879, yapay 0,75–0,77.
  - CNN-only: gerçek 0,835, yapay 0,69.
- **Neden (betimleyici, sonradan; nedensellik test edilmedi):**
  - Gerçek hatalar etiket geçişlerinde toplanıyor: B2'nin hatalı karelerinin %76'sı bir geçişin yakınında; tüm karelerde bu oran %49.
  - Geçiş yakınındaki hatalar çoğunlukla sınırın birkaç kare kaymasıdır ve debounce bunları emer. Yapay gürültü ise ziyaretlerin ortasına düşüp sahte ziyaret üretiyor. B2'de döngü başına sahte ziyaret: gerçek 0,21, yapay patlamalı 0,47.
- **Kol (a) (analitik model): bütün kaynaklarda KALDI.** Kısmen tanım uyuşmazlığından: analitik sayım sınır kaymalarını da sahte ziyaret sayıyor. Sonradan denenen bir varyant da kaldı. Sonuç değişmiyor: hata *oranından* sahte ziyaret oranını tahmin eden basit model işe yaramıyor.
- **Makale cümlesi (dikkatli):**
  - "Döngü düzeyi tespit, kare düzeyi hata oranının kötü bir göstergesi: rastgele yerleşen %10 hata AUROC'u ~0,77'ye düşürürken, aynı oranda ama geçişlerde toplanan gerçek hatalarla B2 0,92'de kalıyor. Durum tahmincilerini kare doğruluğuyla değil, döngü düzeyinde kıyaslamak gerekiyor."
  - "~%X'e kadar dayanır" diye tek bir sayı **verilemez**.
  - Uyarı: B1 bu örüntüye uymuyor (gerçek 0,54 < yapay 0,60).

## A3.2 — Çalışma zarfı: neyi görüyor, neyi göremiyor?
**Süre ekseni:** 1.111 ToF dokunuşu (30 Hz), B2.
- **> 0,5 s dokunuşlar her oturumda güvenilir yakalanıyor:**
  - 0,5–1,0 s: en düşük %96 (ham) / %95 (debounce sonrası).
  - > 1 s: %100 / en düşük %92.
  - Tahmin (b) tuttu.
- **Kısa dokunuşlar zayıf:**
  - < 0,2 s: s01'de %63, s05'te %44 (ham).
  - 0,2–0,3 s: s01'de %64, s05'te %86.
  - 0,3–0,5 s: s01'de %83, diğer oturumlarda ~%100.
- **Tahmin (a) desteklenmedi.** d50'nin [0,2; 0,4] s içinde olması bekleniyordu; tahmin edilebilen iki oturumda daha düşük çıktı (s01 0,05 s, s05 0,12 s). Diğer dört oturumda d50 hesaplanamadı: s03, s04 ve s06'da hiç kaçırma yok; s02'de tek kaçırma var (0,032 s), bu yüzden tam ayrışma oluyor. *(Düzeltme 2026-11-04: önceki sürüm "dört oturumda hiç kaçırma yok" diyordu.)*
  - Bu yüzden 0,4 s'lik "sınır" s01'de desteklenmiyor: 0,3–0,5 s aralığında tespit %83.
  - Dürüst ifade: "≥ 0,5 s güvenilir; < 0,3 s'de tespit %45–65'e düşüyor."
- **Betimleyici bulgu:**
  - Eğitim etiketi (L0) kısa dokunuşların çoğunu silmişti; görünürlük < 0,2 s'de %17, 0,2–0,3 s'de %2.
  - Buna rağmen B2 bu dokunuşların %50'sini ve %77'sini yakalıyor. Yani model, etiketin göremediğini kısmen görüyor.

**Konum ekseni (betimleyici):**
- **Karışıklık kutu aralığıyla monoton değil; 11,8 cm çizgisi B2'nin hatalarını ayırmıyor.**
  - 11,8 cm altındaki 4 çiftte simetrik hata medyanı 0,005 (en çok 0,013).
  - 11,8 cm ve üstündeki 10 çiftte medyan 0,000. Ama en büyük hata da bu grupta: s01'de Box2–Box3, 12,8 cm, 0,147.
  - s04 Box4–Box5 (8,2 cm): 0,004.
- **Sonuç:** "≥ ~12 cm aralık gerekir" cümlesi **desteklenmiyor.**
- Tek hedefte açı *hassasiyeti*, Rayleigh *çözünürlüğünden* iyi. Etiketli bir sınıflandırıcı, çözünürlük altındaki komşu kutuları da büyük ölçüde ayırıyor.
- **Bu, sondajdaki H1 yorumunu daraltır.** H1'deki başarısızlık, *etiketsiz* mod keşfinin çözünürlük sınırına takılmasıydı; *etiketli* ayrımda böyle bir sınır görülmüyor.
- s01'deki zorluk çifte özgü: Box2 ~6 cm kayıyor ve s01'de 5 sınıf var.

## Makale için özet
1. **En temiz katkı (A3.3):** Çevrimdışı conformal garanti, birebir aynı alarm kümesiyle döngü içine taşınıyor. Sıra sapmaları ~0,1 s'de, yavaş tempo döngünün ~%30'unda, hızlı tempo döngü sonunda yakalanıyor. Bu, eski retlerdeki "yalnız çevrimdışı" eleştirisine doğrudan cevap.
2. **Yeni ve önemli bir yöntemsel ders (A3.1, betimleyici kısmı sonradan):** Kare doğruluğu, döngü düzeyi tespitin iyi bir göstergesi değil; hataların *nerede* olduğu önemli. Bu, A2.2'deki sıralamanın ters dönmesi bulgusuyla birlikte A çerçevesinin "ne kadar model gerekiyor?" sorusunu güçlendiriyor: doğru metrikle bakılmalı.
3. **Sınırlar (A3.2):**
   - Süre: ≥ 0,5 s'lik dokunuşlar güvenilir; < 0,3 s zayıf.
   - Konum: etiketli ayrımda ~8 cm'lik aralık bile sorun değil.
   - Sondajdaki "çözünürlük sınırı" yorumu yalnız etiketsiz keşif için geçerli.
4. **Tutmayan tahminler:** slow > 0,8; Tahmin 2; d50 aralığı; kol (a). Makalede dürüstçe "beklenenden farklı çıktı" olarak verilecek.

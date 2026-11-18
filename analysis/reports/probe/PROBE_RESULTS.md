# Sondaj sonuçları — 3 soru (2026-11-04)

Kod: `probe/00_load.py`, `01_dwell_modes.py`, `02_tempo_scaling.py`, `03_few_shot.py` (uzak venv, `tools/remote_run.sh`).
Ayrıntı: `00_schema.md`, `q1.md`, `q2.md`, `q3.md`. Doğrulama: `q1_verify.md`, `q2_verify.md`, `q3_verify.md`.
Her script sıfırdan yeniden çalıştırıldı ve md'ler stdout ile birebir aynı çıktı. Engelleyici hata bulunmadı; her soruda bir parametre değiştirilerek yapılan testler hükmü değiştirmedi.
Tüm bölmeler `(session, loop_id)` bazlıdır; eşikler ve geçme ölçütleri script başında, sonuçlara bakılmadan sabitlendi.

## Tek bakışta
| Soru | Önceden sabitlenen ölçüt | Sonuç | Hüküm |
|---|---|---|---|
| **H1** İstasyonlar etiketsiz bulunuyor mu? | NMI ≥ 0,7 ve K beklenen kümede; s01'de **ve** senaryo oturumlarının ≥ 4/5'inde | Hiçbir vekil s01'i geçemedi (en iyi NMI 0,691). Senaryo oturumlarında tüm noktalar (2B) ile NMI 0,86–1,00, K = 3–4, 5/5 | **✘** (s01'de çözünürlük sınırı) |
| **H2** Süreler tek bir tempo çarpanıyla mı ölçekleniyor? | Etkileşim kısmi η² < 0,06 ve her segmentte ε²(d/T) < 0,06 | Etkileşim η² = **0,286**; 6 segmentin hepsi 0,09–0,62 | **✘** (net) |
| **Q3** Kaç döngü yetiyor? | Doğruluk n=20'ye 2 puan yakın **ve** her segmentte d/T GA yarı-genişliği < 0,02 | Yapı **1 döngüde** doyuyor (doğruluk 0,989). Zamanlama n=20'de 0,0224 ile sınırın hemen üstünde kalıyor (dışdeğerlemeyle ~21 döngü ≈ 1,8 dk) | N yok (ızgarada); yapı 1, zamanlama ~20 döngü |

**Karar kuralları (ikisi farklı yöne işaret ediyor):**
- **Kullanıcının kuralı** (plan aşamasında): sondaj güçlüyse yeni çerçeve, değilse hızlı yol (mevcut boru hattı + yöntemden bağımsız düzeltmeler). H1 ✘ ve H2 ✘ "güçlü" sayılmaz → **hızlı yol**.
- **Brifing §5:** H1 ✘ → ToF kalibrasyonlu ZDA (H1 "future work"). H2 ✘ olduğu için zamanlama skoru **adım bazlı normalize** edilir; tek bir τ ile değil.
- **Ortak zemin:** §3 düzeltmeleri (döngü bazlı bölme, train/cal/test, −log p, conformal eşik, döngü düzeyi metrik) = Faz A protokolü; iki yolda da zorunlu.

## Q1 — İstasyon keşfi (H1 ✘, ama bilgilendirici bir ✘)
- **Çözünürlük sınırı:** Yanal çözünürlük 0,45 m × 15° ≈ **11,8 cm**. s01'de komşu kutu aralıkları 8,0 / 12,8 / 9,8 / 8,5 cm; 4 çiftten 3'ü çözünürlüğün altında. Senaryo oturumlarında 12 çiftten yalnız 1'i altında (s04 Box4–Box5, 8,2 cm).
- **s01 eşiği etiketle bile aşılamıyor:** ToF etiketli en-yakın-merkez sınıflandırıcı bile s01'de NMI 0,693 (doğruluk 0,860) veriyor; 0,7 eşiği bu tavanın üstünde. Etiketsiz en iyi sonuç (tüm noktalar 2B, NMI 0,691) bu tavanın yaklaşık %100'ü. Yani s01'deki başarısızlık mod bulma hatası değil, **sensörün açısal çözünürlük sınırı**.
- **İstasyonlar çözünürlükten uzakken etiketsiz keşif çalışıyor:** Senaryo oturumlarında vekil (i) 2B için NMI 0,95 / 0,96 / 0,86 / 0,92 / 1,00, K = 3/3/4/3/3. *Uyarı:* bu sonuç post-hoc ve kırılgan; SNR ağırlığı kaldırılınca 4/5'e düşüyor.
- **v = 0 statik noktalar:** Kutu bilgisini neredeyse tüm noktalar kadar taşıyor (s01 NMI 0,655; senaryolarda 4/5). Ama "bekleme" sinyali vermiyor: statik nokta payı etiketten bağımsız. Brifingin "yavaş nokta = bekleme" fikri **desteklenmedi**.
- **Kare merkezi ortalaması** tekil noktalardan daha iyi değil; çözünürlük altındaki çiftleri kurtarmadı (beklentinin aksine).
- **Havada birleştirme bölgesi** nokta vekillerinde ayrı bir mod oluşturmuyor; "none" kareleri kutu modlarına emiliyor.
- **Box2 kayması:** s01 ile s02/s05/s06 arasında Box2 yaklaşık 6 cm kayıyor; diğer kutular 1 cm'den az. Oturumlar arası sınıflandırmada Box2↔Box3 karışıklığının kaynağı bu. s01 merkezleriyle doğruluk tüm sınıflarda 0,867; oturumun kendi kutularına sınırlandığında 0,969.

## Q2 — Tempo ölçeklenmesi (H2 ✘)
Normal / Fast / Slow, oturum başına 50 / 47 / 51 döngü (yalnız tam `1-2-5` döngüleri, son döngü hariç). Medyan T: 4,99 / 3,75 / 8,33 s (Fast/Normal 0,75, Slow/Normal 1,67).
İki yönlü ANOVA (log d), kısmi η²: oturum 0,699, segment 0,815, **etkileşim 0,286**. 10 Hz etiketlerle 0,262 çıkıyor; `min_box_s` 0,2'ye çıkarılınca 0,295.

| Segment | Fast endeksi | Slow endeksi | Esneklik | Yorum |
|---|---|---|---|---|
| dwell_1 (Box1 kalış) | **1,20** | 0,96 | 0,74 | Hızlanınca Box1'de görece uzun kalıyor |
| gap_1b (Box1→Box2 geçiş) | **0,89** | **1,47** | 1,64 | **En büyük orantısızlık**; yavaşlamayı en çok bu geçiş taşıyor |
| dwell_b (Box2 kalış) | 0,97 | **0,79** | 0,71 | |
| gap_b5 (havada birleştirme dahil) | 1,08 | **0,89** | 0,76 | |
| dwell_5 (Box5 kalış) | **0,80** | **0,84** | 1,01 | İki yönde de payı küçülüyor |
| gap_51 (dönüş) | **1,16** | **1,35** | 1,23 | İki yönde de payı büyüyor |

Endeks = d oranı / T oranı (1 = orantılı); kalın değerlerin %95 bootstrap GA'sı 1'i dışlıyor. **Karar sayısı log d üzerindeki etkileşim η² = 0,286'dır.** d/T bileşimsel olduğundan (paylar toplamı 1) segment başına ε² değerleri birbirine bağlı; tablo betimleyicidir. Seq135/Seq145 oturumlarında dwell_b, gap_b5 ve dwell_5 için d/T Normal'le aynı; yani orantısızlığı orta istasyonun hangisi olduğu belirlemiyor.
**Bulgu:** Tempo değişimi adımları eşit etkilemiyor. Tek bir "tempo" ile normalize edilen süre skoru, normal bir hızlanmayı anomali sanar ya da gerçek bir anomaliyi kaçırır. Zamanlama modeli adım bazlı olmalı.

## Q3 — Kalibrasyon maliyeti
- **Yapı (ToF kalibrasyonlu):** s02'de kalibrasyon döngüleri dışındaki döngülerde, en-yakın-merkez kare doğruluğu n = 1/3/5/10/20 için 0,989 / 0,989 / 0,989 / 0,990 / 0,991. **Tek döngü (~5 s) yetiyor.** s02 merkezleri Fast'e 0,956, Slow'a 0,997 doğrulukla aktarılıyor ve n'den bağımsız.
- **Zamanlama (d/T):** Segmentler içinde en büyük %95 GA yarı-genişliği n = 3/5/10/20 için 0,039 / 0,035 / 0,029 / **0,022**; bağlayıcı segment gap_b5. Dışdeğerleme ~21 döngü (≈ 1,8 dk) veriyor. *Uyarı:* küçük n'de bootstrap GA aşırı iyimser (n = 3'te kapsama 0,75).
- **Tempo (T):** n ≥ 10 döngüde (~50 s) tüm pencerelerde %10 içinde.
- **Etiketsiz yol:** 2B MeanShift (s01'de seçilen q = 0,1) her n'de fazla bölüyor (n = 20'de bile K ≈ 6,4). 1B x-yalnız varyant n = 10'da pencerelerin %90'ında K ∈ {3, 4} buluyor.

## Ne anlama geliyor (yorum)
1. **"Yapı–Üslup" makalesi (H1 ✔ + H2 ✔) bu veriyle kurulamaz.** Brifingin kuralı ToF kalibrasyonlu ZDA'yı gösteriyor.
2. Sondaj yine de **üç savunulabilir, sayısal bulgu** verdi. Bunlar "mevcut tekniklerin birleşimi" eleştirisine doğrudan cevap olur:
   - (a) Nokta bulutu radarında istasyon ayrımının fiziksel sınırı (~12 cm). Altında etiketle bile ayırt edilemiyor, üstünde etiketsiz bile bulunuyor.
   - (b) Yapı 1 döngüde kalibre oluyor; zamanlama ~20 döngü (~2 dk) istiyor.
   - (c) Tempo adımları orantısız ölçekliyor. Bu da adım bazlı zamanlama skorunu gerekçelendiriyor ve %29,3 yanlış alarmın olası bir kaynağına işaret ediyor (tek tip süre Z-skoru).
3. **Hızlı yol için uyarı:** Kare merkezinde en-yakın-merkez gibi çok basit bir taban çizgisi, kutu karelerinde %98,9 doğruluk veriyor. CNN-GRU'nun %93,66'sı 6 sınıf üzerinden, "none" dahil ve sızıntılı bölmeyle ölçülmüş; doğrudan kıyaslanamaz. Yine de hakemlerin istediği taban çizgisi kıyası eklendiğinde, derin modelin katkısı yalnız "none"/geçiş karelerinde gösterilebilir. Mevcut boru hattını kıyassız göndermek riskli. Ayrıca %98,9, kutuların ≥14 cm aralıklı olduğu s02'de ölçüldü. Derin modelin hâlâ fark yaratabileceği yerler, etiketli merkezlerin bile 0,860'ta kaldığı **s01** ile **none/geçiş kareleri**. Bu yüzden Faz A kıyası bu iki durumu **ayrı ayrı** raporlamalı, yalnız kutu karesi doğruluğunu değil.
4. **Pişmanlıksız sonraki adım: Faz A** (döngü bazlı GroupKFold, train/calibration/test, −log p skoru, conformal eşik, döngü düzeyi AUROC/FAR). Hem hızlı yol hem ZDA yolu için zorunlu (§3 düzeltmeleri). Faz A içinde CNN-GRU sızıntısız yeniden ölçülür ve basit/ZDA taban çizgileriyle aynı protokolde kıyaslanır. Hangi modelin makalenin ana modeli olacağına bu kıyas karar verir.

## Açık uçlar
- Normal oturumlardaki doğal hatalar (`1-2-5-5`, `1-4-4-5`, `1-2-2-2-5`) etiketsiz. Faz A'da gerçek anomali olarak işaretlenebilir.
- s01'in 0,7 NMI eşiği, etiketli tavan 0,693 olduğu için tasarım gereği aşılamazdı. Bu, sonradan yapılan bir ölçüt değişikliği değil, raporlanacak bir bulgu.
- **Operatör sürüklenmesi:** s02'de gap_1b'nin bootstrap kapsaması n=20'de bile 0,548 (`q3.md` tablo 3e). Döngüler iid değil; oturum içinde zamanlama kayıyor. Faz A'da kalibrasyon setini tasarlarken (ardışık mı, aralıklı mı) bu dikkate alınmalı. Bu kendi başına bir bulgu da olabilir.
- Q1'in senaryo oturumlarındaki 5/5 sonucu post-hoc; makalede ana iddia olarak değil, çözünürlük bulgusunun ikinci yarısı olarak kullanılmalı.

---

## Uzlaştırma notu (2026-11-04, Faz A3.2'den sonra eklendi; yukarıdaki metin değiştirilmedi)
Yukarıdaki H1 yorumu "çözünürlük sınırı" diyordu. Bu yorum yalnız **etiketsiz** istasyon keşfi için geçerli.
- Faz A3.2'de etiketli B2 sınıflandırıcısının ikili karışıklığı ölçüldü (`reports/faz_a3/02_envelope.md`):
  - 11,8 cm'nin altındaki 4 komşu kutu çiftinde simetrik hata medyanı 0,005 (en çok 0,013).
  - En kötü çift s01 Box2–Box3; aralığı 12,8 cm, yani çizginin üstünde (0,147).
- Tek hedefte açı *hassasiyeti* Rayleigh *çözünürlüğünden* iyi. Bu yüzden etiketli ayrım bu aralıklarda çözünürlükle sınırlı değil.
- Makalede doğru ifade: "etiketsiz istasyon keşfi başarısız oldu; etiketli ayrım bu aralıklarda çözünürlükle sınırlı değil." **"≥ 12 cm aralık gerekir" gibi bir kural yazılmayacak.**

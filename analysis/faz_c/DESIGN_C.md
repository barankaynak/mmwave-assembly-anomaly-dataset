# Faz C — gece ek denemeleri (keşif; analiz planı çalıştırmadan önce yazıldı, 2026-11-05 ~02:25)

**Amaç:** Faz B'nin makaleye en çok değer katabilecek iki bulgusunu sağlamlaştırmak:
- B.1'deki "çip belleğine sığar" iddiası şu an düğüm başına 8/12 bayt **varsayımına** dayanıyor; C.1 bunu derlenmiş gerçek koda dayandıracak.
- B.0(f)'deki kayıt zaman boşluklarının nedeni bilinmiyor; C.2 bunu açıklayacak.

**Genel kurallar:**
- Sonuçlar `reports/faz_c/` altına yazılır. Kullanıcı görmeden `main.tex`'e girmez.
- Yalnız S1 bölmesi kullanılır. Rastgele bölme yapılmaz.
- Faz A/A2/A3/A4/B dosyalarına yazılmaz.
- Uzak makine (<REMOTE_HOST>) KULLANILMAZ; GPU'su başka oturuma devredildi.
- Hiçbir şey dışarı gönderilmez. Yeni paket kurulmaz; yalnız mevcut venv ve mingw64 gcc kullanılır.

## C.1 — Küçük ormanın C'ye aktarılması ve gerçek boyutu
**Modeller:** B.1'de seçilen model (10 ağaç, derinlik sınırsız). Ölçek karşılaştırması için 25 ağaç × sınırsız ve B2 (300 ağaç × sınırsız).
- Hiperparametreler, tohum (20261104) ve öznitelikler `faz_b/01_tiny_forests.py` ile aynıdır; onun fonksiyonları import edilir.
- 5 kat modeli ve tam veri modeli kullanılır.

**C dışa aktarıcı:** Kendi yazdığımız, bağımlılıksız bir Python → C üreticisi; düğüm dizileri `static const` olarak tutulur.

| Alan | Tip |
|---|---|
| feature | uint8 |
| threshold | float32 |
| left, right | int16; ağaç başına düğüm sayısı > 32767 ise int32 |
| yaprak | aşağıdaki üç kodlamadan biri |

**Yaprak kodlamaları:**
- (a) 6 sınıf olasılığı float32, yumuşak oylama (sklearn ile aynı).
- (b) 6 sınıf olasılığı uint8'e nicemlenmiş (×255, yuvarlama), yumuşak oylama.
- (c) Yaprak başına argmax sınıfı uint8, sert oylama; eşitlikte küçük sınıf numarası kazanır.

**Uygulama ayrıntıları** (kapının yanlışlıkla düşmemesi için, çalıştırmadan önce sabitlendi):
- **Eşik:** sklearn eşiği float64 saklar ve float32 girdiyle `x <= t` karşılaştırır. C'de `t32` = t'yi aşmayan en büyük float32 olarak saklanır (`np.float32(t)` yukarı yuvarlarsa `np.nextafter(·, -inf)`). Böylece `x32 <= t` ile `x32 <= t32` tam olarak eşdeğer olur.
- **Girdi:** Kapı yalnız ağaç çıkarımını test eder. sklearn'in gördüğü float32 öznitelik matrisi Python'dan ham ikili dosya olarak yazılır ve C sürücüsüne verilir. Öznitelikler C'de yeniden hesaplanmaz.
- **Yaprak değerleri:** sklearn 1.9.1'de `tree_.value`'nun düğüm başına normalize olup olmadığı kodda kontrol edilir; varsayılmaz. `predict_proba`, ağaç başına normalize edip ortalamaya eşdeğer olmalı. `predict` = argmax; eşitlikte küçük sınıf.
  - Kapı sürücüsü toplamayı double ile yapar.
  - Gömülü sisteme gerçekçi varyant olan float32 toplama ayrıca raporlanır.
- **Yerleşim:** Diziler alan başına ayrı `static const` olarak tutulur (struct-of-arrays). Struct dolgusu olmadığı için sabit veri boyutu derleyiciden ve ABI'dan bağımsızdır.
- **B2 (612 bin düğüm):** C kaynağı onlarca MB tutacağı için derlenmez; boyutu yalnız aritmetik olarak verilir (düğüm sayısı × tip genişlikleri). 10 ve 25 ağaçlı modeller derlenir.

**Ölçüm:**
- `gcc -Os -c` (mingw64) ile derlenir. `size` / `objdump -h` ile `.rdata`/`.rodata` (model sabitleri) ve `.text` (çıkarım kodu) bölümleri ayrı ayrı raporlanır.
- Ölçülen sabit veri, aritmetik hesapla (düğüm sayısı × tip genişlikleri) karşılaştırılır.
- x86 kod boyutu ARM Cortex-R4F'yi temsil etmez; yalnız sabit veri boyutu mimariden bağımsız sayılır.

**Eşdeğerlik kapısı:** Kodlama (a) ve double toplama ile C çıkarımı, tüm S1 test pencerelerinde ve tam veri modelinde sklearn `predict` ile **%100** aynı tahmini vermeli.
- Uyuşmayan her pencere için olasılık farkı ve eşitlik durumu raporlanır.
- Kapı geçmezse rapor yazılır ama boyut iddiası kurulmaz.

**Kodlama sonuçları:** (b) ve (c) için şunlar verilir:
- sklearn'e göre tahmin uyuşma oranı;
- S1 makro-F1 (kat ortalaması ± sd) ve eşli ΔF1 [%95 t-GA];
- anomali hattında D3 AUROC, FAR@0,10 ve tespit@0,10. Bunlar `a2_cycles` / Faz B'deki pred segmentasyonu ve standart conformal ile, değiştirilmeden hesaplanır.

**Ek ölçüm (ikincil):** Çalışma belleği.
- Pencere tamponu: 5 kare × 7 istatistik × 4 B.
- Öznitelik vektörü: 21 × 4 B.
- Olasılık toplayıcı: 6 × 4 B.
- Sonuç tek satır.

**Önceden yazılmış okuma:**
- (a) için ölçülen sabit veri, B.1'deki 8 B/düğüm hesabından (fold ortalaması 130,9 KB, tam veri 159,8 KB) **büyük** çıkacak; yaprak olasılıkları dahil olduğu için tahminim ~1,5–3×.
- (c) en küçük olacak.
- **Karar kuralı:** Seçilen modelin herhangi bir kodlaması tam veri modelinde ≤ 192 KB sabit veriye sığıyor ve ΔF1 ≥ −0,02 ise makaleye önerilecek cümle: "X KB'lik derlenmiş model, MSS veri RAM'inin (192 KB) içine sığar".
  - Hiçbiri sığmıyorsa: "DSP L2 (256 KB) ya da MSS program RAM'i (512 KB)" ile tekrar denenir. Bunlar da tutmazsa "sığmaz" yazılır.
  - Her durumda "çipte çalıştırılmadı; bellek aygıt yazılımıyla paylaşılır" uyarısı eklenir.

**Dosyalar:**
- `faz_c/01_export_c.py`
- `faz_c/c/*.c|h` (üretilmiş kod ve test sürücüsü)
- `reports/faz_c/01_export_c.md` ve `.csv`

## C.2 — Kayıttaki zaman boşluklarının nedeni (betimsel)
**Soru:** 696 kare aralığı 0,15 s'den uzun (en uzunu 8,4 s); `frame_id` ise ardışık. Boşluklar, radarın nokta bulamadığı karelerin kaydedilmemesinden mi kaynaklanıyor?

**Testler:**
1. Boşluk süresi dt'nin 0,1 s'nin tam katlarına ne kadar yakın olduğu: dt/0,1'in en yakın tam sayıya uzaklığının dağılımı. Rastgele zamanlamaya karşı karşılaştırılır. Kayıp kare tahmini: round(dt/0,1) − 1.
2. Boşluk öncesi ve sonrası karelerin özellikleri:
   - nokta sayısı ve ToF durumu;
   - "kutu yok" oranı;
   - elin radardan uzaklığı (menzil);
   - boşluk öncesinde nokta sayısında düşüş eğilimi.
3. Oturum bazında boşluk oranı, ve ToF zaman ekseninde boşluk anlarında elin nerede olduğu. Ham ToF ~30 Hz'dir ve kayıp radar karelerinden bağımsızdır.
4. Kayıt kodu: kullanıcının diskinde TI UART ayrıştırıcısı / kayıt betiği aranır.
   - Aranacak anahtar kelimeler: `numDetectedObj`, `magic`, `0x0102`, `mmwave_data.csv` yazan kod.
   - Yalnız okunur, çalıştırılmaz.
   - Bulunursa boş karelerin atlanıp atlanmadığı satır numarasıyla raporlanır.

5. **Süre hesabı denetimi:** Oturum başına duvar saati süresi (ilk → son zaman damgası), `kare sayısı × 0,1 s` ile karşılaştırılır.
   - Makaledeki ve raporlardaki kayıt süresi / dakika sayılarının nasıl hesaplandığı kontrol edilir: `reports/faz_a4/04_unseen_minutes.csv`, `reports/paper_prep/05_gaps*`, `paper/main.tex` içindeki dakika ve süre cümleleri.
   - Bu sayılardan biri zaman damgası yerine kare sayısından geliyorsa, **düzeltme önerisi** olarak raporlanır. main.tex'e dokunulmaz.

**İki rakip hipotez, önceden yazılmış tahminleriyle:**
- **H-boş — "boş kareler kaydedilmedi":** Boşluk öncesi ve sonrasında nokta sayısı **az**; boşluklar "kutu yok" anlarında ve el menzil dışındayken.
- **H-yük — "kaydedici yük altında kare düşürdü"** (UART taşması, ayrıştırma hatası): Boşluk öncesi ve sonrasında nokta sayısı **çok**; boşluklar el aktifken ve hareketliyken.

**Testlerin okunuşu:**
- **(i) Tam kat testi:** Tolerans veriye göre belirlenir: düzenli aralıklarda (dt ≤ 0,15 s) |dt − 0,1|'in 95. yüzdeliğinin 2 katı. dt'lerin ≥ %90'ı 0,1 s'nin bir tam katına bu tolerans içinde yakınsa, kayıp kareler bütün radar kareleridir. Bu iki hipotezle de uyumludur; ayrım testi (ii)'dir. dt/0,1 − round(dt/0,1) dağılımının tamamı raporlanır.
- **(ii) Ayrım testi:** Boşluk komşusu karelerdeki nokta sayısı, aynı oturumun tüm karelerine göre medyan ve Mann–Whitney ile karşılaştırılır. Ayrıca boşluk anlarındaki ToF durumu ("kutu yok" oranı) bakılır.
  - Komşu karelerde nokta sayısı **düşük** ve "kutu yok" oranı yüksekse → H-boş.
  - Nokta sayısı **yüksek** ve kutu aktifse → H-yük.
  - Karışık sonuçta iki hipotez de reddedilmez; "belirsiz" yazılır.
- **(iii) Kayıt kodu:** Kayıt kodunda boş kareyi atlayan bir koşul bulunursa bu kesin kanıttır.

**Makale cümlesi:**
- H-boş desteklenirse: "frames in which the radar reported no points were not logged (N frames, X%); a 5-frame window therefore spans more than 0.5 s in Y% of windows".
- H-yük desteklenirse: kayıp karelerin bir sınırlama olarak yazılması önerilir.

**Dosyalar:** `faz_c/02_time_gaps.py`, `reports/faz_c/02_time_gaps.md` ve `.csv`.

## Doğrulama
Her alt aşama için ayrı bir doğrulayıcı:
- betiği sıfırdan yeniden çalıştırır;
- C.1'de C tahminlerini bağımsız bir Python yeniden uygulamasıyla karşılaştırır;
- md'deki sayıları stdout ve CSV ile eşler;
- önceden yazılmış okumanın harfiyen uygulandığını kontrol eder.

## Sapmalar (tarihli)

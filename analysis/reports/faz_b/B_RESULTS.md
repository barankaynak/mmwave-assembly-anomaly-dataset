# Faz B sonuçları — gece keşif denemeleri (2026-11-05)

**Plan:** `faz_b/DESIGN_B.md`. Plan koşudan önce yazıldı, ancak Faz A, A2 ve A3 sonuçları görüldükten sonra. Plan metninin sonradan değişmediği git geçmişi olmadığı için kanıtlanamıyor.

**Sapmalar:** Aynı dosyanın "Sapmalar (tarihli)" bölümünde. Ayrıca B.2 ve B.3'teki dört koşullu diz kuralı uygulamada ayrıntılandırıldı ve bu da sapma olarak kayıtlı. Ayrıntılar: 5 katlı eşli t-GA (df=4); her metrik için nokta ve GA testi; "en yalın" = en az V2 baytı.

**Durum:** B.0, B.1, B.2 ve B.3 tamamlandı.

**Doğrulama:** İki bağımsız doğrulayıcı sayı üreten bütün betikleri sıfırdan yeniden çalıştırdı; rapor üretici betikler (`02b_`, `03b_`) yeniden çalıştırılmadı. Doğrulayıcılar anahtar sayıları kendi kodlarıyla yeniden hesapladı.
- Bulunan tek engelleyici hata B.3'teydi (oturum adı) ve düzeltildi.
- Her alt aşamanın kapısı geçti: Faz A B2 birebir yeniden üretildi.
- Faz A, A2, A3 ve probe dosyalarına dokunulmadı (41 dosyanın md5'i aynı). `paper/` klasörüne yazılmadı.

Bu özetin kendisi de iki ajanla ayrıca denetlendi: ~175 sayı ve ~80 iddia kontrol edildi, bulunan düzeltmeler işlendi.

**Kural:** Bu sonuçlar sen görmeden `main.tex`'e girmeyecek. Aşağıdaki sıralama makale için değere göredir.

**Kısaltmalar:**
- **F1:** S1 bölmesinde (oturum, döngü) gruplu 5 katlı makro-F1.
- **AUROC:** D3 skoruyla döngü düzeyinde, tüm anomali tipleri birlikte.
- **FAR:** α=0,10'da normal döngülerde yanlış alarm oranı.
- **Referans B2 (Faz A):** F1 0,856, AUROC 0,924, FAR 0,078, tespit 0,633.

---

## 1. B.1 — Sınıflandırıcı radar çipinin belleğine sığar mı? (en değerli)
- **Soru:** B2'deki 300 ağaçlık rastgele ormanı küçültürsek doğruluk ne kadar düşer, model kaç KB'ye iner?
- **Yöntem:** 30 yapılandırma: 10–300 ağaç × derinlik 4–12 ya da sınırsız. Seçim kuralı önceden yazıldı: F1 ≥ 0,836 ve AUROC ≥ 0,894 olan en küçük model.
- **Seçilen model:** 10 ağaç, derinlik sınırsız.
  - F1 0,846 ± 0,015; B2'ye göre eşli fark −0,010 ± 0,003. Bu F1 yumuşak oylamayla (`predict_proba` ortalaması) ölçüldü.
  - AUROC 0,937, FAR 0,039, tespit 0,535.
  - Boyut (tam veriyle eğitilen model): 20.460 düğüm. Düğüm başına 8 B ile 160 KB, 12 B ile 240 KB.
  - Pencere başına ortalama 121 karşılaştırma.
- **Karşılaştırma için B2, aynı tam veri temelinde:** 612.946 düğüm, 4,9 / 7,4 MB (8 / 12 B), pencere başına 3.572 karşılaştırma.
- **Çip belleği** (TI veri sayfası SWRS219F): IWR6843'te toplam 1,75 MB. Bunun içinde MSS veri RAM'i 192 KB, DSP L2 256 KB.
  - Seçilen model 256 KB'ye her iki varsayımla sığıyor.
  - 192 KB'ye yalnız 8 B/düğüm varsayımıyla sığıyor. **Uyarı:** 8 B düzeni yaprakta yalnız sınıfı tutar, yani çipte sert oylama gerekir. Yukarıdaki F1 ise yumuşak oylamayla ölçüldü; sert oylamanın doğruluğu B'de değerlendirilmedi. Bunu C.1 ölçüyor.
  - 128 KB'ye sığan en iyi model 10 ağaç × derinlik 12 (F1 0,823). Bu yalnız 8 B/düğüm ya da kat ortalaması boyutu için geçerli. Tam veri modeli 12 B ile 136 KiB tutuyor ve sığmıyor; o temelde en iyisi 10 × derinlik 10 (F1 0,807).
- **Okuma:** Sınıflandırıcının sabitleri kâğıt üzerinde çip içi bir bellek bankasına sığacak kadar küçük. Bu, durum sınıflandırmasının çipte yapılabileceğini düşündürüyor ama göstermiyor.
  - Öznitelik çıkarımı, kare tamponları ve anomali katmanı boyutlandırılmadı. Anomali katmanı: debounce, döngü bölütleme, D1/D2 skorları ve conformal kalibrasyon.
  - Bellek, aygıt yazılımı ve radar küpüyle paylaşılıyor.
  - Hiçbir şey çipte çalıştırılmadı.
- **Sınırlar:**
  - Konfigürasyonlar arası AUROC farkları kat gürültüsü içinde (sd 0,04–0,19). Seçilen modelin AUROC'unun ve FAR'ının B2'den "iyi" görünmesi bir iyileşme değil.
  - Seçim F1 eşiğine 0,010 farkla bağlı. Eşik ≥ 0,847 olsaydı 25 ağaç × sınırsız seçilirdi (F1 0,852, 42 bin düğüm).
- Dosyalar: `reports/faz_b/01_tiny_forests.md`, `.csv`, `.png`.

## 2. B.2 — Nokta bulutunun ne kadarı gerekli? (algılama bütçesi)
- **Yöntem:** Her seferinde tek eksen değiştirildi: SNR eşiği, hız bilgisi, nokta seti, tepe gruplama. Veri ham `mmwave_data.csv`'den yeniden kuruldu; kapı bit düzeyinde birebir geçti.
- **Diz kuralı:** Önceden yazılan kural "F1 düşüşü ≤ 0,02, AUROC düşüşü ≤ 0,03, kat GA'ları daha büyük düşüşü dışlamalı". Uygulamada dört koşul: ΔF1 ≥ −0,02, F1 GA alt sınırı ≥ −0,02, ΔAUROC ≥ −0,03, AUROC GA alt sınırı ≥ −0,03.

| Düzey | F1 | ΔF1 [%95 GA] | AUROC | Nokta/kare (taban %) | Bayt/s | Ham ADC'ye oran |
|---|---|---|---|---|---|---|
| taban (Faz A) | 0,856 | 0 | 0,924 | %100 | 1.573 | 365× |
| SNR ≥ 20 dB | 0,853 | −0,003 [−0,008; +0,001] | 0,976 | %92 | 1.453 | 395× |
| SNR ≥ 25 dB | 0,843 | −0,013 [−0,020; −0,007] | 0,927 | %81 | 1.291 | 444× |
| hız yok | 0,814 | **−0,042** [−0,047; −0,037] | 0,973 | %100 | 1.291 | 444× |
| hız işareti (\|v\| yerine ort. sign v) | 0,867 | **+0,011** [+0,006; +0,016] | 0,898 | %100 | 1.361 | 421× |
| tepe gruplama açık | 0,846 | −0,011 [−0,014; −0,008] | 0,970 | %73 | 1.198 | 479× |
| ham noktalar (v=0 dahil) | 0,848 | −0,008 [−0,014; −0,003] | 0,932 | %125 | 1.929 | 297× |
| yalnız v=0 noktalar | 0,776 | −0,081 | 0,897 | %25 | 484 | 1.186× |

### Okuma
1. **SNR eşiği 20 dB'ye kadar:** F1 kaybı küçük (~0,004) ve 0,02 toleransının içinde. Ama kayıp sistematik: 4 tohumun hepsinde negatif. Tepe gruplama da tek başına dört koşulu geçiyor.
   - Bu iki azaltma **birlikte** uygulanınca kural geçilmiyor: ΔF1 −0,017, GA alt sınırı −0,021.
   - SNR kesimi yalnız menzil-CFAR'ı yaklaşık taklit eden bir öykünme. Çipte eşik değişirse CFAR'ın gürültü tabanı ve algılanan nokta kümesi de değişir. Bu yüzden "çipte 20 dB'ye çıkarılabilir" denemez.
2. **Hız bilgisi durum tahmini için gerekli.** Çıkarılınca F1 4,2 puan düşüyor.
   - Buna karşın anomali AUROC'u yükseliyor: tek tohumda +0,049, dört tohum ortalamasında +0,038, tohumların hepsinde pozitif. Bu ayrışmanın nedeni açıklanmadı.
3. **Veri akışı** düzeye göre 0,5–1,9 kB/s, ham ADC'den 297–1.186 kat az.
   - Makaledeki ~290× (ham noktalar) ile tutarlı: B.2'de ham noktalar için oran 297×.
   - Bayt hesapları TI SDK mesaj düzeninden yapıldı, hatta ölçülmedi. İşaret ve hızsız düzeyler standart dışı aygıt yazılımı varsayıyor.
4. **Hız işareti (keşif):** "Yalnız işaret" düzeyi planda vardı. Ama uygulama biçimi (|v| yerine ortalama sign v) ve F1 kazancı sonradan geldi.
   - F1 +1,1 puan, tek tohum GA'sı sıfırı dışlıyor. Doğrulayıcıya göre 5 katlı t-GA iyimser olabilir.
   - Aynı düzey AUROC'u düşürüyor: −0,026 tek tohumda, −0,035 dört tohumda. Kuralı geçemediği nokta da bu.
   - Sonuç görüldükten sonra B2'ye eklemek seçim yanlılığı olur; en fazla "gelecek iş".
5. **AUROC artışları:** Tek tohumda görülen +0,05'lik artışlar (hız yok, tepe gruplama, SNR 17,5) dört tohumda yaklaşık üçte bir küçülüyor ama pozitif kalıyor; SNR 20 karışık. Hiçbiri 2 SE'yi ~1,2 katından fazla aşmıyor. İyileşme olarak okunmamalı.

### Önceden yazılmış beklentiler
- (c) Ham ve birleştirilmiş noktalar arasında |ΔF1| ≤ 0,01: **tuttu** (0,0084).
- (d) Tepe gruplamada |ΔF1| ≤ 0,01: **tutmadı** (0,0107).

### Sınırlar
- Harfiyen okunan kural pratikte bir AUROC GA genişliği testine dönüşüyor (~50 normal döngü, B.2'de kat sd 0,02–0,09).
- Tepe gruplama, TI'ın range-Doppler işleminin bir vekili.

Dosyalar: `reports/faz_b/02_sensing_budget.md`, `.csv`, `.png`, `02_level_seeds.csv`, `02_noise_floor.csv`.

## 3. B.3 — Kare hızı düşürülebilir mi?
- **Yöntem:** 10 Hz kayıttan her k. kare alındı (k=2 → 5 Hz, k=3 → 3,3 Hz, k=5 → 2 Hz).
- **Pencere:** L = ⌈5/k⌉ kare. Kapsanan süre k=2'de 0,5 s, k=3'te 0,4 s, k=5'te 0,1 s (tek kare).

| k (Hz) | F1 | ΔF1 [%95 GA] | AUROC | Tespit | FAR | Bayt/s |
|---|---|---|---|---|---|---|
| 1 (10) | 0,856 | 0 | 0,924 | 0,633 | 0,078 | 1.573 |
| 2 (5) | 0,826 | −0,030 [−0,036; −0,024] | 0,847 | 0,192 | 0,098 | 787 |
| 3 (3,3) | 0,802 | −0,055 [−0,061; −0,049] | 0,847 | 0,245 | 0,078 | 524 |
| 5 (2) | 0,708 | −0,149 [−0,177; −0,121] | 0,797 | 0,260 | 0,038 | 315 |

- **Diz kuralı:** Hiçbir k geçmiyor; **tam 10 Hz gerekli**. Bu karar F1'e dayanıyor: k=2'de her katta ΔF1 < −0,02.
  - Diz 10 ile 5 Hz arasında; ara hızlar denenmedi.
- **Anomali tespiti:** Önceden belirlenen, süreye sabitlenmiş debounce kuralıyla 5 Hz'te tespit düşüyor. Ama düşüşün büyüklüğü debounce yuvarlamasına bağlı:
  - aynı tahminlerle kareye sabitlenmiş 2/1 debounce kullanılınca k=2'de AUROC 0,983, tespit 0,563, FAR 0;
  - yani bu düşüş kare hızının değil yuvarlama kuralının sonucu olabilir.
  - Ek etkenler: etiketler 10 Hz'te düzleştirildi ve kalış süreleri 0,1·k s'ye nicemlendi. Bunlar D0/D2 süre skorlarını etkiler.
- **Uzun dokunuşlar** (ToF'a göre ≥ 0,5 s; s02–s04 birlikte, 347 dokunuş):
  - 10 ve 5 Hz'te 347/347; 3,3 Hz'te 0,974; 2 Hz'te 0,942.
  - Oturum bazında 2 Hz'te s04 0,899'a düşüyor. A3.2'nin oturum bazlı "≥ 0,9" tahmini yalnız burada tutmuyor.
- **Okuma:** Durum sınıflandırması 10 Hz istiyor. Uzun dokunuşları yakalamak daha düşük hızlarda da mümkün.
- **Sınırlar:**
  - Kare hızı ile pencere uzunluğu birlikte değişti.
  - Gerçek düşük hızlı bir radar farklı bir chirp ayarı kullanırdı.
  - Tek tohumla ölçülen tespit gürültülü: k=1'de tohumlara göre 0,65–0,87.
- **Düzeltilen hata:** "s02–s04" satırları başlangıçta yalnız s02'yi içeriyordu; `03_frame_rate.py:266`'da iki oturum adı yanlıştı. Düzeltilip yeniden çalıştırıldı ve yalnız bu satırlar değişti. Eski çıktılar `<LOCAL_BACKUP>/bk_b3/` altında.
- Dosyalar: `reports/faz_b/03_frame_rate.md`, `.csv`, `.png`.

## 4. B.0 — Hakemin kalan üç maddesi (referee.md e–g)
**(e) D3'e alternatif olarak D0+D1 birleşimi:**
- Toplamda D3'e eşit: tespit 0,631 vs 0,633; FAR ikisinde de 4/51 = 0,078; AUROC 0,908 vs 0,924.
- Belirgin tek fark hızlı tempoda: D0+D1 0,700, D3 0,921.
- Sıra anomalilerindeki fark (0,417 vs 0,356) gürültü içinde: Wilson aralıkları örtüşüyor ve sıra AUROC'u neredeyse aynı (0,898 vs 0,897).
- **Okuma:** D3'ü tutmak için gerekçe var. Makaleye "denendi, toplamda fark yok" cümlesi olarak girebilir.

**(f) Eksik kare içeren pencereler — yeni bir veri gerçeği:**
- Kayıttaki `frame_id` ardışık bir sayaç, ama zaman damgalarında boşluklar var: 696 kare aralığı 0,15 s'den uzun, en uzunu 8,4 s.
- Pencerelerin **%13'ü** böyle bir boşluğu kapsıyor ve bunların %98'i "kutu yok" sınıfında. Neden henüz incelenmedi; C.2 inceliyor.
- Bu pencereler dışlanınca:
  - F1 küçük ölçüde düşüyor: B2'de 0,856 → 0,852, CNN-GRU'da 0,841 → 0,837.
  - **Kare doğruluğu 1,6–1,7 puan düşüyor:** B2'de 0,884 → 0,868. Sebebi, boşluklu pencerelerin neredeyse hepsinin kolay "kutu yok" pencereleri olması; bu pencerelerin kendi doğruluğu 0,99.
  - Makale B2 doğruluğunu %88,4 olarak veriyor; bu, boşluklu pencerelerin dahil olduğu değer.
- Döngü süreleri zaman damgasından hesaplanıyor (`a2_cycles.build_cycles`, `dur_s`), yani süre birimleri bozulmuyor. Ancak boşlukların anomali skorlarına etkisi **test edilmedi**: debounce kare sayısıyla çalışıyor ve bir boşluğun üstünden köprü kurabilir.
- **Okuma:** Veri bölümüne bir açıklama cümlesi gerekli: "5 karelik pencere her zaman 0,5 s değildir". Doğruluk rakamının yanına bu not düşülmeli. Bu bir dürüstlük meselesi; önerim eklemek.

**(g) Zaman sıralı kalibrasyon (sürüklenme testi):**
- **Tasarım gereği bilgisiz.** n_cal=15 iken en küçük D3 p-değeri 0,125; α=0,10'da hiç alarm çıkamıyor. "Sürüklenme saptanmadı" sonucu kanıt olarak KULLANILMAMALI.
- Bileşenlerle tek tek bakıldığında (D0, D1, D2): izleme döngülerinde yanlış alarm A düzeninde 0–1/21, B düzeninde 0–1/16.
  - Ama bu testlerin saptayabileceği en küçük FAR artışı +0,14 (A) ile +0,21 (B). Yani "sürüklenme işareti yok" zayıf bir kanıt.
- D1 B2 × pred hücresinde gücünü kaybediyor (sıra tespiti 0).
  - Neden: kalibrasyondaki 2 döngüde sahte bir tahmin geçişi var ve bunlar en yüksek skoru eşliyor.
  - ToF × tof hücresinde aynı n_cal ile D1 sıra anomalilerinin hepsini yakalıyor.
- **Okuma:** En fazla sınırlamalar bölümüne bir cümle: "küçük kalibrasyon setleri conformal çözünürlüğü sınırlar (en küçük p = 1/(n+1))".

---

## Senin kararın gereken noktalar
1. **B.1 makaleye girsin mi?** Konu: sınıflandırıcının sabitleri çip belleğine sığıyor.
   - Önerim: evet, ama C.1'in derlenmiş gerçek boyut ve sert oylama sonuçlarıyla birlikte.
   - Kısa bir alt bölüm olur. "Yalnız sınıflandırıcı boyutlandırıldı, çipte çalıştırılmadı" açıkça yazılır.
2. **B.2 makaleye girsin mi?** Konular: hız durum tahmini için gerekli, SNR öykünmesi, ~0,5–1,9 kB/s veri akışı.
   - Önerim: bir paragraf, tablo ek dosyada.
   - Hız işareti bulgusu yalnız "gelecek iş" olarak.
3. **B.3 makaleye girsin mi?** Konu: 10 Hz gerekli. Önerim: ek dosyada bir paragraf, debounce uyarısıyla.
4. **Veri boşluğu notu (f) ve D0+D1 cümlesi (e) eklensin mi?** Önerim: ikisi de ana metne birer cümle; (f) doğruluk rakamının yanına.

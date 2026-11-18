# Faz A3 — "Ne kadarı yeter, sınırlar nerede?" (ÖN-KAYIT, 2026-11-04)

## Statü (A2 ile aynı; değiştirilemez)
- Faz A sonuçları ve A2'nin bir kısmı görüldü. A3, Faz A'ya göre **sonradan** yapılan **ikincil** analizlerdir.
- Fikirler Fable'ın yenilik taramasından geliyor (`reports/novelty/NOVELTY_SCOUT.md`, yargıç sırası 1–3). Kullanıcı üçünü onayladı (2026-11-04).
- **Birincil sonuçlar Faz A'nın ön-kayıtlı sayılarıdır; ana model B2 + D3 değişmez.** Burada tanımlanan bütün sayılar raporlanır; seçici raporlama yok.
- Yeni veri yok, yeniden eğitim yok (yalnız mevcut OOF tahminleri ve ToF etiketleri).
- Faz A ve A2 dosyalarının üzerine yazılmaz; çıktılar `reports/faz_a3/` altına gider.
- Kod `faz_a2/a2_cycles.py` modülünü kullanır (Faz A 04'ü birebir yeniden ürettiği A2.0 kapısında doğrulandı). Koşular `tools/local_run.sh` / `tools/local_job.py` ile yerel makinede yapılır.
- **Yasak:** s02'deki 4 doğal sapma döngüsünün Faz A'da alarm alıp almadığına bakmak (o bir sonuçtur; bu ön-kayıtta yer almıyor).
- Sapmalar en altta tarihli olarak tutulur.

## A3.1 — Döngü düzeyi anomali tespiti ne kadar kare doğruluğu gerektiriyor? (hata yayılımı)
**Soru:** B2 ile CNN-GRU arasındaki 1,5 puanlık macro-F1 farkı, D3 AUROC'unda 4,5 puanlık farka nasıl dönüşüyor? Hangi hata oranında ve hangi hata yapısında dedektör çöküyor?

**Kol (a) — analitik model:**
- S1 OOF'tan her durum kaynağı (B2, CNN-GRU, CNN-only, B1) için hata koşusu uzunluğu dağılımları çıkarılır.
- Oturum başına iki oran tahmin edilir:
  - sahte ziyaret oranı: none ya da başka kutu segmenti içinde ≥ MIN_BOX = 2 karelik hata patlaması;
  - döngü bölünme oranı: ≥ 2 karelik sahte Box1 koşusu.
- Tahminler gözlenen değerlerle karşılaştırılır (`reports/faz_a/04_cycle_counts.csv` ve ziyaret dizileri).
- **Geçme ölçütü:** dört kaynağın hepsinde |tahmin − gözlenen| ≤ 0,05.

**Kol (b) — simülasyon (dağıtılan yolun tamamı bozulur):**
- ToF-oracle kare etiketleri r ∈ {0,02; 0,05; 0,10; 0,15; 0,20; 0,30} oranında bozulur. İki gürültü tipi:
  - (i) bağımsız çevirmeler (iid);
  - (ii) geometrik patlamalar. Ortalama patlama uzunluğu **yalnız B2'den** uydurulur; **diğer kaynakların** gözlenen noktalarıyla sınanır (kaynaklar arası; kendi kendini tahmin yok).
- Bozuk bir karenin yeni etiketi, kaynağın gözlenen karışıklık matrisinden çekilir: B2 için B2'nin, iid için düzgün.
- Döngüler **bozuk etiketlerden** kesilir (debounce + Box1 girişi, pred segmentasyonu). D3 aynı r'deki bozuk normal döngülerle kalibre edilir.
- Faz A katları, R = 50 tohum, α = 0,10.
- **Raporlanan:** r'ye karşı AUROC (tümü ve tür başına) ve normal FAR, %95 aralıklarla.
- **Sabit karar kuralı:** "yeterli doğruluk" = AUROC ≥ 0,90 ve FAR ≤ α + 2SE sağlayan en büyük r. AUROC SE'si 0,03–0,05 olduğu için tek nokta değil, bant olarak raporlanır.

**Önceden yazılmış tahminler:**
- Patlamalı gürültü, iid'den daha düşük r'de bozar.
- Gerçek CNN-GRU ve CNN-only noktaları, B2'den uydurulan patlamalı eğrinin 0,03 AUROC yakınındadır.

**Önceden yazılmış uyarılar:** Gürültü sentetik; gerçek hatalar geçiş yapılı; tek operatör.

- Çıktı: `reports/faz_a3/01_error_propagation.md` (+ csv, şekil).
- **Desteklediği cümle:** "Kalibre edilmiş prosedürel anomali dedektörü kare düzeyinde ~%X hata oranına kadar dayanıyor; hatanın oranı kadar patlama yapısı da önemli. Bu, 1,5 puanlık bir durum farkının neden 4,5 puanlık bir AUROC farkına dönüştüğünü açıklıyor."

## A3.2 — Çalışma zarfı: sistem neyi göremiyor?
**Birim ve tanımlar:**
- **Birim:** ToF dokunuşu = 30 Hz'de mesafe < 140 mm olan kesintisiz koşu (kutu ve oturum başına). Eşiğin üstünde ≥ 1 örnek dokunuşu böler.
- **Tespit tanımları (önceden sabit):**
  - **D-ham:** Son karesi dokunuşun içine düşen en az bir S1 OOF penceresi o kutuyu tahmin ediyor.
  - **D-deb:** Dokunuş, OOF tahmin akışına uygulanan `debounce_labels` (MIN_BOX = 2, MAX_GAP = 1) sonrasında bir ziyaret olarak kalıyor. D1/D2'yi yöneten budur ve öyle etiketlenir.
- **Süre kutuları:** < 0,2; 0,2–0,3; 0,3–0,5; 0,5–1,0; > 1,0 s.
- Oturum başına raporlanır, havuzlanmaz: kısa dokunuşların neredeyse hepsi s05 Box5 ve s01 Box4'te.
- Model başına raporlanır: B2 birincil; CNN-GRU, CNN-only, B1 ikincil. Güven aralığı Wilson %95.
- Oturum başına "tespit ~ süre" lojistik uydurmasından **d50** ve **d90** çıkarılır.

**Önceden yazılmış tahminler:**
- B2'de d50 ∈ [0,2; 0,4] s (D-ham).
- Normal tempolu oturumlarda > 0,5 s dokunuşlar için tespit ≥ 0,9.
- D-deb'de d50 daha yüksek çıkar; dağıtım sınırı olarak raporlanır.
- d50 > 0,4 s çıkarsa beyan edilen sınır gözlenen d90'a yükseltilir. Yeniden ayar yapılmaz.

**Konum ekseni (yalnız betimleyici):**
- 14 komşu kutu çifti için ikili karışıklık, `b2__S1` OOF'tan hesaplanır; üçüncü sınıf tahminleri dışarıda bırakılır.
- Kutu aralığına karşı çizilir, 11,8 cm çözünürlük çizgisiyle. **Uydurulmuş bir tasarım sabiti yok.**
- s04 Box4–Box5 (8,2 cm) ayrıca raporlanır.

**Önceden yazılmış uyarılar:**
- Bu sınır **eğitilmiş sistemin** sınırıdır, radarın değil (eğitim etiketleri de kısa dokunuşları silmişti).
- L = 5 pencere ve hızlı tempo hareketleriyle karışık.
- Tek operatör, tek mesafe.

- Çıktı: `reports/faz_a3/02_envelope.md` (+ csv, şekil).
- **Desteklediği cümle:** "Sistem ≥ ~12 cm aralıklı kutularda ve ≥ d90 s süren dokunuşlarda güvenilir; daha kısa dokunuşlar ve daha yakın kutular sınırın dışında."

## A3.3 — Döngü içi (çevrimiçi) alarm: yanlış alarm garantisi birebir korunarak
**Kurulum:**
- **Hücreler:** B2 × pred (dağıtılabilir, ana); referans olarak ToF × tof.
- **Olay tanımları (önceden sabit):**
  - a → b geçişi, b ziyareti onaylandığında tamamlanır (debounce sonrası ≥ MIN_BOX kare).
  - dwell_b, b ziyareti bittiğinde skorlanır.
  - Boşluk, bir sonraki ziyaret onaylandığında skorlanır.
  - → END geçişi ve son dwell yalnız bir sonraki Box1 girişinde skorlanır.
- Tamamlanan olaylar üzerinden D1/D2'nin koşan maksimumu alınır. p değerleri **değişmemiş Faz A kalibrasyon skorlarından** hesaplanır.
- Alarm, 2·min(p1, p2) ≤ 0,10 olan ilk olayda verilir.

**Sert geçme/kalma ölçütü:**
- Alarm alan döngü kümesi, her katta Faz A'nın çevrimdışı D3 alarm kümesiyle **birebir aynı** olmalı.
- Değilse analiz durur ve rapor edilir; yama yapılmaz.

**Metrikler (anomali türü başına):**
- Alarm anında geçmiş döngü oranının medyanı ve IQR'ı.
- seq135/seq145 için gerçek başlangıçtan itibaren saniye. Başlangıç = Box3/Box4'e ilk ToF girişi (`merged_df` box3_active/box4_active).
- Döngünün son ziyaretinden önce verilen alarmların payı.
- Saniyeler OOF dosyalarındaki kare zaman damgalarından hesaplanır.

**Önceden yazılmış tahminler:**
- seq135/145'te medyan oran < 0,5.
- fast/slow'da medyan oran > 0,8 (süre skorlamasının doğası gereği geç). Bu bir özellik olarak raporlanır.

Bu turda aşım (exceedance) kuralı yok.

- Çıktı: `reports/faz_a3/03_online.md`.
- **Desteklediği cümle:** "Aynı yanlış alarm garantisiyle sıra sapmaları döngünün ~ilk yarısında yakalanıyor; tempo sapmaları ancak süre ölçülünce, döngü sonuna doğru."

## Sapmalar (tarihli)
- 2026-11-04 (A3.3, `faz_a3/03_online.py`): belirtilmemis ayrintilar su sekilde dolduruldu: zaman = kare tablosu `t` (kare zaman damgasi); onay karesi = debounce'lu kosunun MIN_BOX'uncu karesi, ziyaret sonu = ilk ziyaret-sonrasi kare; ->END ve son dwell = sonraki Box1 ziyaretinin onay karesi (ToF x tof'ta dongu sonrasi ilk Box1 ziyareti; oturumun son dongusunde sonraki giris yoksa oturumun son karesi); baslangic = dongu araligindaki ilk kare icin merged_df box3_active/box4_active kare bayragi; (dongu,f) ciftleri havuzlu; ek olarak nedensel duyarlilik varyanti (onay = ham etiketin MIN_BOX'uncu karesi, ziyaret sonu e+MAX_GAP) raporlandi; dogal-sapma donguleri ayristirilmadi.
- 2026-11-04 (A3.1, `faz_a3/01_error_propagation.py`; sonuçlar görülmeden sabitlendi): (1) "kare hata oranı" = L0 (`frame_table.y_true`) != OOF `y_pred` olan karelerin payı, s02-s06'nın TÜM kareleri üzerinden (oturum başına pencere-siz ilk 4 kare `source_labels` gibi none sayılır ve L0 != none ise hata sayılır); B2 karışıklığı/hata koşuları aynı kare kümesinden. (2) Kol (a): oranlar = olay sayısı / oturumun ToF referans döngü sayısı (`n_tof_ref`). BİRİNCİL gözlenen: debounce sonrası pred ziyaretlerinden, aynı sınıftan hiçbir L0 koşusuyla kare kesişimi olmayanlar = "sahte ziyaret"; bunların Box1 olanları = "döngü bölünmesi". İKİNCİL gözlenen (ayrıca raporlanır): `04_cycle_counts.csv` `n_loops_with_2plus/n_tof_ref`. Analitik tahmin = havuzlanmış (s02-s06) "sabit yanlış kutu etiketli, uzunluk >= 2" koşu uzunluğu dağılımından kare başına tehlike x oturum kare sayısı / n_tof_ref (debounce birleştirmesi modellenmez); LOSO tehlike ek. GEÇME (sıkı) = dört kaynağın hepsinde, tüm oturumlarda ve iki oranda max |tahmin-gözlenen| <= 0,05 (birincil gözlenene göre). (3) Kol (b): iid yedek etiket = doğru etiket dışındaki 5 sınıftan düzgün (gerçekleşen oran = r); patlama = B2 hata koşularının (p != y) ortalama uzunluğundan geometrik (ölçüm: B2 hata koşularının >=2 uzunluklu olanlarının %95,4'ü tek sabit yanlış etiketli -> patlama başına TEK ortak U ile B2'nin hata-koşullu karışıklık satırından ters-CDF etiket); patlama/boşluk dönüşümlü geometrik, durağan kapsama = r. (4) AUROC = D3, kat bazında ortalama (Faz A gibi), 'tum anomali' + 4 tür; FAR = D3 p <= 0,10, havuz normal test çiftleri (SE tohum başına n_pairs ile); "95% aralık" = 50 tohum üzerinden 2,5-97,5 yüzdelik. (5) Bant: nokta = ortalama AUROC >= 0,90 ve ortalama FAR <= alfa+2SE sağlayan en büyük ızgara r; alt sınır = aynı kural AUROC'un tohum-2,5 yüzdeliğiyle; üst sınır = 97,5 yüzdelikle. (6) Tahmin 1 geçer <=> r*_patlamalı < r*_iid (nokta kuralı); ek olarak her r'de AUROC_patlamalı < AUROC_iid raporlanır. Tahmin 2: Faz A D3 pred 'tum anomali' auroc_mean ile B2-uydurmalı patlamalı eğrinin ızgarada doğrusal enterpolasyonu |fark| <= 0,03; B2 noktası kendi kendine uydurma kontrolüdür (tahmin değil); B1 (r=0,376) ızgara dışındadır. (7) Ek (ızgara dışı, saf tanılama): r=0 kapı noktası (ToF/pred D3 Faz A'yı birebir üretmeli) ve gözlenen r_obs noktalarında doğrudan simülasyon.
- 2026-11-04 (A3.2, `faz_a3/02_envelope.py`; sonuclar gorulmeden sabitlendi, tek istisna asagida): belirtilmemis ayrintilar su sekilde dolduruldu: (1) dokunus suresi = t[son ornek] - t[ilk ornek] + oturumun medyan ToF dt'si; NaN mesafe esik-ustu sayilir (dokunusu boler); (2) ToF->kare hizalamasi A2.3 ile ayni `(t_kare-100 ms, t_kare]` kurali, bir dokunusun kare kumesi = en az bir orneginin atandigi kareler; kare kumesi bos dokunuslar icin tespit tanimsiz sayilip paydadan dusulur (n_nomap raporlanir; bu veride 0/1111, kural bagayici olmadi); (3) D-ham/D-deb "kare kumesinde en az bir karenin (ham / debounce'lu) tahmini o kutu"; s01 dahil 6 oturum, s01 akisi `ac.labels_from_oof` s01'i dusurdugu icin elle kuruldu (s02-s06 icin modulle birebir ayni dogrulandi); (4) sure kutulari sol-kapali [lo, hi), son kutu >= 1.0 s; (5) lojistik = duzenlilestirmesiz ML, sure dogrusal saniye; tahmin edilemez = n=0 / tum sonuclar ayni / tam ayrisma / b1<=0 / yakinsamama; bu durumlarda d50 icin aralik ([en uzun kacirilan, en kisa yakalanan] ya da [0, min sure]) verilir; d50 gozlenen sure araligi disindaysa (post-hoc eklenen tek istisna: s02 D-deb, B1/CNN-GRU/CNN-only'nin bazi hucrelerinde negatif d50 cikti) nokta degeri "ekstrapolasyon" isaretlenir ve dogrulamalarda yalniz sinir ([0, min] / [max, inf)) olarak kullanilir; (6) "normal tempolu" = s02, s03, s04 (verdikt kumesi; s01/s05/s06 bilgi); (7) "beyan edilen sinir" baslangici = on-yazili d50 araliginin ust ucu 0.4 s, d50 > 0.4 s ise o oturumun gozlenen d90'i (tam ayrismada aralik ust ucu); (8) mekansal eksende kutu araligi `probe/01_dwell_modes.py adjacent_pairs` ile yeniden hesaplandi (q1.md ile 14/14 uyumlu), karisiklik dogru etiketi = OOF y_true (L0), ucuncu sinif (none dahil) tahminleri disarida, sim_hata = (a->b + b->a)/(kalan a + kalan b).
- 2026-11-04 (A3.1, POST-HOC, ilk tam koşudan sonra eklendi): yukarıdaki sıkı kol (a) kuralı (tüm gözlenen-tahmin farkları ≤ 0,05, havuzlanmış tehlike) dört kaynakta da FAIL verdi (tahmin, sınır-kayması koşularını da sahte ziyaret saydığı için gözlenenden çok büyük). Birincil karar FAIL olarak kalır; ek olarak, brief'teki "none ya da başka kutu segmenti İÇİNDE hata patlaması" tanımına sadık ikincil bir varyant raporlanır: yanlış-kutu koşusunun her iki komşu L0 karesi de o kutu değilse ("iç patlama") sayılır (sınır-kayması dışlanır); aynı 0,05 ölçütü ikincil olarak uygulanır. Ayrıca betimleyici: hata karelerinin `near_change` (L0 geçişinin ±2 karesi) payı. Birincil karar / ızgara / ön-yazılı tahminler değişmedi.

# Faz B — Gece keşif denemeleri (ANALİZ PLANI, 2026-11-05 00:20)

## Statü
- **Keşif amaçlı.** Plan koşudan önce yazıldı ama Faz A/A2/A3 sonuçları görüldükten sonra.
- Tanımlanan her sayı raporlanır. Sonuçlar **kullanıcı görmeden makaleye girmez** (`reports/faz_b/`, sabah özeti).
- Yeni veri yok. Faz A–A4 dosyalarının üzerine yazılmaz.
- Kod `faz_b/` altında; yeniden kullanılan modüller `faz_a/fa_common.py` ve `faz_a2/a2_cycles.py`. Koşular `tools/local_run.sh` / `tools/local_job.py` ile.
- CPU: Keras işleri çalışırken n_jobs ≤ 3, OMP ≤ 3.
- s02 doğal sapma döngülerinin alarm durumuna bakmak yasak.
- Fikirler `reports/novelty/NOVELTY_SCOUT.md`'den (yargıcın "ayrı makale / beklet" dediği maddeler) ve hakem raporunun kalan maddelerinden (`paper/review/referee.md` e–g).
- Sapmalar en altta tarihli olarak tutulur.

## B.0 — Hakem raporunun kalan maddeleri (makaleye girebilecek; dakikalık iş)
**(e) D0 + D1 kombinasyonu:**
- B2 × pred için 2·min(p0, p1) (Bonferroni), α = 0,10 ve 0,05.
- FAR (Wilson), tür başına tespit, AUROC; D3'ün yanında verilir.

**(f) Eksik kare içeren pencereler:**
- Pencerelerin hangi payı boş kare ya da frame_id boşluğu içeriyor? (fa_common `windows()` → `contiguous` ve kare tablosu.)
- B2, CNN-GRU ve B1 için S1 metrikleri: bu pencereler dahil ve hariç.

**(g) Zaman sıralı kalibrasyon (sürüklenme):**
- s02'nin 51 normal döngüsü zaman sırasıyla iki düzende bölünür:
  - (A) uydurma 1–15 / kalibrasyon 16–30 / izleme 31–51;
  - (B) uydurma 1–20 / kalibrasyon 21–35 / izleme 36–51.
- Anomali oturumları her zaman testtir. Hücreler B2 × pred ve ToF × tof; D3, α = 0,10.
- Raporlanan: izlemedeki FAR (Wilson) ve tespit.
- **Önceden yazılmış okuma:** iki düzende de FAR α ± 2SE içindeyse "sürüklenme saptanmadı" denir. Bu örneklemde saptanabilir en küçük FAR artışının ~0,15 olduğu yazılır.
- "Yorgunluk" kelimesi kullanılmaz.

Çıktı: `reports/faz_b/00_referee_leftovers.md`.

## B.1 — Küçük ormanlar: model radar çipinin belleğine sığar mı? (tahmin; port yok)
- **Izgara:** n_trees ∈ {10, 25, 50, 100, 300} × max_depth ∈ {4, 6, 8, 10, 12, None}. Özellikler, tohum ve sınıf ağırlığı B2 ile aynı; S1 5 kat.
- **Raporlanan:**
  - S1 macro-F1 (ortalama ± sd);
  - toplam düğüm sayısı;
  - model boyutu, düğüm başına 8 B ve 12 B varsayımıyla;
  - pencere başına karşılaştırma sayısı (ağaç derinliklerinin toplamı);
  - B2 × pred D3 AUROC ve FAR. Anomali hattı her yapılandırmanın OOF tahminleriyle, Faz A protokolüyle koşulur.
- **Bellek çizgileri:** 128 / 256 / 512 KB. IWR6843'ün çip içi RAM'i TI veri sayfasından alınır, kaynağı yazılır.
- **Önceden yazılmış seçim kuralı:** macro-F1 ≥ 0,856 − 0,02 **ve** AUROC ≥ 0,924 − 0,03 olan en küçük (bayt) yapılandırma. Yoksa "sığan yapılandırma yok" denir.
- B2 birincil model olarak kalır; iki sayı yan yana verilir.

Çıktı: `reports/faz_b/01_tiny_forests.md`.

## B.2 — Algılama bütçesi: çip içi nokta bulutunun ne kadarı yetiyor? (öykünme)
- **Kare istatistikleri** her seviyede ham `mmwave_data.csv`'den yeniden hesaplanır: cx, cy (SNR-ağırlıklı), n_points, snr_sum, mean_abs_v, std_x, std_y.
- **Kapı:** "merged / tam" seviyesi fa_common `frame_table()` istatistiklerini birebir yeniden üretmeli. Üretemezse B.2 durur.
- **Eksenler ve seviyeler** (önceden sabit):
  - (a) SNR eşiği τ ∈ {15 (kayıttaki), 17,5; 20; 22,5; 25 dB}: eşiğin altındaki noktalar atılır.
  - (b) Hız: tam / yalnız işaret / tamamen çıkarılmış. Çıkarılmışta mean_abs_v sabit tutulur.
  - (c) Noktalar: merged (v ≠ 0, Faz A) / ham, v = 0 dahil / yalnız v = 0.
  - (d) Tepe gruplama: kapalı (kayıttaki) / açık. Açıkta her (menzil, açı) hücresinde yalnız en yüksek SNR'li nokta tutulur.
- **Her seviyede:** B2 S1 5 katta yeniden eğitilir; macro-F1 Faz A B2 ile eşli kat farkı olarak verilir. Anomali hattı (pred, D3, α = 0,10) AUROC ve FAR verir. Ayrıca bayt/kare ve bayt/s hesaplanır (TLV düzeni `reports/paper_prep/05_gaps.md`'deki varsayımlarla).
- **Önceden yazılmış diz kuralı:** en yalın seviye, macro-F1 düşüşü ≤ 0,02 ve AUROC düşüşü ≤ 0,03 sağlayan seviyedir (kat GA'ları daha büyük bir düşüşü dışlamalı). Dize ulaşamayan eksen için "tam bütçe gerekli" denir.
- **Önceden yazılmış beklentiler:**
  - (c) ham ile merged arasında |Δ macro-F1| ≤ 0,01.
  - (d) gruplama açıkken |Δ macro-F1| ≤ 0,01.
- **Öykünme uyarıları** (koşudan önce yazıldı):
  - SNR kesimi yalnız menzil-CFAR'ı yaklaşık taklit eder.
  - Clutter removal geri alınamaz.
  - Daha küçük bir anten dizisi taklit edilmez (açı kuantizasyonu iyimser olurdu).

Çıktı: `reports/faz_b/02_sensing_budget.md`.

## B.3 — Kare hızı (yalnız B.2 erken biterse)
- Her oturumda her k'inci kare tutulur, k ∈ {2, 3, 5}; yani 5 / 3,3 / 2 Hz.
- Pencere uzunluğu L_k = ceil(5/k) kare. L = 1'de eğim özellikleri 0 alınır.
- Debounce süre olarak sabit tutulur: MIN_BOX = 0,2 s, MAX_GAP = 0,1 s, kareye yuvarlanarak (en az 1 ve 0).
- B2 S1 macro-F1 ve anomali D3 AUROC / FAR raporlanır. Ayrıca A3.2 yöntemiyle dokunuş süresine göre tespit (D-ham) verilir.
- Diz kuralı B.2 ile aynı.

Çıktı: `reports/faz_b/03_frame_rate.md`.

## Sapmalar (tarihli)
- 2026-11-05 (B.1 uygulama ayrintilari, tek satir): düğüm sayısı = 5 kat modelinin ortalaması (tam-veri modeli ek sütun, kurala girmez); karşılaştırma = test pencerelerinde ağaç başına kök→yaprak iç düğüm sayısının toplamı, ortalaması; anomali sayıları = D3 'tüm anomali' kat-ortalama AUROC ve havuzlanmış FAR (α=0,10); OOF'lar yalnız reports/faz_b/oof/ altına yazıldı (faz_a/oof'a değil); n_jobs=3; seçim kuralı sabitleri 0,856/0,924 aynen, bayt sıralaması düğüm sayısıyla aynı (8 ve 12 B/düğüm), eşitlikte az karşılaştırma; çip RAM'i TI SWRS219F (Nisan 2025) veri sayfasından.
- 2026-11-05 (B.0 uygulama ayrıntıları, tek satır): (e) AUROC = kat bazında ortalama±sd (skor 1−p), FAR havuzlanmış 51 normal test döngüsü, tespit Wilson'ı benzersiz anomali döngüsü sayısıyla (k = oran×n); D0 ve D1 tek başına ve D3 yan yana, α=0,05 satırı da verildi. (f) 'boş kare' kare tablosunda frame_id boşluğu olarak yalnız 3 pencerede görünüyor; kayıttaki frame_id ardışık sayaç olduğundan ek olarak zaman boşluğu (pencere içinde ardışık kareler arası dt > 0,15 s; ayrıca > 0,25 s) tanımlandı ve S1 metrikleri her iki tanımla dahil/hariç (ve yalnız-boşluklu) verildi; 'tüm' satırı Faz A 03 ile birebir doğrulandı. (g) sıra = s02 normal döngülerinin start_idx sırası; bölmeler tam brief'teki gibi (izleme = kalan, A:21 B:16); kalibrasyon n_cal=15 olduğundan D3 α=0,10'da yapısal olarak alarm veremez — brief harfiyen uygulandı, ek olarak D0/D1/D2 tek tek, izleme normal-vs-anomali AUROC ve süre ortalamaları (betimsel) eklendi; ToF×tof havuzundaki doğal-sapma döngülerinin alarm durumuna bakılmadı, ayrı hesap yapılmadı.

- 2026-11-05 (B.2 uygulama ayrintilari, tek satir): kare kumesi tum seviyelerde sabit 16.272 kare, nokta kalmayan kare n_points=0/snr_sum=0, digerleri B2 Faz A dolgusu; eksenler TEK TEK degisir (digerleri taban; 9 seviye + taban, 50 RF); 'yalniz isaret' seviyesinde mean_abs_v yuvasina mean(sign v) konuldu (merged kumede mean(|sign v|)=1 sabit olur ve seviye 'cikarilmis'la ozdeslesirdi), 'cikarilmis'ta mean_abs_v=0 sabit; SNR eslesigi snr >= 10*tau (kayit 0,1 dB biriminde, min=150); tepe gruplama hucresi = (kare, menzil kutusu, aci kutusu), kutular sirali benzersiz degerlerde bosluk esigiyle (menzil 1e-4 m -> 6 kutu, aci 1e-3 derece -> 57 kutu), esitlikte dosyada ilk nokta; 'en yalin' = eksen icinde en az V2 bayt/kare; diz kurali dort olcutle ayri ayri (ΔF1>=-0,02; ΔF1 GA alt>=-0,02; ΔAUROC>=-0,03; ΔAUROC GA alt>=-0,03), GA = 5 kat esli fark t-GA (df=4), 'literal' = dordu birden, 'yalniz nokta' ayrica verildi; bayt/kare V2 = 16 + n*(12 + hiz alani + 4) (hiz alani full 4 B, isaret 1 B, cikarilmis 0 B), V3 = n*(8 + hiz + 2), bos karede 0 B, bayt/s = 10 Hz x ortalama bayt/kare; Faz A B2 kat bazli AUROC'u ayni hat fc.load_oof('b2') ile uretildi (ort. 0,923697); EK koşular (brief disi, ikincil, kurala girmez): eksen-basina secilenlerin birlesimleri (literal ve nokta-tahmin), taban icin RF tohumu 1-8 gurultusu, tum seviyeler icin tohum 1-4 tekrari.
- 2026-11-05 (B.3 uygulama ayrintilari, tek satir): seyreltme = oturum ici kare konumu % k == faz (ana faz 0; faz 1..k-1 ikincil); pencere etiketi = son tutulan karenin 10 Hz y_true'su (yeniden yumusatma yok), L=1'de egim 0; kat = pencerenin son karesinin gid'inin Faz A S1 kati (gid tek katta, dogrulandi); debounce kare sayisi = max(1, round(0,2 s/(0,1k))) ve max(0, round(0,1 s/(0,1k))) Python round (k=2'de MAX_GAP 0,5 -> 0 belirsiz; yari-yukari/MAX_GAP=1 ve kare-sayisi-sabit 2/1 ikincil), debounce tutulan akista, sonra 10 Hz'e sifirinci-derece tutma, ac.build_cycles('pred') degistirilmeden (ic (2,1) debounce'un etkisiz oldugu assert ile dogrulandi); anomali hatti Faz A protokolu, 'tum anomali' D3 AUROC kat ortalamasi, FAR havuzlanmis; diz kuralinda 'en yalin' = en buyuk k, dort olcut B.2 gibi, referans = Faz A B2 kat degerleri (ek: ayni tutulan karelerle kisitli referans); D-ham = ToF dokunusunun kare kumesi tam hizda A3.2 atamasiyla (t-100 ms, t] belirlenip tutulan karelerle kesistirildi, payda = tam hizda degerlendirilebilen 1111 dokunus, tutulan karesi olmayan = kacirma (kosullu varyant ikincil), lojistik d50/d90 A3.2 fit_logit (importlib ile, degistirilmeden); kapilar: k=1 kat F1/AUROC/FAR/tespit ve D-ham sayilari/lojistik Faz A ve A3.2 ile birebir; B.3'te onceden yazilmis sayisal beklenti yok; EK kosular (brief disi, kurala girmez): faz kaydirmasi, RF tohumu 1-4 (k=1 dahil), debounce varyantlari; s02 dogal-sapma dongulerinin alarm durumuna bakilmadi.

- 2026-11-05 (B.3 duzeltme turu, tek satir): dogrulayici bulgusu uzerine hata duzeltildi: 03_frame_rate.py satir 266 oturum adlarini yanlis yaziyordu (session_03_seq135 / session_04_seq145; gercek: session_03_normal_seq135 / session_04_normal_seq145), bu yuzden "s02-s04 >= 0,5 s" satirlari yalniz s02 idi (n=118); filtre SHORT[s] in (s02,s03,s04) olarak degistirildi ve n=347 assert eklendi; 03_frame_rate.py + 03b yeniden kosuldu. Degisen tek sey vs_n/vs_hit/vs_rate/vs_lo/vs_hi (33 satir); macro-F1, AUROC, FAR, tespit, dokunus tespiti CSV ve fits CSV ile 165 OOF dosyasi bayt/deger olarak ayni (diz kurali sonucu "tam 10 Hz gerekli" degismedi). Rapora oturum bazinda (s02/s03/s04 ayri, faz araliklariyla) >= 0,5 s tablosu ve A3.2 sinamasi satiri eklendi (brief disi ek; havuz tek basina A3.2 sinamasi degildir). Duzeltilmis havuz (faz 0): k=1 347/347, k=2 347/347, k=3 338/347=0,974, k=5 327/347=0,942 [0,913; 0,962]; s04 k=5 98/109=0,899 (<0,9; fazlarda 0,835-0,954), yani A3.2 "D-ham >= 0,9" tahmini k=5, s04 icin tutmuyor. Eski sayilar ve rapor: <LOCAL_BACKUP>/bk_b3/.

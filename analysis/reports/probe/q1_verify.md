# Q1 dogrulama (adversarial verifier, tur 1)

## Ozet
- Yeniden uretildi mi: **EVET.** `tools/remote_run.sh probe/01_dwell_modes.py` sifirdan calistirildi (exit=0); uretilen `q1.md` ve `q1_table.csv` calistirma oncesi kopyaladigim orijinallerle bayt bayt AYNI (diff bos). Stdout ile `q1.md` bos satirlar/bosluklar disinda ozdes; yani md'deki her sayi stdout'ta da var. Uyusmazlik: **yok**.
- Verdikt (uc proxynin hicbiri H1'i gecmiyor) **korunuyor**.
- Blocking bug: **yok**.

## Kod incelemesi
Kontrol edilenler ve sonuc:
- Split/leakage: rastgele bolme yok. MeanShift oturumdaki tum karelere etiketsiz fit ediliyor; etiket yalniz degerlendirmede (NMI/ARI, label>0) ve s01 quantile secim kuralinda kullaniliyor (beyan edilen kural, `select_quantile`, satir 368-377). s02-s06'da quantile AYNEN uygulaniyor (satir 443-453). Test-set tuning yok.
- Kriterler (NMI_PASS, EXPECTED_K, MIN_SCEN_PASS) dosya basinda sabit (satir 29-33); `passes()` ile tutarli.
- NMI/ARI: `normalized_mutual_info_score(y, a)` ve `adjusted_rand_score` dogru argumanlarla; yalniz `ok = box & assign>=0` karelerinde. Statik proxy icin atanamayan kareler dusuyor ama `coverage` raporlaniyor (sessiz drop degil).
- Spacing/dx_res: dx_res = 0.45*rad(15) = 0.118; s01 aralik tablosu (8.0, 12.8, 9.8, 8.5 cm) ve "3/4 cift < dx_res" dogru. Senaryo oturumlari aralik iddiasi ("14-30 cm, yalniz s04 Box4-5 8 cm") tabloyla uyumlu (14.2-30.5 cm; s04 4-5 = 8.2 cm).
- Loop/off-by-one: bu scriptte loop kullanilmiyor; Q1 icin ilgisiz.

## Knob testi
1. Quantile (secilmeyen degerler) — `q1_table.csv` uzerinden, yeniden kosu gerekmeden: s01'de **hicbir** proxy/variant/quantile kombinasyonu NMI >= 0.70 ve K in {5,6} birlikte vermiyor (en yuksek NMI 0.691, all_pts/xy q=0.2; q=0.1 -> K=8-11, q=0.3 -> K=2-4). Senaryo gecis sayilari q'ya gore 1/5..5/5 arasi oynuyor ama s01 kosulu hep basarisiz oldugu icin verdikt q'dan bagimsiz FAIL.
2. Proxy (i) agirliklandirma: SNR-agirlikli yeniden ornekleme kapatildi (tum noktalar agirliksiz; `_verify_q1.py`, sed ile tek satir). Sonuc: (i) 2B s01 K=4, NMI=0.677 -> FAIL; senaryo 4/5 (s04 K=2 NMI=0.70 x); (i) 1B s01 NMI=0.619, 3/5; (ii) 2B 4/5, 1B 4/5. Hepsi s01'de FAIL -> **verdikt degismedi**. `_verify_q1.py` silindi, orijinal script yeniden kosuldu; `q1.md`/`q1_table.csv` orijinallerle ayni (dogrulandi).

## Bulgular
Blocking: yok.

Minor:
1. **Kriter fiilen ulasilamaz (s01).** Denetimli "tavan" (etiketli en-yakin-merkez) s01'de NMI=0.693 < 0.70 (q1.md satir 121). Yani NMI >= 0.70 kapisi s01'de etiketli yontem icin bile gecilemiyor; "FAIL" kismen kapinin tasarim sonucu. Rapor bunu "cozunurluk bulgusu" olarak zaten soyluyor (dogru yorum), fakat verdikt tablosu bunu FAIL ile ayni agirlikta gosteriyor. (i)/2B s01 NMI=0.691 ile tavanin %100'u; yani esige 0.009 kalan bu proxy "neredeyse etiketli kadar iyi" diye okunmali. Oneri: verdikt satirina "s01 kapisi etiketli tavanda bile gecilmiyor" notu.
2. **"Denetimli tavan" ifadesi asiri iddia.** `nn_classify` s01 medyan merkezleriyle resubstitution; optimal siniflandirici degil (ornegin cx,cy yerine tum kare ozellikleri veya cok-nokta siniflandirici NMI'yi artirabilir). "Tavan" yerine "etiketli referans (nearest-centroid)" demek daha dogru (satir 210, 476, 485).
3. **s3 cumlesi post-hoc secime dayaniyor.** Yorumun "(i)-2B uzerinde 3 istasyonlu senaryolarda cok iyi (5/5, NMI 0.86-1.00)" iddiasi (i)-2B'nin s01'de FAIL ettigi, ama senaryolarda en iyi gorundugu icin secildi. Ayrica kirilgan: agirliksiz (i)-2B 4/5'e dusuyor (s04 K=2, NMI 0.70), s04 aralik 8 cm oldugu icin zaten sinirda. Q3'te (i)-2B kullanilacaksa bu kirilganlik ve secimin post-hoc oldugu belirtilmeli.
4. **Sabit kodlanmis sayilar yorum metninde** (satir 608: "14-30 cm", "NMI 0.86-1.00", "5/5"; satir 526: "%53-60"; satir 614 "(i)-2B 5/5") sonuclardan hesaplanmiyor; bugun dogru ama kosu degisirse sessizce yanlis kalir. Boyle kalirsa risk dusuk (seedler sabit, kosu deterministik).
5. `box_centroids` kenar durumu: tek kare iceren kutu icin `std` NaN olur (sep_ratio NaN); bu verilerde olusmuyor.
6. Birlesme (merge-zone) modu yalniz s04'te (1 mod, none paylasimi %21.6) goruluyor; diger tum oturum/proxylerde yok (`none` kareleri kutu modlarina dagiliyor). Rapor bunu tablo olarak veriyor; "hava modu yok" sonucu yorumda acikca soylenmemis (kucuk eksik).

## Verdiktin degerlendirmesi
Verdikt (hicbiri H1 PASS degil; s01 basarisizligi mod-bulma degil cozunurluk bulgusu; 3 istasyonlu senaryolarda etiketsiz kesif calisiyor) sayilarla destekleniyor, bagimsiz yeniden kosuda birebir uretiliyor ve quantile/agirliklandirma degisikliklerine dayanikli. Tek dikkat: s01 esigi etiketli referansta bile 0.693 oldugundan "FAIL" bir mod-bulma zayifligindan cok kapinin ulasilamazligini yansitiyor; senaryo basarisi ise s04'te kirilgan.

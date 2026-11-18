# C.2 — Kayıttaki zaman boşluklarının nedeni (betimsel)

`faz_c/02_time_gaps.py` üretir; plan `faz_c/DESIGN_C.md` C.2. Statü: KEŞİF AMAÇLI; kullanıcı görmeden `main.tex`'e girmez. Her sayı stdout'a ve `02_time_gaps.csv`'ye de yazıldı (ek ayrıntı CSV'leri: `02_time_gaps_*.csv`). Yalnız okuma: veri klasörü, `reports/*`, `paper/*.tex`. Doğal sapan s02 döngülerinin alarm durumuna bakılmadı.

## 0. Özet

- **Karar (önceden yazılmış kurala harfiyen): H-bos DESTEKLENDI.**
- Boşluk sayısı 696 (dt > 0,15 s) — Faz B B.0(f) ile aynı; ham `mmwave_data.csv` ile kare tablosu aynı 696 aralığı verir. Tahmini kayıp kare yuvası 2801 (kayıp payı 14.7%: 2801/19073). `frame_id` sayacı ardışık olduğundan bu sayı yalnız zaman damgasından kestirilir.
- **Test (i) tam kat: GEÇTİ** — boşluk dt'lerinin 99.3%'i 0,1 s'nin tam katına ±5.0 ms içinde (eşik ≥ %90; rastgele fazda ~10.0%). Bu iki hipotezle de uyumludur; ayrım (ii)'dedir.
- Boşluktan sonra 100 ms kare ızgarasının aynı fazda sürmesi, kaybın tam kare birimleriyle olduğunu gösterir; kaybın radar çıktısında mı (boş kare gönderilmedi) yoksa kaydedici/ayrıştırıcıda mı olduğunu ayırt etmez.
- Test (ii) n_points: komşu karelerin oturum-içi yüzdelik medyanı 0.060 (0,5 = fark yok); MWU tek yönlü p(düşük) = 1e-300, p(yüksek) = 1 (1e-300 = sayısal taban; gerçek p daha küçük) → **dusuk**. 'Kutu yok' payı komşularda 97.5% (tüm kareler 56.9%) → **yuksek**. Boşluk aralığındaki ham ToF aktif payı 0.9% (rastgele pencere 36.1%, permütasyon p = 0.001) → **dusuk(aktif az)**.
- Test (iv) kayıt kodu: kullanıcının diskinde (<LOCAL_HOME>, D:\) kayıt/UART ayrıştırma kodu **bulunamadı**; `experiment_info.txt` kayıt makinesinin Linux'ta `<REMOTE_HOME>/Github/mmwave/mmwave_sensor` altında olduğunu söylüyor, bu kod diskte yok. Bu nedenle (iii) 'kesin kanıt' yolu kapalı.
- Test (v): zaman damgası süresi, kare × 0,1 s'den toplamda 1907 s'ye karşı 1627 s (oran 1.172). Kare sayısından hesaplanmış dakikalar var (§6, düzeltme ÖNERİSİ).

## 1. Test (i) — tam kat testi

Düzenli aralıklar (dt ≤ 0,15 s; n = 15570): |dt − 0,1| medyanı 0.60 ms, 95. yüzdelik 2.51 ms → tolerans = 2 × p95 = **5.01 ms** (0.050 birim). Boşluklar: n = 696.

- Tolerans içinde: **99.3%** (eşik ≥ %90 → GEÇTİ).
- Rastgele zamanlama karşılaştırması: aynı dt'lere U(0, 0,1 s) rastgele faz eklenince tolerans içinde pay ortalama 10.0% (2000 tekrarın en büyüğü 14.1%); analitik beklenti 2 × tol_birim = 10.0%.
- |dt/0,1 − round| medyanı 0.006 birim, p95 0.029 birim; ortalama işaretli sapma -0.05 ms.
- Yedek (bilgi): düzenli medyan periyot 0.1000 s ile aynı test: 99.3%.

Kayıp kare sayısına (k = round(dt/0,1) − 1) göre:

| sinif | n | tol_ici | medyan_resid_ms | p95_abs_ms |
|---|---|---|---|---|
| k=1 | 263 | 0.9924 | -0.001 | 3.117 |
| k=2 | 140 | 0.9929 | 0.044 | 2.984 |
| k=3-4 | 114 | 1 | -0.035 | 2.894 |
| k=5-9 | 116 | 0.9828 | 0.126 | 2.814 |
| k>=10 | 63 | 1 | 0.061 | 2.172 |

`dt/0,1 − round(dt/0,1)` dağılımının tamamı (0,1 birimlik kutular; birim = 0,1 s):

| kutu_alt | kutu_ust | n | pay |
|---|---|---|---|
| -0.5000 | -0.4000 | 0.0000 | 0.0000 |
| -0.4000 | -0.3000 | 1.0000 | 0.0014 |
| -0.3000 | -0.2000 | 0.0000 | 0.0000 |
| -0.2000 | -0.1000 | 2.0000 | 0.0029 |
| -0.1000 | -0.0000 | 319.0000 | 0.4583 |
| -0.0000 | 0.1000 | 374.0000 | 0.5374 |
| 0.1000 | 0.2000 | 0.0000 | 0.0000 |
| 0.2000 | 0.3000 | 0.0000 | 0.0000 |
| 0.3000 | 0.4000 | 0.0000 | 0.0000 |
| 0.4000 | 0.5000 | 0.0000 | 0.0000 |

Okuma: Boşluk süreleri radar kare periyodunun tam katlarıdır; kayıp olan şey bütün radar kareleridir (kısmi/yarım kare değil) ve boşluktan sonra zaman ızgarası aynı fazda sürer. Bu, kare sınırından bağımsız (rastgele fazda) bir takılma ile tutarlı DEĞİLDİR; kare sınırlarında kayıpla tutarlıdır, ve "boş kare kaydedilmedi" ile "kare kaydedilemedi" (UART taşması vb.) ikisi de bunu öngörür, yani ayırt edemez. Aykırılar: tolerans dışında kalan 5 boşluk (bunların 3'ü ±0,1 birimin ötesinde, kalanı ±0,1 içinde ama toleransın dışında). Not: zaman damgaları ana bilgisayar tarafında basıldığı için ±5 ms jitter vardır; sapma uzun boşluklarda (k ≥ 10) da tolerans içinde kalır (k bölmesi tablosu), yani radar saati ile ana bilgisayar saati arasında ölçülebilir kayma yok.

## 2. Test (ii) — ayrım testi: boşluğun komşu kareleri

Komşu = boşluktan hemen önceki ve sonraki kayıtlı kare (benzersiz kareler: önceki 696, sonraki 696, havuz 1217). Kıyas tabanı: aynı oturumun tüm kareleri; MWU, oturum-içi yüzdelik-sıra (tüm kareler içindeki oturum içi yüzdelik) üzerinde (oturum farklarını yok eder). Komşu kareler tabana dahil olduğu için (≈%7.5) testler hafif tutucudur; komşuluk örtüşmesi p-değerlerini kabaca yorumlanır kılar.

Önceden yazılmış okuma kuralı (çalıştırmadan önce sabit): n_points 'düşük' = yüzdelik medyan < 0,5 ve MWU tek yönlü p < 0,01 ('yüksek' simetrik); 'kutu yok' Fisher tek yönlü p < 0,01; boşluk aralığındaki ham ToF aktif payı rastgele pencerelere göre permütasyon p < 0,01 (iki yönlü). İki 'kutu yok' kaynağı aynı yönü göstermezse belirsiz. (Bu kural brifin 'düşük/yüksek'ini sayısallaştırır; brifteki hipotez tahminlerini değiştirmez.)

| komsu | degisken | n_komsu | komsu_medyan | tum_kare_medyan | komsu_pctl_medyan | p_iki_yonlu | p_dusuk | p_yuksek |
|---|---|---|---|---|---|---|---|---|
| onceki | n_points (merged, v!=0) | 696 | 2 | 7 | 0.06 | 1.309e-258 | 6.545e-259 | 1 |
| onceki | n_all (ham, tum noktalar) | 696 | 2 | 9 | 0.05672 | 8.296e-272 | 4.148e-272 | 1 |
| onceki | kare ort. menzil (m) | 696 | 0.496 | 0.477 | 0.9237 | 2.813e-148 | 1 | 1.406e-148 |
| onceki | ort. |hiz| (m/s) | 696 | 0.1207 | 0.1811 | 0.07894 | 4.343e-68 | 2.172e-68 | 1 |
| sonraki | n_points (merged, v!=0) | 696 | 2 | 7 | 0.06 | 1.376e-253 | 6.879e-254 | 1 |
| sonraki | n_all (ham, tum noktalar) | 696 | 2 | 9 | 0.05672 | 8.906e-266 | 4.453e-266 | 1 |
| sonraki | kare ort. menzil (m) | 696 | 0.496 | 0.477 | 0.9277 | 2.247e-185 | 1 | 1.124e-185 |
| sonraki | ort. |hiz| (m/s) | 696 | 0.1207 | 0.1811 | 0.07894 | 6.152e-76 | 3.076e-76 | 1 |
| havuz | n_points (merged, v!=0) | 1217 | 2 | 7 | 0.06 | 1e-300 | 1e-300 | 1 |
| havuz | n_all (ham, tum noktalar) | 1217 | 2 | 9 | 0.05672 | 1e-300 | 1e-300 | 1 |
| havuz | kare ort. menzil (m) | 1217 | 0.4884 | 0.477 | 0.9175 | 1.851e-265 | 1 | 9.253e-266 |
| havuz | ort. |hiz| (m/s) | 1217 | 0.1207 | 0.1811 | 0.08287 | 8.825e-91 | 4.412e-91 | 1 |

- 'Kutu yok' payı: tüm kareler 56.9%; komşu havuz 97.5% (Fisher p(yüksek) = 6.66e-230, p(düşük) = 1); önceki 97.8%; sonraki 97.8%.
- Boşluk öncesi son 5 karenin nokta-sayısı eğimi (kare başına): medyan -1.500 (tüm pencereler +0.000); negatif eğim payı 92.9% (tümü 48.0%); MWU p(düşük) = 8.89e-219. Ortalama eğim -1.449 vs -0.001.
- Nokta sayısı dağılımı: n=1 payı tüm kareler 3.2%, komşu havuz 28.8%; n=2: 4.0% vs 27.6%.

Oturum bazında (komşu havuz / tüm kareler):

| oturum | n_komsu | npts_medyan_komsu | npts_medyan_tum | kutuyok_komsu | kutuyok_tum | menzil_komsu | menzil_tum |
|---|---|---|---|---|---|---|---|
| session_01_training | 245 | 2.000 | 7.000 | 0.931 | 0.596 | 0.483 | 0.477 |
| session_02_normal_seq125 | 183 | 2.000 | 7.000 | 0.989 | 0.555 | 0.483 | 0.477 |
| session_03_normal_seq135 | 198 | 2.000 | 7.000 | 1.000 | 0.561 | 0.496 | 0.477 |
| session_04_normal_seq145 | 147 | 2.000 | 7.000 | 1.000 | 0.529 | 0.496 | 0.477 |
| session_05_anomaly_fast | 146 | 3.000 | 7.000 | 0.993 | 0.580 | 0.480 | 0.477 |
| session_06_anomaly_slow | 298 | 2.000 | 6.000 | 0.966 | 0.563 | 0.496 | 0.477 |

## 3. Test (iii-a) — oturum bazında boşluk oranı ve boşluk anlarında ham ToF

| session | kare | aralik | bosluk | bosluk_orani_kare_basina | kayip_yuva | kayip_pay | en_uzun_bosluk_s | medyan_bosluk_s | bosluk_ge_1s | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|
| session_01 | 4683 | 4682 | 139 | 0.02968 | 395 | 0.07779 | 1.102 | 0.3001 | 5 | 507.8 |
| session_02 | 2274 | 2273 | 110 | 0.04837 | 480 | 0.1743 | 8.4 | 0.3004 | 8 | 275.4 |
| session_03 | 2407 | 2406 | 120 | 0.04985 | 313 | 0.1151 | 1.6 | 0.3 | 2 | 272 |
| session_04 | 1928 | 1927 | 85 | 0.04409 | 356 | 0.1559 | 1.6 | 0.3999 | 8 | 228.4 |
| session_05 | 1780 | 1779 | 85 | 0.04775 | 219 | 0.1096 | 2 | 0.299 | 2 | 199.9 |
| session_06 | 3200 | 3199 | 157 | 0.04906 | 1038 | 0.2449 | 3.601 | 0.3007 | 49 | 423.8 |

Ham ToF (~30 Hz) ve video radar karelerinden bağımsız örneklenir ve aynı ana bilgisayar saatini kullanır. Boşluk aralığı = (önceki kaydın zaman damgası, sonraki kaydın zaman damgası); duyarlılık için aralık 0,1 s ve 0,2 s geriye kaydırıldı (radar zaman damgası ToF sayaç değişiminden medyan 86 ms sonra basılır; fiziksel yakalama damgadan önce olduğundan geri kaydırma haklıdır). ToF/video satırlarındaki `frame_id` 'şimdiye dek yazılan radar karesi sayısıdır' (ToF satırı, yazılmış son radar karesinin damgasından ~14 ms sonra bir sonraki kimliğe geçer).

- 696 boşlukta aralığa düşen ToF satırı var (11003 satır). Aktif ('kutu' ToF < 140 mm) pay: havuz **0.9%**, boşluk ortalaması 2.2%; aynı satır sayısında rastgele ToF pencereleri: ort. 36.1% [q2,5–q97,5 = 33.3%, 38.7%] (R = 1000); permütasyon p (iki yönlü) = 0.001 (boşluk-ortalaması için p = 0.001). Tüm ToF'de aktif pay 36.6%.
- Boşlukların 97.7%'i tümüyle 'kutu yok' iken 2.2%'i baştan sona 'kutu aktif'.
- ToF etiket değişim oranı (satır başına): boşluklarda 0.0002, rastgele pencerelerde 0.0344 (p(düşük) = 0, p(yüksek) = 1) — el boşlukta durağan mı?
- Kayıp yuva başına tehlike (S1): P(kayıp | ToF 'kutu yok') = 0.2312; P(kayıp | kutu aktif) = 0.0023; oran 101.58. Kayıp yuvaların 99.4%'i 'kutu yok' (kayıtlı karelerde 56.9%).
- Kaydedici canlılığı (S3): boşluk ≥ 1 s olan 74 boşlukta ToF oranı min/medyan 30.8/31.4 satır/s, video 23.6/25.0 kare/s; tüm boşluklarda ToF iç-aralık max dt 0.040 s, video 0.054 s. Boşluk aralığındaki ToF satırlarının 100.0%'inde `frame_id` = önceki kayıtlı kare + 1 (yani sonraki kaydın kimliği; önceki kimliğe eşit olanlar 0.0%): sayaç boşluk boyunca ilerlemiyor, yalnız YAZILAN radar karesiyle artıyor. Bu yüzden `frame_id` ardışıklığı iki hipotezi de aynı şekilde öngörür.

- Not (şeffaflık): permütasyon p-değerleri R = 1000 ile sınırlıdır (en küçük ölçülebilir p ≈ 1/R = 0.001); gözlenen aktif pay, 1000 rastgele pencere kümesinin hiçbirinde bu kadar düşük çıkmadı. İlk koşuda ToF aralığı yanlışlıkla `frame_id`'nin 'önceki kare' kimliği taşıdığı varsayımıyla (sayaç gecikmesi düzeltmeli) kurulmuştu; ToF satırlarının aslında 'şimdiye dek yazılan kare sayısı' (= sonraki kare kimliği) taşıdığı görülünce aralık (ts_önceki, ts_sonraki) olarak yeniden tanımlandı. Yön değişmedi (aktif pay %0,58 → %0,87), karar kuralı aynı.

En uzun 10 boşluk:

| session | t_pre | dt | k | act_frac | n_pre | n_post | lab_pre | lab_post | range_pre | range_post |
|---|---|---|---|---|---|---|---|---|---|---|
| session_02 | 266 | 8.4 | 83 | 0 | 2 | 1 | 0 | 0 | 0.477 | 0.496 |
| session_02 | 260 | 6.8 | 67 | 0 | 2 | 2 | 0 | 0 | 0.458 | 0.477 |
| session_06 | 173 | 3.6 | 35 | 0 | 2 | 4 | 0 | 0 | 0.496 | 0.486 |
| session_06 | 350 | 3 | 29 | 0 | 1 | 6 | 0 | 0 | 0.496 | 0.477 |
| session_06 | 147 | 2.9 | 28 | 0 | 4 | 6 | 0 | 0 | 0.458 | 0.483 |
| session_06 | 107 | 2.6 | 25 | 0 | 4 | 1 | 0 | 0 | 0.486 | 0.496 |
| session_06 | 182 | 2.5 | 24 | 0 | 1 | 1 | 0 | 0 | 0.496 | 0.496 |
| session_06 | 324 | 2.4 | 23 | 0 | 1 | 2 | 0 | 0 | 0.496 | 0.496 |
| session_06 | 28.7 | 2.3 | 22 | 0 | 8 | 4 | 0 | 0 | 0.486 | 0.486 |
| session_06 | 123 | 2.2 | 21 | 0 | 2 | 5 | 0 | 0 | 0.496 | 0.481 |

## 4. Ek ölçümler (sezgisel; karar kuralına girmez)

- **S2 (sıfır-nokta ekstrapolasyonu).** Kayıtlı karelerde nokta sayısı payı p1 = 0.0319, p2 = 0.0404. Boş karelerin kayıtlı karelere oranı (gözlenen) M/N = 0.1721. n_points dağılımının sıfıra Poisson (p1²/(2·p2)) ekstrapolasyonu 0.0126, geometrik (p1²/p2) 0.0252. Bu karşılaştırma model varsayımına bağlıdır; H-boş'u KANITLAMAZ, yalnız büyüklük mertebesi kontrolüdür.
- **S4 (statik noktalar).** Ham noktaların 20.1%'i v = 0 (merged_df bunları atar). 12995 karede ≥ 1 statik nokta var ama yalnız-statik kare sayısı 0. Tek-nokta karelerinde yalnız-statik payı 0.0% (bağımsızlık altında beklenen ≈ 20.1%) — kayıtlı kareler daima en az bir hareketli nokta içeriyor.

## 5. Test (iv) — kayıt kodu araması (salt okunur, çalıştırılmadı)

Arama: `rg -l` ile `<LOCAL_HOME>` ve `D:\` altında (AppData, .git, node_modules, site-packages, venv hariç; ≤ 5 MB; *.py/ipynb/c/cpp/h/js/ts/m/sh/txt/md/cs/java), anahtarlar `numDetectedObj`, `0x0102`, `0x0201`, `mmwave_data.csv`, `magic word`; ikinci tur: `serial.Serial`, `experiment_data`, `distance_sensors`, `video_frames.csv`, `mmwave_sensor`, `IWR6843`; ayrıca dosya/klasör adı taraması (`*mmwave*`, `*uart*`, `*radar*`, `*recorder*`) ve `mmwave_anomaly.zip` içerik listesi. Eşleşen dosya listesi: `faz_c/02_time_gaps_search.txt`.

- **Kayıt/UART ayrıştırma kodu bulunamadı.** `mmwave_data.csv`'yi yalnız *okuyan* kod var: `00_dataprepare_all.ipynb` (üç kopya: `Documents\GitHub\mmwave_anomaly`, `Github\jupyter_notebook\data\working`, dondurulmuş 4 numaralı klasör `dataset\notebooks`) ve çıkarım not defterleri. Bunlar kayıt sonrası hazırlık: hücre 0, satır 40–42 `velocity != 0` ile v = 0 noktalarını atar; satır 46–58 `mmwave`/ToF `frame_id` kesişimini iç birleştirir (`how='inner'`) ve ToF'de `frame_id` başına ilk satırı tutar. **Bu kod boş kareyi atlayan bir koşul içermez ve içeremez** (kayıt sonrası; boş kareler zaten dosyada satır üretmez). Kayıt kodu (`mmwave_sensor`, FastAPI + pyserial + opencv; `experiment_info.txt`'e göre `<REMOTE_HOME>/Github/mmwave/mmwave_sensor/venv` altında) bu diskte yok; muhtemelen kayıt makinesinde (Linux).
- İlgili diğer bulgu: `mmwave_data.csv` uzun biçimli (nokta başına bir satır). Sıfır noktalı bir kare, kaydedici onu sayaçla numaralasa bile **dosyada hiç satır üretmez**. `frame_id` ardışık olduğuna göre kaydedici ya boş kareleri sayaca dahil etmiyor ya da hiç görmedi — dosyadan ayırt edilemez.
- Sonuç: (iii) 'kesin kanıt' yolu bu oturumda kapalı. **Önerilen eylem:** kayıt makinesindeki `mmwave_sensor` kodunda UART okuma/ayrıştırma döngüsünde `numDetectedObj == 0` ya da magic-word hatası durumunda `continue`/`return` olup olmadığına ve `frame_id` artışının nerede yapıldığına bakılmalı (kullanıcı veya erişimi olan oturum).

## 6. Test (v) — süre hesabı denetimi

Oturum başına duvar saati (ilk → son radar zaman damgası + 0,1 s) ve kare × 0,1 s:

| session | kare_ham | kare_merged | wall_s | kare_x_0p1_merged_s | oran_wall_per_kare_merged | fark_s | kayip_yuva | kare_plus_kayip_x_0p1_s |
|---|---|---|---|---|---|---|---|---|
| session_01 | 4683 | 4683 | 507.8 | 468.3 | 1.084 | 39.52 | 395 | 507.8 |
| session_02 | 2274 | 2274 | 275.4 | 227.4 | 1.211 | 48.01 | 480 | 275.4 |
| session_03 | 2407 | 2407 | 272 | 240.7 | 1.13 | 31.29 | 313 | 272 |
| session_04 | 1928 | 1928 | 228.4 | 192.8 | 1.185 | 35.6 | 356 | 228.4 |
| session_05 | 1782 | 1780 | 199.9 | 178 | 1.123 | 21.87 | 219 | 199.9 |
| session_06 | 3200 | 3200 | 423.8 | 320 | 1.324 | 103.8 | 1038 | 423.8 |

Toplam: duvar saati 1907.3 s (31.79 dk), kare × 0,1 s = 1627.2 s (27.12 dk); oran 1.1721. (kayıtlı + tahmini kayıp) × 0,1 s duvar saatiyle ±0,1 s'lik jitter ölçeğinde örtüşür (tablo, son sütun). Ham dosyada 16,274, merged'de 16,272 kare var; s05'te 2 kare (`frame_id` 2 ve 4) ToF'de karşılığı olmadığından iç birleştirmede düşmüş (merge artefaktı, boşluk nedeni değil).

**Kaynak denetimi.**

| Kaynak | Hesap | Sonuç |
|---|---|---|
| `reports/faz_a4/04_unseen_minutes.csv` (`04_unseen_sessions.py` L261–273) | `etiketli_dakika = egitim_penceresi / 600` | **Kare sayısından**; zaman damgası değil → düzeltme önerisi aşağıda |
| `reports/paper_prep/05_gaps_minutes_*.csv` | döngü süresi `t(sonraki döngünün ilk karesi) − t(bu döngünün ilk karesi)` | **Zaman damgasından** (`session_wall_min` bu betikteki wall_min ile aynı; tablo CSV'de) |
| `reports/paper_prep/05_gaps.md` §1 | 'merged_df … yalnız en az bir noktalı kareleri içerir' | Bu, H-boş hipotezinin **olgu gibi yazılmış hali**; bu rapor onu doğrulayamadı (bkz. §0, §5) |

`04_unseen_minutes.csv` için zaman damgası temelli ÖNERİ (oturum oranı wall/(kare × 0,1 s) ile ölçeklenmiş; dosyalara dokunulmadı):

| kaynak | egitim_penceresi | etiketli_dakika | etiketli_dakika_zaman_damgasi_onerisi |
|---|---|---|---|
| Faz A (S1 OOF) kat modeli | 12998.40 | 21.66 | 25.39 |
| (i) S2: s02'yi tahmin eden model | 13978.00 | 23.30 | 27.16 |
| (i) S2: s03'ü tahmin eden model | 13845.00 | 23.07 | 27.22 |
| (i) S2: 6 modelin ort. | 13540.00 | 22.57 | 26.45 |
| (ii) ii_f0 | 6464.00 | 10.77 | 12.06 |
| (ii) ii_f1 | 6528.00 | 10.88 | 12.19 |
| (ii) ii_f2 | 6506.00 | 10.84 | 12.14 |
| (ii) ii_f3 | 6499.00 | 10.83 | 12.13 |
| (ii) ii_f4 | 6474.00 | 10.79 | 12.08 |
| (ii) ii_all | 6948.00 | 11.58 | 13.04 |

`paper/main.tex` / `supplementary.tex` içindeki dakika/süre cümleleri (satır numarası, kaynak sınıfı; düzeltme ÖNERİSİ, dosyaya dokunulmadı):

- `main.tex:506`: \emph{Commissioning.} With 20 calibration cycles fixed, the D3 AUROC was 0.893, 0.920, 0.928 and 0.928 for 5, 10, 15 and 20 normal training cycles (figure~\ref{fig:commission}a); 30 normal cycles (10 for training, 20 for calibration) take 2.5 to 2.7\,min to record, although the detection rate at $\a…
- `main.tex:508`: B2 reached at least 95\% of its full-data macro-F1 with 10 labelled cycles per session (0.829), 5.9\,min over the six sessions (figure~\ref{fig:commission}b). % 0.829, 95%, 5.88 [S] faz_a2/04_commissioning.md; paper_prep/05_gaps.md (gap 1b)
- `main.tex:510`: For a station that records no deviation, estimator (ii) of section~\ref{sec:res_unseen} is the relevant set-up: ToF-labelled recordings of the training and the normal session, about 7.8 and 3.8\,min (11.6\,min, against 21.7\,min of labelled frames per pooled S1 model), the latter also supplying the …
- `main.tex:515`: \caption{Commissioning cost, in cycles (secondary analysis; estimators trained on cycles of all six sessions). (a) D3 AUROC (B2 states, predicted cycles) as the number of normal training cycles grows from 5 to 20; 20 further normal cycles calibrate the $p$-values. The three curves are all anomalies …
- `main.tex:558`: Commissioning as tested here needs ToF-labelled recordings of a training session and a normal session, about 11.6\,min in total, with the detector fitted and calibrated on about 40 normal cycles of the latter; installing and aligning the sensors and a new person at a new station were not part of thi…
- `supplementary.tex:85`: \emph{Commissioning:} the 51 normal cycles were reassigned 50 times to 11 test, 20 calibration and 20 pool cycles, and the detector was fitted on the opening $n\in\{5,10,15,20\}$ pool cycles; B2 and B1 were retrained in each S1 fold on $k\in\{1,2,5,10,20\}$ labelled cycles per session (10 repetition…
- `supplementary.tex:306`: \emph{Recording time.} Measured as windows divided by 600 (0.1\,s frames), a pooled S1 fold model was trained on about 21.7\,min of labelled frames and a leave-one-session-out model on about 22.6\,min; estimator (ii) used s01 (about 7.8\,min) and s02 (about 3.8\,min), 10.8\,min for the fold models a…
- `supplementary.tex:307`: The anomaly detector additionally fits on about 20 and calibrates on about 20 normal cycles of s02 (about 1.7\,min each), which lie within the s02 recording. % 20.0, 1.67 [S] faz_a4/04_unseen_sessions.md (section 6)
- `supplementary.tex:363`: \emph{Commissioning.} In recording time, about 30 normal cycles (10 for the parameters and 20 for calibration) take about 2.7\,min at the mean cycle duration of 5.4\,s, or 2.5\,min at the median of 5.0\,s. % 30, 2.7, 5.4, 2.5, 5.0 [S] paper_prep/05_gaps.md (gap 1a)
- `supplementary.tex:365`: For state estimation, B2 reached a macro-F1 of 0.713, 0.770, 0.810, 0.829 and 0.842 with 1, 2, 5, 10 and 20 labelled cycles per session, against 0.856 with all training cycles; ten cycles take 0.7 to 1.5\,min per session to record. % 0.713, 0.770, 0.810, 0.829, 0.842, 0.856 [S] faz_a2/04_commissioni…

| Satır | Sayı/cümle kaynağı | Öneri |
|---|---|---|
| `main.tex:506` | zaman damgası | değişiklik gerekmez (05_gaps §1a; döngü süreleri zaman damgasından) |
| `main.tex:508` | zaman damgası | değişiklik gerekmez (5,88 dk = 05_gaps_minutes_state_total.csv `min_b`) |
| `main.tex:510` | **kare sayısı** (pencere/600) | "about 7.8 and 3.8 min (11.6 min …)" yerine zaman damgalı kayıt süresi: s01 8.5 ve s02 4.6 dk (13.1 dk); "21.7 min of labelled frames" etiketli veri miktarı olarak kalabilir (kare sayısı), ama kayıt süresiyle yan yana konursa duvar saati karşılığı 25.4 dk; "about 1.7 min each" zaman damgalı (20 × 5,0 s), tamam |
| `main.tex:515` | zaman damgası (başlık bunu söylüyor) | değişiklik gerekmez |
| `main.tex:558` | **kare sayısı** (pencere/600) | "about 11.6 min in total" → "about 13 min" (13.05 dk duvar saati) |
| `supplementary.tex:85` | zaman damgası | değişiklik gerekmez |
| `supplementary.tex:306` | **kare sayısı** (metin 'windows divided by 600' diye açıkça söylüyor) | tanım açık olduğu için kendi içinde tutarlı; ama ana metin 510/558 aynı sayıları kayıt süresi diye aktarıyor. Öneri: duvar saati karşılıklarını yan yana ver (pooled S1 kat modeli ≈ 25.4 dk, LOSO ≈ 26.5 dk, s01 8.5, s02 4.6, (ii) kat modelleri ≈ 12.1, tüm-veri ≈ 13.0 dk) |
| `supplementary.tex:307` | zaman damgası | değişiklik gerekmez |
| `supplementary.tex:363` | zaman damgası | değişiklik gerekmez |
| `supplementary.tex:365` | zaman damgası | değişiklik gerekmez |

Ek: `main.tex` 'What the radar stream contains' bölümündeki "About 14.7% of radar periods produced no frame row" cümlesi bu betikle uyumlu (kayıp yuva payı 14.7%: 2801/19073; 05_gaps 2801/19075'i, ham kare toplamıyla, kullanır) ve nedeni iddia etmiyor. **Nedeni olgu gibi yazan yer `reports/paper_prep/05_gaps.md` §1.**

## 7. Okuma, makale cümlesi ve sınırlar

**Önceden yazılmış kurala göre karar: H-bos DESTEKLENDI.** Dayanak (hepsi önceden yazılmış eşiklerle): (a) komşu karelerde nokta sayısı belirgin düşük (medyan 2 nokta, tüm kareler 7; n = 1 payı 28.8% vs 3.2%); (b) komşu karelerin 97.5%'i 'kutu yok' (tüm kareler 56.9%); (c) boşluk aralığındaki ham ToF'de aktif pay 0.9% (rastgele pencere 36.1%) ve aralıkta etiket değişimi yok; (d) boşluk öncesi son 5 karede nokta sayısı eğimi medyan -1.50 (tümü +0.00); (e) yuva başına kayıp tehlikesi 'kutu yok'ta 0.231, kutu aktifken 0.0023 (oran ≈ 102). Önceki ve sonraki komşular ayrı ayrı aynı yönü verir.

**Ne gösterilebildi, ne gösterilemedi.**

- Gösterilen: kayıp kareler bütün radar kareleridir (test (i)); kayıp kareler el kutu dışındayken ve nokta sayısı zaten azalırken kümelenmiştir; ToF/video kaydı boşluk boyunca çalışmaya devam eder (ToF ≈ 31 satır/s, video ≈ 25 kare/s), yani tüm kayıt süreci durmuyor. Bu örüntü 'boş kareler kaydedilmedi' ile uyumlu; 'kaydedici yük altında içerikten bağımsız kare düşürdü' ile uyumsuz (o durumda komşu kareler tüm karelere benzerdi).
- Gösterilemeyen: boş karelerin **kim tarafından** atlandığı (radar yazılımı mı, UART ayrıştırıcı mı, kaydedici mi) ve atlamanın bilinçli bir koşul mu, yoksa 'sıfır nesne → ayrıştırma başarısız → kare düştü' yan etkisi mi olduğu. Kayıt kodu diskte yok (§5). `frame_id` sayacı boşluk boyunca ilerlemediği için ardışıklığı iki hipotezde de aynıdır ve ayrım sağlamaz.
- Sınırlar: (1) 'nokta az' ile 'kare yok' sebep-sonuç değil korelasyondur; ortak bir neden (ör. el uzakken ayrıştırıcının yavaşlaması) bu verilerle tamamen dışlanamaz, ama içerikten bağımsız rastgele düşürme dışlanır. (2) Nokta sayısı dağılımından sıfıra Poisson/geometrik ekstrapolasyon (S2) kayıp/kayıtlı oranından (0.172) çok küçük (0.013–0.025); bu, H-boş'u nicel olarak ne destekler ne çürütür (nokta sayısı 'kutu yok' anlarında bimodal olabilir: boş sahnede ya hiç ya birkaç nokta), yalnız ekstrapolasyon kaba bir araçtır. (3) Zaman damgaları sunucu saatiyle basılır; tolerans ±5.0 ms.

**Fiziksel yorum (varsayım, doğrulanmadı).** Yapılandırmada `clutterRemoval -1 1` (statik kirlilik temizleme açık). El kutu dışında ve durağanken radar hareketli nokta üretmeyebilir; sıfır nesneli kare çıktıda ya hiç yer almaz ya da kaydedilmez. Ham dosyada kayıtlı karelerin tümü ≥ 1 hareketli nokta içerir (yalnız-statik kare yok, §4 S4) — bu da 'v = 0 olan ama hareketli noktası olmayan kare' diye bir sınıfın hiç kaydedilmediğini gösterir. Kayıt kodu görülünce doğrulanır.

**Makale cümlesi.** Plan kuralına göre H-boş destekleniyor; önerilen cümle (kullanıcı onayı olmadan `main.tex`'e girmez; "About 14.7% of radar periods produced no frame row" cümlesinin yerine geçebilir):

> *Frames in which the radar reported no points were not logged (2,801 slots, 14.7% of radar periods); a 5-frame window therefore spans more than 0.5 s in 13.0% of windows.*

Doğrulanabilirlik uyarısı: "no points … were not logged" kısmı kayıt koduyla doğrulanmadı (yalnız dolaylı kanıt). Temkinli alternatif: "Radar periods without a logged frame (2,801 slots, 14.7%) occurred almost only while no box was active and the point count was already low; a 5-frame window therefore spans more than 0.5 s in 13.0% of windows." Pencere payı (2114/16248) Faz B B.0(f) ile aynı; boşluklu pencerelerin 2073'ü (98.1%) 'none' sınıfında, yani boşluk neredeyse hiç kutu aktifken olmuyor. Pencere payı tanımı: en az bir ardışık aralığın > 0,15 s olması (%13.0); alternatif tanım (son−ilk kare + 0,1 s > 0,5 s + tolerans) %13.4 verir (jitter birikimi nedeniyle biraz şişer); ≥ 1 kayıp yuva içeren (kapsam > 0,6 s) pencere payı %11.2.

**Düzeltme önerileri (özet; dosyalara dokunulmadı).**

1. `main.tex` 510 ve 558 ile `supplementary.tex` 306: kare sayısından hesaplanmış dakikalar (11.6 / 21.7 / 7.8 / 3.8 / 10.8 / 22.6) kayıt süresi diye kullanılıyorsa zaman damgalı karşılıklarla değiştirilmeli (s01 8.5 dk, s02 4.6 dk, toplam 13.1 dk). Fark tüm oturumlarda %17 (toplam oran 1.172); s01+s02 için oran 1.126.
2. `reports/faz_a4/04_unseen_minutes.csv` ve `04_unseen_sessions.py` L261–273: `etiketli_dakika` kare sayısıdır; sütun adı 'etiketli_kare_dakikası' yapılmalı ya da zaman damgalı sütun eklenmeli (öneri sütunu `02_time_gaps_t5_minutes_proposal.csv`).
3. `reports/paper_prep/05_gaps.md` §1: "`merged_df`/the frame table contain only radar frames that had at least one point" cümlesi, olgu gibi yazılmış bir hipotezdir; 'dolaylı kanıtla uyumlu, kayıt koduyla doğrulanmamış' diye yumuşatılmalı.

## 8. Dosyalar

`faz_c/02_time_gaps.py`, `faz_c/02_time_gaps_search.txt` (test 4 arama çıktısı), `reports/faz_c/02_time_gaps.md`, `02_time_gaps.csv` (tüm sayılar; sütunlar section,item,stat,value,n,note), `02_time_gaps_t1_by_k.csv`, `02_time_gaps_t1_resid_hist.csv`, `02_time_gaps_t2_neighbors.csv`, `02_time_gaps_t3_sessions.csv`, `02_time_gaps_gaps_detail.csv` (696 boşluğun tek tek özellikleri), `02_time_gaps_t5_wall.csv`, `02_time_gaps_t5_minutes_proposal.csv`.


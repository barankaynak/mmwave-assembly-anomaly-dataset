# A4.2 (M2) - Tempo tespitinin kanal dokumu ve oracle'in slow'daki davranisi (2026-11-05)

Brief: `faz_a4/DESIGN_A4.md` (A4.2). Betik: `faz_a4/02_tempo_channels.py`. Girdi: `reports/faz_a/04_cycle_scores.csv.gz` (role = test).
Birim = (dongu, f) cifti (Faz A 'tespit' tanimi; anomali donguleri her katta test). GA = dongu-kumeli bootstrap (dongu basina 5 kat ortalamasi, 4000 tekrar); normal FAR icin Wilson. Kanal esigi: D3 alarmi (alpha = 0.10) <=> min(p1,p2) <= 0.05; yani 'D1 kanali' = p1 <= 0.05, 'D2 kanali' = p2 <= 0.05, D3 alarmi = bunlarin birlesimi (betikte dogrulandi, her satirda).
Kontrol: a2_cycles ile ToF x tof ve B2 x pred donguleri yeniden kuruldu; CSV'deki visit_seq ve atype birebir ayni b2 x pred: 254/254; tof x tof: 258/258. Oracle slow icin D2 degerleri adim adim yeniden uretildi (fark < 1e-9).

## 1. D3 alarmi alan fast/slow donguleri: hangi kanal?

### B2 x pred

| tur | n cift | D3 alarm (oran [GA]) | yalniz D1 (D3 alarmlarinin %'si) | yalniz D2 | ikisi birden | alarm yok |
|---|---|---|---|---|---|---|
| fast | 240 (48 dongu) | 221 (0.921 [0.862; 0.967]) | 36 (16.3%) | 169 (76.5%) | 16 (7.2%) | 19 |
| slow | 255 (51 dongu) | 237 (0.929 [0.875; 0.973]) | 0 (0.0%) | 237 (100.0%) | 0 (0.0%) | 18 |
| tempo (fast+slow) | 495 (99 dongu) | 458 (0.925 [0.887; 0.960]) | 36 (7.9%) | 406 (88.6%) | 16 (3.5%) | 37 |


### ToF (oracle) x tof

| tur | n cift | D3 alarm (oran [GA]) | yalniz D1 (D3 alarmlarinin %'si) | yalniz D2 | ikisi birden | alarm yok |
|---|---|---|---|---|---|---|
| fast | 245 (49 dongu) | 228 (0.931 [0.849; 0.992]) | 163 (71.5%) | 18 (7.9%) | 47 (20.6%) | 17 |
| slow | 260 (52 dongu) | 156 (0.600 [0.600; 0.600]) | 0 (0.0%) | 156 (100.0%) | 0 (0.0%) | 104 |
| tempo (fast+slow) | 505 (101 dongu) | 384 (0.760 [0.715; 0.806]) | 163 (42.4%) | 174 (45.3%) | 47 (12.2%) | 121 |


Diger iki hucre (ToF x pred, B2 x tof) icin ayni dokum: `02_tempo_channel_split.csv`.

## 2. Fast donguler: kisalmis dizi (ziyaret dizisinde Box5 yok) ve D1 ile alarm

### B2 x pred

| tur | grup | n cift | kisalmis payi | D1 alarm (p1<=0.05) | D2 alarm (p2<=0.05) | D3 alarm |
|---|---|---|---|---|---|---|
| normal | tum | 51 (51 dongu) | 0.000 | 1/51 = 0.020 | 3/51 = 0.059 | 4/51 = 0.078 |
| normal | kisalmis(Box5 yok) | 0 (0 dongu) |  | - | - | - |
| normal | tam(Box5 var) | 51 (51 dongu) |  | 1/51 = 0.020 | 3/51 = 0.059 | 4/51 = 0.078 |
| fast | tum | 240 (48 dongu) | 0.271 | 52/240 = 0.217 | 185/240 = 0.771 | 221/240 = 0.921 |
| fast | kisalmis(Box5 yok) | 65 (13 dongu) |  | 52/65 = 0.800 | 19/65 = 0.292 | 55/65 = 0.846 |
| fast | tam(Box5 var) | 175 (35 dongu) |  | 0/175 = 0.000 | 166/175 = 0.949 | 166/175 = 0.949 |
| slow | tum | 255 (51 dongu) | 0.000 | 0/255 = 0.000 | 237/255 = 0.929 | 237/255 = 0.929 |
| slow | kisalmis(Box5 yok) | 0 (0 dongu) |  | - | - | - |
| slow | tam(Box5 var) | 255 (51 dongu) |  | 0/255 = 0.000 | 237/255 = 0.929 | 237/255 = 0.929 |


### ToF (oracle) x tof

| tur | grup | n cift | kisalmis payi | D1 alarm (p1<=0.05) | D2 alarm (p2<=0.05) | D3 alarm |
|---|---|---|---|---|---|---|
| normal | tum | 51 (51 dongu) | 0.000 | 0/51 = 0.000 | 4/51 = 0.078 | 4/51 = 0.078 |
| normal | kisalmis(Box5 yok) | 0 (0 dongu) |  | - | - | - |
| normal | tam(Box5 var) | 51 (51 dongu) |  | 0/51 = 0.000 | 4/51 = 0.078 | 4/51 = 0.078 |
| fast | tum | 245 (49 dongu) | 0.816 | 210/245 = 0.857 | 65/245 = 0.265 | 228/245 = 0.931 |
| fast | kisalmis(Box5 yok) | 200 (40 dongu) |  | 200/200 = 1.000 | 47/200 = 0.235 | 200/200 = 1.000 |
| fast | tam(Box5 var) | 45 (9 dongu) |  | 10/45 = 0.222 | 18/45 = 0.400 | 28/45 = 0.622 |
| slow | tum | 260 (52 dongu) | 0.000 | 0/260 = 0.000 | 156/260 = 0.600 | 156/260 = 0.600 |
| slow | kisalmis(Box5 yok) | 0 (0 dongu) |  | - | - | - |
| slow | tam(Box5 var) | 260 (52 dongu) |  | 0/260 = 0.000 | 156/260 = 0.600 | 156/260 = 0.600 |


`kisalmis payi` yalniz 'tum' satirinda: tur icindeki (dongu, f) ciftlerinin Box5'siz olan kismi. Hem normal hem anomali icin Wilson GA'li tum oranlar: `02_tempo_truncation.csv` (seq135/seq145 dahil; diger hucreler dahil).

## 3. Onceden yazilmis ikincil tempo tanimi: D2 tek basina, alpha = 0.10 ('tempo kanali')

Makalede 'tempo deviations detected' bu sayiyla nitelenir. Alarm = p2 <= 0.10 (D3'un Bonferroni esiginden farkli: alpha bolunmez). Referans olarak D2 tek basina alpha = 0.05 ve D3 alpha = 0.10 de verilir.

| hucre | tur | D2 tek basina a=0.10 (tespit [GA]) | D2 tek basina a=0.05 | D3 a=0.10 (Faz A tespit) |
|---|---|---|---|---|
| B2 x pred | fast | 0.800 [0.688; 0.900] | 0.771 | 0.921 [0.863; 0.967] |
| B2 x pred | slow | 0.992 [0.980; 1.000] | 0.929 | 0.929 [0.871; 0.976] |
| B2 x pred | tempo | 0.899 [0.840; 0.947] | 0.853 | 0.925 [0.885; 0.960] |
| ToF x tof | fast | 0.322 [0.208; 0.445] | 0.265 | 0.931 [0.857; 0.992] |
| ToF x tof | slow | 0.881 [0.842; 0.915] | 0.600 | 0.600 [0.600; 0.600] |
| ToF x tof | tempo | 0.610 [0.529; 0.691] | 0.438 | 0.760 [0.717; 0.808] |
| ToF x pred | fast | 0.365 [0.230; 0.504] | 0.304 | 0.930 [0.848; 0.996] |
| ToF x pred | slow | 0.988 [0.973; 1.000] | 0.882 | 0.882 [0.843; 0.918] |
| ToF x pred | tempo | 0.693 [0.600; 0.781] | 0.608 | 0.905 [0.862; 0.942] |
| B2 x tof | fast | 0.698 [0.600; 0.784] | 0.673 | 0.788 [0.735; 0.837] |
| B2 x tof | slow | 0.877 [0.800; 0.942] | 0.827 | 0.827 [0.738; 0.904] |
| B2 x tof | tempo | 0.790 [0.729; 0.850] | 0.752 | 0.808 [0.756; 0.857] |


Normal (yanlis alarm) karsiligi, ayni test ciftleri uzerinden (Wilson %95 GA):

| hucre | n normal | FAR D2 tek basina a=0.10 | FAR D2 a=0.05 | FAR D3 a=0.10 |
|---|---|---|---|---|
| B2 x pred | 51 | 0.078 [0.031; 0.185] | 0.059 | 0.078 [0.031; 0.185] |
| ToF x tof | 51 | 0.098 [0.043; 0.210] | 0.078 | 0.078 [0.031; 0.185] |
| ToF x pred | 50 | 0.080 [0.032; 0.188] | 0.060 | 0.060 [0.021; 0.162] |
| B2 x tof | 51 | 0.039 [0.011; 0.132] | 0.039 | 0.059 [0.020; 0.159] |


## 4. Oracle (ToF x tof) slow'da neden ~%60'ta kaliyor? (betimleyici)

### ToF x tof (brief'in hucresi)

| olcut | slow tespit [GA] | normal FAR |
|---|---|---|
| D1 (p1<=0.05) | 0/260 = 0.000 [0.000; 0.000] | 0.000 |
| D2 (p2<=0.05) | 156/260 = 0.600 [0.600; 0.600] | 0.078 |
| D3 (alpha=0.10) | 156/260 = 0.600 [0.600; 0.600] | 0.078 |
| D2 tek basina (p2<=0.10) | 229/260 = 0.881 [0.842; 0.919] | 0.098 |
| D1 tek basina (p1<=0.10) | 0/260 = 0.000 [0.000; 0.000] | 0.000 |


Slow ciftlerinde p dagilimi (p1, p2, p3 = min(1, 2 min(p1,p2))). Alarm esikleri: p1, p2 icin <= 0.05; p3 icin <= 0.10 (yani p3 icin alarm kutusu (.05,.10] sutunu ve altidir; <=.05 sutunu p3 icin anlamsizdir):

| p | n | min | q10 | q25 | medyan | q75 | q90 | max | <=.05 | (.05,.10] | (.10,.20] | (.20,.50] | >.50 | p=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p1 | 260 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 0 | 260 | 260 |
| p2 | 260 | 0.045 | 0.045 | 0.048 | 0.048 | 0.095 | 0.136 | 0.182 | 156 | 73 | 31 | 0 | 0 | 0 |
| p3 | 260 | 0.091 | 0.091 | 0.095 | 0.095 | 0.190 | 0.273 | 0.364 | 0 | 156 | 73 | 31 | 0 | 0 |


D2'yi belirleyen adim (slow ciftleri, en buyuk |z|), D2 alarmina gore:

| D2'yi belirleyen adim | D2 alarm | D2 alarm yok |
|---|---|---|
| dwell_1 | 4 | 1 |
| dwell_2 | 3 | 2 |
| dwell_5 | 0 | 3 |
| gap_1_2 | 138 | 86 |
| gap_2_5 | 11 | 12 |


### ToF x pred (ek: ayni oracle etiketler, tahmin-segmentasyonu)

| olcut | slow tespit [GA] | normal FAR |
|---|---|---|
| D1 (p1<=0.05) | 0/255 = 0.000 [0.000; 0.000] | 0.000 |
| D2 (p2<=0.05) | 225/255 = 0.882 [0.843; 0.918] | 0.060 |
| D3 (alpha=0.10) | 225/255 = 0.882 [0.843; 0.918] | 0.060 |
| D2 tek basina (p2<=0.10) | 252/255 = 0.988 [0.973; 1.000] | 0.080 |
| D1 tek basina (p1<=0.10) | 0/255 = 0.000 [0.000; 0.000] | 0.000 |


Slow ciftlerinde p dagilimi (p1, p2, p3 = min(1, 2 min(p1,p2))). Alarm esikleri: p1, p2 icin <= 0.05; p3 icin <= 0.10 (yani p3 icin alarm kutusu (.05,.10] sutunu ve altidir; <=.05 sutunu p3 icin anlamsizdir):

| p | n | min | q10 | q25 | medyan | q75 | q90 | max | <=.05 | (.05,.10] | (.10,.20] | (.20,.50] | >.50 | p=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p1 | 255 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 0 | 255 | 255 |
| p2 | 255 | 0.045 | 0.045 | 0.048 | 0.048 | 0.050 | 0.095 | 0.143 | 225 | 27 | 3 | 0 | 0 | 0 |
| p3 | 255 | 0.091 | 0.091 | 0.095 | 0.095 | 0.100 | 0.190 | 0.286 | 0 | 225 | 21 | 9 | 0 | 0 |


D2'yi belirleyen adim (slow ciftleri, en buyuk |z|), D2 alarmina gore:

| D2'yi belirleyen adim | D2 alarm | D2 alarm yok |
|---|---|---|
| dwell_1 | 4 | 1 |
| dwell_2 | 3 | 2 |
| dwell_5 | 0 | 3 |
| gap_1_2 | 200 | 20 |
| gap_2_5 | 18 | 4 |


Betimleyici ayrintilar (kat bazinda tespit, dongu bazinda tutarlilik, alarm/kacirilan donguler, sure ve skor medyanlari) stdout'tadir ve asagida ozetlenmistir.

Kat bazinda slow tespiti (ToF x tof; D1 / D2 / D3): f=0: 0.00 / 1.00 / 1.00; f=1: 0.00 / 1.00 / 1.00; f=2: 0.00 / 0.00 / 0.00; f=3: 0.00 / 0.00 / 0.00; f=4: 0.00 / 1.00 / 1.00.

Dongu bazinda D3: 0 dongu 5/5 katta alarm, 0 dongu hic alarm yok, 52 dongu karisik (toplam 52 slow dongusu).

- alarm (D3): n = 156 cift; dongu suresi medyan 8.30 s [IQR 7.48; 8.70]; D0 medyan 1.26, D1 medyan 0.21, D2 medyan 4.60; ziyaret dizisi {'1-2-5': 156}.
- kacirilan (D3): n = 104 cift; dongu suresi medyan 8.30 s [IQR 7.48; 8.70]; D0 medyan 5.23, D1 medyan 0.21, D2 medyan 5.15; ziyaret dizisi {'1-2-5': 104}.
- normal (referans): dongu suresi medyan 5.00 s [IQR 4.55; 5.25]; D0 medyan 0.55, D2 medyan 1.35.

### Kat bazinda: slow tespiti kalibrasyon kumesine bagli (D2 skoru ve kalibrasyon ustleri)

D2 alarmi p2 <= 0.05 gerektirir; n_cal ~ 21 oldugundan bu, test D2 skorunun o kattaki kalibrasyon kumesinin **en yuksek** D2'sini asmasi demektir (en kucuk p = 1/(n_cal+1)). Slow icin kat bazinda (D2 skoru medyani, kalibrasyon ilk uc D2, en yuksegi asan cift sayisi, tespit):

**B2 x pred, slow**

| f | n_cal | test D2 medyan [IQR] | cal D2 ilk 3 | D2 >= cal max | D1 (p1<=.05) | D2 (p2<=.05) | D3 a=0.10 | D2 tek basina a=0.10 |
|---|---|---|---|---|---|---|---|---|
| 0 | 19 | 4.11 [3.53; 4.66] | 3.22 / 2.48 / 2.19 | 45/51 | 0.00 | 0.88 | 0.88 | 1.00 |
| 1 | 20 | 5.17 [4.47; 6.02] | 4.00 / 3.25 / 2.98 | 45/51 | 0.00 | 0.88 | 0.88 | 0.98 |
| 2 | 21 | 4.63 [3.82; 5.30] | 3.13 / 2.85 / 2.56 | 46/51 | 0.00 | 0.90 | 0.90 | 0.98 |
| 3 | 22 | 5.00 [4.17; 5.95] | 3.00 / 2.60 / 1.94 | 51/51 | 0.00 | 1.00 | 1.00 | 1.00 |
| 4 | 20 | 4.26 [3.51; 4.65] | 2.53 / 2.22 / 1.96 | 50/51 | 0.00 | 0.98 | 0.98 | 1.00 |


**B2 x pred, fast**

| f | n_cal | test D2 medyan [IQR] | cal D2 ilk 3 | D2 >= cal max | D1 (p1<=.05) | D2 (p2<=.05) | D3 a=0.10 | D2 tek basina a=0.10 |
|---|---|---|---|---|---|---|---|---|
| 0 | 19 | 4.20 [3.75; 4.23] | 3.22 / 2.48 / 2.19 | 36/48 | 0.27 | 0.75 | 0.96 | 0.79 |
| 1 | 20 | 4.85 [3.92; 4.88] | 4.00 / 3.25 / 2.98 | 36/48 | 0.27 | 0.75 | 0.96 | 0.75 |
| 2 | 21 | 4.61 [3.39; 4.64] | 3.13 / 2.85 / 2.56 | 36/48 | 0.00 | 0.75 | 0.75 | 0.77 |
| 3 | 22 | 4.05 [3.23; 4.07] | 3.00 / 2.60 / 1.94 | 39/48 | 0.27 | 0.81 | 0.98 | 0.83 |
| 4 | 20 | 3.48 [3.36; 3.56] | 2.53 / 2.22 / 1.96 | 38/48 | 0.27 | 0.79 | 0.96 | 0.85 |


**ToF (oracle) x tof, slow**

| f | n_cal | test D2 medyan [IQR] | cal D2 ilk 3 | D2 >= cal max | D1 (p1<=.05) | D2 (p2<=.05) | D3 a=0.10 | D2 tek basina a=0.10 |
|---|---|---|---|---|---|---|---|---|
| 0 | 20 | 4.50 [3.71; 5.18] | 2.73 / 2.48 / 2.26 | 52/52 | 0.00 | 1.00 | 1.00 | 1.00 |
| 1 | 20 | 5.59 [4.60; 6.46] | 2.77 / 2.43 / 2.28 | 52/52 | 0.00 | 1.00 | 1.00 | 1.00 |
| 2 | 20 | 5.31 [4.37; 6.17] | 11.27 / 4.03 / 3.03 | 0/52 | 0.00 | 0.00 | 0.00 | 0.88 |
| 3 | 21 | 5.06 [4.33; 5.51] | 11.05 / 4.97 / 3.51 | 0/52 | 0.00 | 0.00 | 0.00 | 0.52 |
| 4 | 21 | 4.25 [3.71; 4.67] | 2.62 / 2.45 / 2.03 | 52/52 | 0.00 | 1.00 | 1.00 | 1.00 |


**ToF (oracle) x tof, fast**

| f | n_cal | test D2 medyan [IQR] | cal D2 ilk 3 | D2 >= cal max | D1 (p1<=.05) | D2 (p2<=.05) | D3 a=0.10 | D2 tek basina a=0.10 |
|---|---|---|---|---|---|---|---|---|
| 0 | 20 | 2.48 [1.72; 3.03] | 2.73 / 2.48 / 2.26 | 17/49 | 0.86 | 0.35 | 0.94 | 0.43 |
| 1 | 20 | 2.25 [2.21; 3.93] | 2.77 / 2.43 / 2.28 | 17/49 | 0.86 | 0.35 | 0.94 | 0.35 |
| 2 | 20 | 2.55 [2.51; 4.23] | 11.27 / 4.03 / 3.03 | 7/49 | 0.86 | 0.14 | 0.92 | 0.35 |
| 3 | 21 | 2.15 [2.12; 3.61] | 11.05 / 4.97 / 3.51 | 7/49 | 0.86 | 0.14 | 0.92 | 0.14 |
| 4 | 21 | 1.80 [1.57; 2.75] | 2.62 / 2.45 / 2.03 | 17/49 | 0.86 | 0.35 | 0.94 | 0.35 |


ToF x tof: her katin kalibrasyon kumesindeki en yuksek D2 ve sahibi dongu (yalniz kalibrasyon-rolu skorlari; alarm durumuna bakilmadi):

| f | oturum | dongu | ToF kati | en yuksek cal D2 | sonraki cal D2 | dongu suresi (s) | dizi |
|---|---|---|---|---|---|---|---|
| 0 | session_02_normal_seq125 | 19 | 2 | 2.728 | 2.481 | 4.801 | 1-2-5 |
| 1 | session_02_normal_seq125 | 1 | 3 | 2.766 | 2.432 | 5.9 | 1-2-5 |
| 2 | session_02_normal_seq125 | 51 | 4 | 11.274 | 4.026 | 24.998 | 1-2-5 |
| 3 | session_02_normal_seq125 | 51 | 4 | 11.052 | 4.974 | 24.998 | 1-2-5 |
| 4 | session_02_normal_seq125 | 21 | 0 | 2.624 | 2.453 | 6.0 | 1-2-5 |


**Betimleyici ozet (oracle slow):** ToF x tof slow tespiti 0.600 = {f=0,1,4: 52/52 cift} + {f=2,3: 0/52}. f=2 ve f=3 kalibrasyon kumelerinin ikisi de ayni tek normal dongu icerir (session_02 dongu 51, 25.0 s, D2 = 11.3 / 11.1; sonraki en yuksek cal D2 4.0 / 5.0). p2 <= 0.05 icin D2 >= kalibrasyon maksimumu gerekir; bu iki katta hicbir slow dongusu 11'i asmaz (n_above_top1 = 0/52), ikinci en yuksegi asan 46/52 ve 27/52 cift vardir (D2 tek basina alpha = 0.10: 0.885 ve 0.519). Diger uc katta ilk uc cal D2 2.0-2.8 ve butun slow donguleri bunu asar. Yani %60, slow sinyalinin zayifligindan degil, **iki kalibrasyon kumesindeki tek bir uc degerden** gelir (tek ornek; olcut: n_cal ~ 20, D3 'tum kalibrasyon skorlarini asma' kurali).

Not: slow/fast anomali donguleri her katta test oldugu icin, bir katin kalibrasyon kumesi (20 civari normal dongu) o kattaki tespit oranini tek basina belirler; bu yuzden dongu-kumeli bootstrap GA'si bu degiskenligi (kalibrasyon kumesi degiskenligini) YANSITMAZ, yalniz dongulerin ornekleme degiskenligini yansitir. Oracle slow'da D3 = 0.600 [0.600; 0.600] bu durumun uc ornegidir: her dongu 3/5 katta alarm verir, bootstrap'in varyansi sifirdir. Gercek belirsizlik kat-duzeyidir (5 kat); kat bazinda aralik tablolarda.

**Not (betimleyici, nedensellik test edilmedi):** yukaridaki sayilar slow kacirmalarinin hangi kanalda ve hangi p araliginda yigildigini gosterir; 'neden' sorusuna tek bir mekanizma onerilmez.

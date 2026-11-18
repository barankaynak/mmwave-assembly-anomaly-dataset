# A3.3 Cevrim ici (online) alarm - dongu icinde

**Statu:** Faz A sonuclari BIRINCILdir; A3.3 sonradan yapilan IKINCIL bir analizdir (ana model B2 + D3 degismez).

Kaynak: `faz_a3/03_online.py`; on-kayit: `faz_a3/DESIGN_A3.md` A3.3. Hucreler: B2 x pred (ana), ToF x tof (referans). Birim: (dongu, f) cifti; anomali donguleri her f'de test.

## 0. SERT KONTROL - cevrimici alarm kumesi == Faz A cevrimdisi D3 alarm kumesi (alpha = 0.10, kat basina)

| hucre | f | n_test | online alarm | Faz A alarm | fark (simetrik) | anahtar kumesi esit |
|---|---|---|---|---|---|---|
| b2 x pred | 0 | 214 | 101 | 101 | 0 | True |
| b2 x pred | 1 | 212 | 196 | 196 | 0 | True |
| b2 x pred | 2 | 213 | 92 | 92 | 0 | True |
| b2 x pred | 3 | 213 | 151 | 151 | 0 | True |
| b2 x pred | 4 | 214 | 107 | 107 | 0 | True |
- b2 x pred: sonraki Box1 girisi bulunamayan (son olaylari oturum sonunda skorlanan) cift sayisi: 0
- b2 x pred: nedensel (causal) varyantin alarm kumesi literal ile farkli olan cift sayisi: 0
| tof x tof | 0 | 218 | 206 | 206 | 0 | True |
| tof x tof | 1 | 217 | 204 | 204 | 0 | True |
| tof x tof | 2 | 217 | 151 | 151 | 0 | True |
| tof x tof | 3 | 217 | 151 | 151 | 0 | True |
| tof x tof | 4 | 217 | 206 | 206 | 0 | True |
- tof x tof: sonraki Box1 girisi bulunamayan (son olaylari oturum sonunda skorlanan) cift sayisi: 21
- tof x tof: nedensel (causal) varyantin alarm kumesi literal ile farkli olan cift sayisi: 0

**SERT KONTROL GECTI** (tum hucre x kat; ayrica her cift icin son kosan maksimumlar D1/D2 cevrimdisi skorlarla 1e-12 icinde ayni).

## ANA (dagitilabilir): B2 x pred

### Alarm zamani - gecen dongu orani (literal tanim; (dongu,f) ciftleri havuzlu, yalniz alarm alan ciftler)

| tur | n_cift | n_alarm | alarm orani | medyan oran | Q25 | Q75 | oran>1 payi | son ziyaretten ONCE verilen pay (alarm < son ziyaret baslangici) | ... (alarm < son ziyaret onayi) | son olayda (->END/son dwell) verilen pay |
|---|---|---|---|---|---|---|---|---|---|---|
| normal | 51 | 4 | 0.078 | 0.295 | 0.286 | 0.483 | 0.250 | 0.750 | 0.750 | 0.250 |
| seq135 | 280 | 71 | 0.254 | 0.289 | 0.263 | 0.353 | 0.085 | 0.901 | 0.901 | 0.085 |
| seq145 | 240 | 114 | 0.475 | 0.293 | 0.257 | 0.330 | 0.105 | 0.895 | 0.895 | 0.105 |
| fast | 240 | 221 | 0.921 | 1.025 | 1.021 | 1.028 | 0.801 | 0.158 | 0.158 | 0.801 |
| slow | 255 | 237 | 0.929 | 0.317 | 0.282 | 0.359 | 0.055 | 0.899 | 0.899 | 0.055 |

### Gercek baslangictan saniye (seq135/seq145; baslangic = box3/box4'e ilk ToF girisi, merged_df)

| tur | n_alarm (baslangic var) | medyan s | Q25 | Q75 | alarm <= baslangic payi | baslangicin dongudeki oransal konumu (medyan [Q25,Q75]) | baslangici bulunamayan cift |
|---|---|---|---|---|---|---|---|
| seq135 | 71 | 0.100 | 0.099 | 0.102 | 0.239 | 0.262 [0.244, 0.289] | 0 |
| seq145 | 114 | 0.100 | 0.100 | 0.200 | 0.149 | 0.261 [0.237, 0.286] | 0 |

### Alarmi tetikleyen olay turu (sayi)

| tur | trig_dwell | trig_trans_mid | trig_gap | trig_trans_end | trig_trans_start |
|---|---|---|---|---|---|
| normal | 2 | 1 | 1 | 0 | 0 |
| seq135 | 14 | 56 | 0 | 1 | 0 |
| seq145 | 20 | 94 | 0 | 0 | 0 |
| fast | 172 | 3 | 13 | 33 | 0 |
| slow | 47 | 0 | 190 | 0 | 0 |

### Duyarlilik: nedensel (causal) tanim (onay = ham etiketin MIN_BOX'uncu karesi; ziyaret sonu e+MAX_GAP'te bilinir) - alarm kumesi ayni

| tur | medyan oran | Q25 | Q75 | son ziyaretten once pay | saniye medyan (seq) [Q25,Q75] |
|---|---|---|---|---|---|
| normal | 0.305 | 0.300 | 0.484 | 0.750 | - |
| seq135 | 0.289 | 0.263 | 0.359 | 0.901 | 0.100 [0.099, 0.102] |
| seq145 | 0.294 | 0.257 | 0.330 | 0.895 | 0.100 [0.100, 0.200] |
| fast | 1.025 | 1.021 | 1.028 | 0.158 | - |
| slow | 0.317 | 0.282 | 0.359 | 0.899 | - |

## REFERANS: ToF x tof

### Alarm zamani - gecen dongu orani (literal tanim; (dongu,f) ciftleri havuzlu, yalniz alarm alan ciftler)

| tur | n_cift | n_alarm | alarm orani | medyan oran | Q25 | Q75 | oran>1 payi | son ziyaretten ONCE verilen pay (alarm < son ziyaret baslangici) | ... (alarm < son ziyaret onayi) | son olayda (->END/son dwell) verilen pay |
|---|---|---|---|---|---|---|---|---|---|---|
| normal | 51 | 4 | 0.078 | 0.689 | 0.457 | 0.882 | 0.000 | 0.500 | 0.500 | 0.250 |
| seq135 | 285 | 285 | 1.000 | 0.283 | 0.264 | 0.300 | 0.000 | 0.842 | 0.842 | 0.000 |
| seq145 | 245 | 245 | 1.000 | 0.286 | 0.262 | 0.308 | 0.000 | 0.918 | 0.918 | 0.000 |
| fast | 245 | 228 | 0.931 | 1.023 | 0.350 | 1.028 | 0.627 | 0.079 | 0.079 | 0.627 |
| slow | 260 | 156 | 0.600 | 0.310 | 0.284 | 0.338 | 0.000 | 0.994 | 0.994 | 0.000 |

### Gercek baslangictan saniye (seq135/seq145; baslangic = box3/box4'e ilk ToF girisi, merged_df)

| tur | n_alarm (baslangic var) | medyan s | Q25 | Q75 | alarm <= baslangic payi | baslangicin dongudeki oransal konumu (medyan [Q25,Q75]) | baslangici bulunamayan cift |
|---|---|---|---|---|---|---|---|
| seq135 | 285 | 0.100 | 0.100 | 0.100 | 0.014 | 0.261 [0.244, 0.280] | 0 |
| seq145 | 245 | 0.100 | 0.100 | 0.100 | 0.000 | 0.262 [0.239, 0.282] | 0 |

### Alarmi tetikleyen olay turu (sayi)

| tur | trig_dwell | trig_trans_mid | trig_gap | trig_trans_end | trig_trans_start |
|---|---|---|---|---|---|
| normal | 3 | 0 | 1 | 0 | 0 |
| seq135 | 4 | 281 | 0 | 0 | 0 |
| seq145 | 0 | 240 | 0 | 0 | 5 |
| fast | 3 | 10 | 62 | 143 | 10 |
| slow | 22 | 0 | 134 | 0 | 0 |

### Duyarlilik: nedensel (causal) tanim (onay = ham etiketin MIN_BOX'uncu karesi; ziyaret sonu e+MAX_GAP'te bilinir) - alarm kumesi ayni

| tur | medyan oran | Q25 | Q75 | son ziyaretten once pay | saniye medyan (seq) [Q25,Q75] |
|---|---|---|---|---|---|
| normal | 0.697 | 0.473 | 0.882 | 0.500 | - |
| seq135 | 0.283 | 0.264 | 0.300 | 0.842 | 0.100 [0.100, 0.100] |
| seq145 | 0.286 | 0.262 | 0.308 | 0.918 | 0.100 [0.100, 0.100] |
| fast | 1.023 | 0.350 | 1.028 | 0.079 | - |
| slow | 0.310 | 0.284 | 0.338 | 0.994 | - |

## On-kayitli tahminler

| tahmin | hucre | tur | gozlenen medyan | sonuc |
|---|---|---|---|---|
| medyan oran <0.5 | b2xpred (ana) | seq135 | 0.289 | TUTTU |
| medyan oran <0.5 | b2xpred (ana) | seq145 | 0.293 | TUTTU |
| medyan oran >0.8 | b2xpred (ana) | fast | 1.025 | TUTTU |
| medyan oran >0.8 | b2xpred (ana) | slow | 0.317 | TUTMADI |
| medyan oran <0.5 | tofxtof (ref) | seq135 | 0.283 | TUTTU |
| medyan oran <0.5 | tofxtof (ref) | seq145 | 0.286 | TUTTU |
| medyan oran >0.8 | tofxtof (ref) | fast | 1.023 | TUTTU |
| medyan oran >0.8 | tofxtof (ref) | slow | 0.310 | TUTMADI |

Yorum (sonuclarin okunusu): slow tempo B2 x pred'de alarmlarin cogunda ilk ziyaretler arasi bosluk (gap, Box2 onayinda skorlanir) ile ~dongunun 0.3'unde yakalaniyor; yani 'slow gec yakalanir' tahmini TUTMADI. fast ise yalniz dongu tamamlaninca (son dwell / ->END, sonraki Box1 onayinda skorlanir) yakalaniyor (medyan oran ~1.0). DESIGN cumlesindeki 'tempo sapmalari ancak dongu sonuna dogru' yalniz fast icin dogru; slow icin dogru degil.
fast'te oran > 1 (~1.02-1.03): son olay sonraki Box1 ziyaretinin onay karesinde (dongu sinirindan MIN_BOX-1 = 1 kare = 0.1 s sonra) skorlandigi icin; 1 + 0.1/dur_s ile uyumlu, literal ->END kuraliyla bekleniyor.
- b2xpred seq135: baslangic anina esit/oncesi verilen alarm 17/71; tetikleyen olay dagilimi: {'dwell': 9, 'trans_mid': 8}
- b2xpred seq145: baslangic anina esit/oncesi verilen alarm 17/114; tetikleyen olay dagilimi: {'trans_mid': 12, 'dwell': 5}
- tofxtof seq135: baslangic anina esit/oncesi verilen alarm 4/285; tetikleyen olay dagilimi: {'dwell': 4}
- tofxtof seq145: baslangic anina esit/oncesi verilen alarm 0/245; tetikleyen olay dagilimi: {}
- Baslangictan saniye medyani ~0.1 s, 10 Hz'de MIN_BOX onay gecikmesinin (onay = ziyaret baslangici + 1 kare) kendisidir; ToF x tof'ta tum ceyreklikler tam 0.100 s cikmasi bundandir. Dolayisiyla saniye degeri 'tespit gecikmesi' degil, onay kurali tabanidir.

Sekil: `reports/faz_a3/03_online.png` (alarm anindaki dongu orani, kumulatif dagilim; noktali cizgiler 0.5 ve 0.8).

## Uygulanan tanimlar (kare duzeyinde) ve belirsizlik giderimleri

- Zaman = OOF/kare tablosu `t` (oturumun ilk karesinden saniye; OOF ile kare tablosu birebir ayni, maks fark 0). Oran = (t_alarm - t[start_idx]) / dur_s.
- Ziyaret = Faz A debounce'lu etiket kosusu (MIN_BOX=2, MAX_GAP=1); D1/D2 parametreleri ve kalibrasyon skorlari Faz A rolleriyle (egit f+3,f+4; kal f+1,f+2) ayni.
- Literal (on-kayitli) onay: b ziyareti = debounce'lu kosunun MIN_BOX'uncu karesi (s+1); ziyaret sonu: ilk ziyaret-sonrasi kare e. Gecis a->b ve bosluk onay karesinde; dwell_a, e karesinde.
- ->END gecisi ve son dwell: sonraki Box1 ziyaretinin onay karesinde (pred segmentasyonunda bu sonraki dongunun ilk ziyareti = dongu siniri). Dongu orani bu yuzden 1'i biraz asabilir (`oran>1 payi`).
- ToF x tof: sonraki Box1 girisi = dongu sonu (end_idx) sonrasi ilk Box1 ziyareti; oturumun son dongusunde (sonraki giris yok) son olaylar oturumun son karesinde skorlanir (`end_fallback`).
- Alarm = ilk olayda 2*min(p1,p2) <= 0.10 (+1e-9), p = Faz A standart conformal, kosan maksimum D1 (Laplace, START baglami) ve D2 (log(d+0.05) z, min-ref 3). Asim kurali yok.
- Anomali (cycle,f) ciftleri 5 f'de tekrar eder (her f'nin kalibrasyonu farkli); metrikler ciftler uzerinden havuzlanmistir, dongu basina degil.
- 'Normal' satiri: yanlis alarm veren normal dongulerin zamanlamasi; dogal-sapma donguleri ayristirilmamistir (on-kayit yasagi).
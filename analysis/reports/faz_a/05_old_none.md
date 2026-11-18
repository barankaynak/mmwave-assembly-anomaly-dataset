# 05 — Eski boru hatti (CNN-GRU/ToF -> TFT |Δ| + Z-score) protokol altinda (Faz A)

`faz_a/05_old_pipeline.py` tarafindan uretilir; DESIGN.md 'OLD (eski boru hatti)'. Tum sayilar stdout'a da yazilir.

## Uygulama ve protokol
- Durum kaynaklari: ToF oracle (istenip OOF'u/etiketi olmayan atlandi: CNN-GRU).
- **State-change dizisi** (notebook `extract_state_changes_and_features`): oturum boyunca kare etiketindeki her degisim bir kayit (none = 0 dahil); `prev_box_duration_s` = biten durumun suresi; ilk kayit prev_box = 0, sure = 0. Dizi turu = `raw` (debounce UYGULANMAMIS ham kare etiketleri (notebook ile ayni)); tahmin kaynaklarinda dizi 5. kareden baslar (OOF penceresi), ToF'ta 1. kareden.
- **Dongu eslemesi:** `04_cycle_anomaly.py` fonksiyonlari (`build_cycles`, `debounce_labels`; AST ile yuklendi, 04 kosmaz) ile **ayni** dongu sinirlari/anahtarlari; bir gecis, yeni durumunun basladigi karenin icinde bulundugu donguye aittir. Dongu skoru = icindeki gecislerin max'i (04 anahtari: `(state_source, segmentation, session, cycle_idx, start_frame_id)` + `f`).
- **TFT:** pytorch-forecasting 1.8.0; hedef = kutu numarasi (float), encoder 24, tahmin 1, hidden 32, 2 head, dropout 0.1, lr 0.001, grad clip 0.1, max 50 epoch, early-stop patience 10 (min_delta 1e-5), tek nicel (0.5) `QuantileLoss`, `reduce_on_plateau_patience=4`, batch 64; unknown categorical = `prev_box`, unknown real = `prev_box_duration_s`; `GroupNormalizer(softplus, center=False)` (sonlu degilse yedek: `transformation=None`; kullanilan asagida), `add_relative_time_idx/target_scales/encoder_length`. Optimizer: **adam** (pytorch-forecasting varsayilani `ranger` icin `pytorch_optimizer` kurulu degil; Adam notebook'un `ranger`'indan **port farkidir**).
- **Egitim verisi (her f):** s02 **egitim** dongulerinin gecisleri zaman sirasiyla birlestirilir (egitim katlari = `anom_roles(f)['train']`, 2 kat ≈ 20 dongu); bunlarin `inner_frac` = 0.20'si (dongu bazli `fa_common.inner_val_split`) **sona** eklenip ic dogrulama hedefleri olur (encoder baglami egitim satirlarina geri uzanir; kal/test hicbir yerde yok). Erken durdurma + checkpoint secimi yalniz bu ic dogrulama kaybiyla. Tum dizilerde **tek sabit `group_id`** (notebook'taki oturum-basina grup kimlikleri `GroupNormalizer` icin bilinmeyen grup sorunu yaratir).
- **Cikarim:** her oturumun TAM gecis dizisi (gozlenen baglam; komsu katlarin donguleri encoder'a girebilir — DESIGN: etiket sizintisi degil). `time_idx < 24` olan gecislere TFT skoru verilemez (notebook ile ayni): bu gecisler dongu skoruna girmez; hepsi bu gecislerden olusan donguler `old_tft = NaN` alir.
- **Sure Z-skoru** (`compute_duration_table.py` ile ayni mantik): `prev_box` basina mean/std (ddof 1, std >= 1e-6, count >= 3) yalniz **egitim-dongusu** gecislerinden (ic dogrulama donguleri dahil); oturum ilk kaydi skorlanmaz; istatistigi olmayan `prev_box` -> NaN. Dongu skoru = max |Z|.
- Tohumlar: bolme 20261104; TFT `seed_everything(20261104 + 10f)`; ic bolme `20261104 + 100f + (1|2)`.

## TFT egitim ozeti (her segmentasyon × f; veri-acligi)
| state_source | segmentation | f | n_train_cycles | n_inner_val_cycles | n_train_rows | n_val_rows | n_train_samples | n_val_samples | epochs | best_val | normalizer | status | sure_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tof | pred | 0 | 19 | 4 | 90 | 24 | 66 | 24 | 50 | 0.9869 | None | ok | 5.7000 |
| tof | pred | 1 | 20 | 4 | 96 | 24 | 72 | 24 | 50 | 1.0708 | None | ok | 5.8000 |
| tof | pred | 2 | 21 | 4 | 102 | 24 | 78 | 24 | 50 | 0.7805 | None | ok | 5.7000 |
| tof | pred | 3 | 20 | 4 | 96 | 24 | 72 | 24 | 50 | 0.5257 | None | ok | 5.8000 |
| tof | pred | 4 | 20 | 4 | 96 | 24 | 72 | 24 | 50 | 1.0462 | None | ok | 6.1000 |
| tof | tof | 0 | 20 | 4 | 96 | 24 | 72 | 24 | 50 | 0.9292 | None | ok | 5.8000 |
| tof | tof | 1 | 21 | 4 | 102 | 24 | 78 | 24 | 50 | 1.1971 | None | ok | 5.0000 |
| tof | tof | 2 | 21 | 4 | 102 | 24 | 78 | 24 | 38 | 1.1964 | None | ok | 3.8000 |
| tof | tof | 3 | 20 | 4 | 96 | 24 | 72 | 24 | 50 | 0.5484 | None | ok | 4.7000 |
| tof | tof | 4 | 20 | 4 | 96 | 24 | 72 | 24 | 50 | 0.9697 | None | ok | 5.0000 |

Kullanilan normalizer: {'None': 10}. Egitim ornek sayisi: min 66, medyan 72, maks 78; ic-dogrulama ornegi medyan 24. Epoch (erken durdurma) medyan 50/50.

### Z istatistik referanslari (egitim-dongusu gecisleri; prev_box:count)
| state_source | segmentation | prev_box | count_min_over_f | count_max_over_f |
|---|---|---|---|---|
| tof | pred | 0 | 57 | 63 |
| tof | pred | 1 | 19 | 21 |
| tof | pred | 2 | 19 | 21 |
| tof | pred | 5 | 19 | 21 |
| tof | tof | 0 | 60 | 63 |
| tof | tof | 1 | 20 | 21 |
| tof | tof | 2 | 20 | 21 |
| tof | tof | 5 | 20 | 21 |

## TFT trivial tahminciden daha iyi mi? (veri-acligi tanisi)
Test-dongusu gecislerinde (s02 normal, havuzlanmis; TFT skoru olan) TFT MAE'si ile iki trivial tahminci: (a) egitim ortalamasi, (b) **Markov-modu**: egitim-dongusu gecislerinde `prev_box` icin en sik yeni durum. MAE = ortalama |gercek − tahmin| (Denk. 9).
| kaynak | seg | n_gecis | TFT_MAE | egitim_ortalamasi_MAE | Markov_modu_MAE |
|---|---|---|---|---|---|
| ToF oracle | pred | 277 | 0.857 | 1.444 | 0.830 |
| ToF oracle | tof | 283 | 0.954 | 1.444 | 0.830 |

## Gecis duzeyinde '% esik ustu' (eski makale metrigi; esikler sabit: |Δ| > 0.5, |Z| > 1.0)
Normal: s02 **test** dongulerinin gecisleri, 5 f havuzlanmis (her normal gecis, kendi donguyu hic gormeyen bir modelle tam bir kez skorlanir). Anomali oturumlari: her f'nin kendi TFT/istatistigiyle oturumdaki **tum** gecisler; 5 f ortalamasi ± sd. `n` = skoru olan gecis sayisi (TFT ilk 24 gecis icin skor veremez). Yayinlanmis degerler sizintili CNN-GRU tahmin dizisi + normal veride egitim=dogrulama=test ile elde edilmisti; **ToF-oracle satiri ust sinirdir, birebir kiyas degildir**.
| kaynak | seg | senaryo | n_tft | %state>0.5 | n_z | %|Z|>1 | n_or | %OR | yayinlanan state/dur/OR (nan = makalede yok) |
|---|---|---|---|---|---|---|---|---|---|
| ToF oracle | tof | normal | 283 | 50.53 | 306 | 25.16 | 283 | 60.07 | 9.09/21.27/29.34 |
| ToF oracle | tof | seq135 | 301 | 48.04 ± 23.32 | 267 | 21.72 ± 5.57 | 248 | 57.50 ± 14.38 | nan/nan/31.16 |
| ToF oracle | tof | seq145 | 262 | 53.05 ± 17.77 | 236 | 23.98 ± 8.77 | 217 | 62.95 ± 11.79 | nan/nan/75.14 |
| ToF oracle | tof | fast | 172 | 52.09 ± 13.18 | 195 | 55.38 ± 7.03 | 172 | 79.88 ± 11.10 | 25.29/38.89/nan |
| ToF oracle | tof | slow | 289 | 38.62 ± 12.83 | 312 | 43.91 ± 9.08 | 289 | 60.42 ± 13.36 | 52.05/58.1/nan |
| ToF oracle | pred | normal | 277 | 40.07 | 300 | 28.33 | 277 | 55.60 | 9.09/21.27/29.34 |
| ToF oracle | pred | seq135 | 301 | 38.47 ± 18.84 | 267 | 25.17 ± 1.31 | 248 | 58.23 ± 13.38 | nan/nan/31.16 |
| ToF oracle | pred | seq145 | 262 | 42.82 ± 14.27 | 236 | 28.56 ± 2.57 | 217 | 63.59 ± 7.77 | nan/nan/75.14 |
| ToF oracle | pred | fast | 172 | 44.19 ± 9.05 | 195 | 56.92 ± 5.85 | 172 | 78.95 ± 7.44 | 25.29/38.89/nan |
| ToF oracle | pred | slow | 289 | 36.82 ± 5.12 | 312 | 49.55 ± 1.43 | 289 | 67.27 ± 2.47 | 52.05/58.1/nan |

Okuma notu: normal satirlarindaki `%OR`, yayinlanan **29.3**'un egitim-disi (test dongusu) karsiligidir. ToF oracle/tof: state 50.53 %, |Z|>1 25.16 %, OR 60.07 %; ToF oracle/pred: state 40.07 %, |Z|>1 28.33 %, OR 55.60 %. (Tanim geregi normal dagilimda |Z| > 1 ~%32 isaretler.)

## Dongu duzeyi OLD skorlari (04'e girdi; cal + test rolleri)
Dongu skoru = icindeki gecislerin max'i. Asagida yalniz 05'in kendi ozeti (tam AUROC/AUPRC/FAR/tespit tablosu `04_anomaly.md`'de, conformal p-degerleriyle).
| kaynak | seg | skor | normal n | normal NaN | normal medyan | seq135 medyan | seq145 medyan | fast medyan | slow medyan | AUROC seq135 | AUROC seq145 | AUROC fast | AUROC slow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ToF oracle | pred | OLD TFT | 47 | 3 | 4.002 | 3.904 | 3.907 | 1.340 | 4.151 | 0.356 | 0.378 | 0.149 | 0.639 |
| ToF oracle | pred | OLD Z | 50 | 0 | 1.514 | 1.361 | 1.328 | 1.806 | 3.533 | 0.454 | 0.404 | 0.656 | 0.932 |
| ToF oracle | tof | OLD TFT | 48 | 3 | 3.855 | 3.865 | 3.889 | 1.165 | 4.028 | 0.443 | 0.502 | 0.159 | 0.808 |
| ToF oracle | tof | OLD Z | 51 | 0 | 1.481 | 1.139 | 1.168 | 1.841 | 3.283 | 0.433 | 0.372 | 0.670 | 0.907 |

(AUROC burada kat basina hesaplanip ortalanmistir; resmi sonuc `04_anomaly.md`.)

## Yorum (sayilardan)
- ToF oracle/pred: TFT MAE = 0.857; egitim-ortalamasi tahmincisi 1.444; Markov-modu tahmincisi 0.830 -> TFT Markov-modunu **gecemiyor** (n = 277 gecis).
- ToF oracle/tof: TFT MAE = 0.954; egitim-ortalamasi tahmincisi 1.444; Markov-modu tahmincisi 0.830 -> TFT Markov-modunu **gecemiyor** (n = 283 gecis).
- Egitim ornegi sayisi 66-78 (DESIGN'in ~36 tahmini 3 gecis/dongu varsayiyordu; none durumu dahil ham dizide ~6 gecis/dongu oldugundan daha fazla, ama yine de ~70 ornek: TFT icin veri-acligi vardir). Erken durdurma epoch medyani 50/50.
- TFT MAE'si Markov-modundan kotuyse 'state' kanali TFT'nin ogrendigi bir seyden degil, dizideki kutu-numarasi farkindan gelir; sira bilgisi D1 (Markov −log p) ile aynidir.
- Sabit esikler (0.5 / 1.0) conformal esikle degil, makaledeki gibi uygulandi; conformal karsiligi `04_anomaly.md` OLD satirlarindadir.
- Toplam sure 64 s.
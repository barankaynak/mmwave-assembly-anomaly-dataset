# Q2 doğrulama (verifier, tur 1)

## Sonuç
- **Yeniden üretildi mi?** Evet. `tools/remote_run.sh probe/02_tempo_scaling.py` sıfırdan çalıştırıldı (exit=0); stdout ile pull edilen `q2.md` satır satır aynı (diff boş). Çalıştırmadan önce okunan q2.md değerleri (interaction eta^2 0.2863, session 0.6990, segment 0.8147, eps^2 d/T 0.090/0.620/0.156/0.204/0.274/0.449, T oranları 0.752/1.668, n = 50/47/51) tekrar çalıştırmada aynen çıktı (seed sabit). Uyuşmazlık: yok.
- **Karar (H2 KALDI) tutuyor mu?** Evet, ve marjı çok geniş (interaction 0.286 vs eşik 0.06; max eps^2 0.62 vs 0.06).

## Betik incelemesi
Kontrol edilenler ve bulgu:
- Split/leakage/tuning: yok. Hiçbir model eğitilmiyor, rastgele bölme yok; eşikler (satır 25-29) sonuçlardan önce sabit sabitler.
- Loop seçimi (satır 44-91): brief ile uyumlu. Loop 0 ve son loop atılıyor, box_order tam eşleşmesi aranıyor, atılanlar md §7'de raporlanıyor (Fast loop 15 = 1-2-2-2-5, Seq145 loop 17 = 1-4-4-5). Off-by-one yok: T = sonraki loop'un ilk Box1 girişi - bu loop'un Box1 girişi; gap_51 aynı sınırı kullanıyor. Kimlik T = sum(6) max sapma 8.9e-16 s, ihlal 0.
- İstatistik formülleri: partial eta^2 = SS/(SS+SS_resid) (satır 111), Type II ANOVA, KW eps^2 = (H-k+1)/(n-k) (satır 124) brief ile aynı. Sessiz veri düşmesi yok (d<=0 sayısı 0 ve raporlanıyor).
- Şekil/tablo uyumu: q2_table.csv ve md aynı SP/SA nesnelerinden üretiliyor; tutarsızlık görülmedi.

### Bloklayan hata
Yok.

### Küçük/minör
1. `scaling_stats` satır 137: `scen_sess` her senaryo için tek oturum varsayıyor (groupby.first); şimdilik doğru, ileride aynı senaryoda ikinci oturum eklenirse sessizce yanlış olur. Q3 import ederken dikkat.
2. md §6 "eps2 (5 oturum)" sütunu orta segmentlerde (gap_1b, dwell_b, gap_b5) farklı kutuları (2/3/4) karıştırıyor; başlıkta konumsal olduğu belirtilmiş ama bu sütun yorumlanmamalı. Ayrıca `SA` içindeki ANOVA hesaplanıp kullanılmıyor (gereksiz iş, zararsız).
3. ANOVA p-değerleri (1e-227 vb.) loop otokorelasyonu nedeniyle anlamsız; md bunu zaten söylüyor, karar partial eta^2 üzerinden. Doğru.
4. d/T KW'si bir kompozisyon üzerinde (md notu var). dwell_1 eps^2 = 0.090 eşiğin yalnız biraz üstünde; verdict onu gerektirmiyor (gap_1b 0.62 tek başına yeter). Yani "her segment eşiği aşıyor" ifadesi doğru ama dwell_1 sınırda.
5. Elasticity 3 noktalı log-log eğim (serbestlik derecesi 1); md betimleyici olarak sunuyor, çıkarım olarak kullanılmamalı. Oturum-içi eğimler part-whole yanlılığı taşıyor (md'de belirtilmiş).
6. "Sonuçlara bakmadan önce sabitlendi" iddiası betikten doğrulanamaz (yalnız sabitlerin yeri üstte); kabul edilebilir.

## Knob testi (min_box_s 0.1 -> 0.2)
`tof_segments(..., min_box_s=0.2)` ile kopya çalıştırıldı (çıktılar ayrı dosyaya, sonra silindi; orijinaller dokunulmadı, yeniden üretim zaten orijinal betikle yapıldı).
- Fast loop sayısı 47 -> 43 (kısa Box dwell'leri 0.2 s altında kalan 4 loop düştü; Normal 50, Slow 51 aynı). Çıkan Fast loop'larda "1-2" (tek kutu eksik) örüntüleri görüldü: Fast dwell'ler kısa olduğundan bu eşik bazı gerçek ziyaretleri siliyor.
- interaction partial eta^2 0.2863 -> 0.2947; session 0.6990 -> 0.7009; max eps^2 (d/T) 0.6195 -> 0.6329; 10 Hz çapraz kontrol 0.2619 -> 0.2677.
- Karar: **H2 KALDI** değişmedi. Kimlik yine 0 ihlal.
Kare-tabanlı (10 Hz) kaynak zaten md §5'te raporlu (0.2619 / 0.5794) ve aynı kararı veriyor.

## Değerlendirme
Karar sağlam: hiçbir tek knob onu çevirmiyor, eşik aşımı 5x mertebesinde. Segmentler orantısız ölçekleniyor; en belirgin sapmalar Slow'da gap_1b (indeks 1.47) ve gap_51 (1.35), Fast'te dwell_5 (0.80) ve dwell_1 (1.20). Q3 H3 skoru için md'nin önerdiği segment-bazlı normalizasyon yönü verilerle tutarlı.

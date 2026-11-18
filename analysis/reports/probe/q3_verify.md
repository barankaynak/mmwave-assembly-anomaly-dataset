# Q3 doğrulama (adversarial verifier, tur 1)

## Sonuç özeti
- **Yeniden üretildi: EVET.** `tools/remote_run.sh probe/03_few_shot.py` sıfırdan çalıştırıldı (exit=0). Üretilen `q3.md` ve `q3_table.csv` orijinallerle **bayt-bayt aynı** (diff boş); stdout, `q3.md` satırlarıyla birebir aynı (tek fark harness'ın "done 0" satırı). Eşleşmeyen sayı yok.
- Q1 tutarlılık kontrolü: `q3.md` içindeki referans (tüm s02 kareleri, 2B q=0.1 → K=6 NMI=0.876; 1B q=0.2 → K=3 NMI=0.912) `q1.md` satır 236/255 ile aynı; s01'de seçilen quantile'lar (0.1 / 0.2) `q1.md` satır 155 ile aynı.
- **Bloklayıcı hata yok.** Verdict (birincil kural n ≤ 20 ızgarasında karşılanmadı, N = None) ayakta.

## Kod incelemesi (probe/03_few_shot.py)
Kontrol edilenler ve sonuç:
- Bölmeler loop_id bazlı; rastgele kare/pencere bölmesi yok. `windows()` (satır 79-86) ardışık loop_id şartını doğru uyguluyor; `eval_ids()` (89-91) guard'ı doğru uyguluyor (assert'ler satır 172-173 ile ayrıca doğrulanıyor). Off-by-one yok.
- Quantile s01'de Q1 koduyla türetiliyor (`chosen_quantiles`), s02/s05/s06'da parametre ayarı yok. Kalibrasyon yalnız s02; s05/s06 yalnız test. Sızıntı yok.
- Kriterler dosya başında sabit (satır 36-46); sonuç sonrası ayarlanmış görünmüyor. Sonradan eklenen ölçümler (t-GA, en kötü pencere, ekstrapolasyon) "duyarlılık/betimleyici" diye etiketli ve CLAIM'i değiştirmiyor.
- NMI/ARI Q1'in `evaluate_assignment` fonksiyonundan geliyor (kopyalanmamış). Bootstrap formülü (satır 125-135) ve t-GA (138-142) doğru. `dT_np[[pos[l] ...]]` indeksleme doğru (ids ve pivot ikisi de sıralı).
- Sessiz veri düşmesi: s02'de 50/50 loop kalıyor (dışlanan yok); s05/s06 pencereleri kendi kept loop'larından. Q2 kuralları `Q2.loop_selection` ile aynen kullanılıyor.
- Şekil/tablo uyumu: `q3_curves.png` panelleri tablolarla tutarlı (örn. n=20 gap_b5 0.0224 eşik çizgisinin üstünde; başlık "N = not reached in grid").

## Küçük (minor) bulgular — sayıyı/verdict'i değiştirmez
1. **(a) kriteri fiilen boş**: n=20 için fark referansın kendisi olduğundan tanım gereği 0 (satır 229-232); ayrıca n arttıkça değerlendirme kümesi küçülüyor/değişiyor (28 vs 47 loop), yani n'ler arası doğruluk karşılaştırması aynı eval kümesinde değil. Tüm n'lerde (a) geçtiği için CLAIM pratikte yalnız (b)'ye indirgeniyor; rapor bunu "(a) tek başına geçen n: [1,3,5,10,20]" ile zaten gösteriyor ama yorumda vurgulanabilir.
2. **Bootstrap GA iid varsayıyor, kayma (drift) görmüyor**: 3e'de gap_1b için n=20'de kapsama 0.548 (nominal 0.95). §6 "Kapsama" cümlesi segment ortalamasını (0.92) verip "n=20'de yapay yüksek" diyor; bu gap_1b'deki zayıf kapsamayı gizliyor. Sayılar doğru, yorum cümlesi eksik/iyimser.
3. **K_found(n) eğrisi bandwidth tahmincisiyle karışık**: `estimate_bandwidth(quantile=q)` örneklem boyutuna bağlı (n=1'de ~44 kare → ~4 komşu); küçük n'de aşırı bölünmenin bir kısmı veri azlığından değil, quantile-bandwidth kuralının küçük örneklemdeki davranışından geliyor. Q3 metninde bu karışıklık belirtilmemiş.
4. Pencereler örtüşüyor ve kenardaki loop'lar daha az pencereye giriyor (3f'de küçük yanlılık, örn. gap_1b 0.1205 → 0.1178); metin örtüşmeyi söylüyor, yanlılığı yalnız 3f tablosu gösteriyor.
5. N verilmediği için "calibrates in N cycles" cümlesi çıkmıyor; ekstrapolasyon (n≈21.4, ~107 s) betimleyici ve normal yaklaşımla; test edilmiş değil. Verdict ızgara sınırıyla sınırlı (marj: 0.0224 vs 0.02, ~%12).
6. Şekil panel (a): ±2 pp bandı y eksenini domine ediyor, eğri düz görünüyor; kozmetik.

## Knob testi
`probe/_verify_q3.py` kopyaları (orijinale dokunulmadı), remote_run ile çalıştırıldı:
- **GUARD = 0**: s02 doğruluk farkları −0.19/−0.15/−0.14/−0.11/0 puan (orijinal −0.21/−0.17/−0.16/−0.12/0); NMI/K değerleri ±0.001-0.002 oynadı; (b) hiç değişmedi (guard bootstrap GA'ya girmez). **N = None, verdict aynı.** Not: guard, (b)'yi yapısal olarak etkilemediği için bu knob verdict için zayıf bir sınama; guard'ın pratik etkisi ihmal edilebilir (≤0.03 puan).
- **Bootstrap seed 20261104 → 777**: n=20 max GA 0.0226 (orijinal 0.0224), n=10 0.0286 (aynı), n=5 0.0347 (0.0348), n=3 0.0389 (0.0391). **N = None, verdict aynı**; seed varyansı (~%1) eşik marjından (~%12) çok küçük.
- Temizlik: `_verify_q3.py` silindi; verify koşuları `reports/probe/q3.*` dosyalarını ezdiği için orijinal script yeniden çalıştırıldı ve `q3.md` / `q3_table.csv` orijinal kopyalarla byte-byte aynı doğrulandı (`q3_curves.png` de yeniden üretildi, içerik aynı çıktıdan).

## Değerlendirme
Sayılar yeniden üretiliyor, yöntem brief'e uygun, sızıntı/test-set ayarlaması yok. Verdict ("n ≤ 20'de kural karşılanmadı; yapı 1 döngüde oturuyor, timing ~20+ döngü ister; darboğaz gap_b5'in mutlak 0.02 eşiği") savunulabilir. Önerilen (zorunlu olmayan) düzeltmeler: §6 kapsama cümlesine gap_1b'nin 0.548'ini ekle; K_found(n) için bandwidth/örneklem karışıklığını bir cümleyle not et; N'nin ızgara sınırlı olduğunu ve ekstrapolasyonun test edilmediğini abstract/özetlerde tekrarla.

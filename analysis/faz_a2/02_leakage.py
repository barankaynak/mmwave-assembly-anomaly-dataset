"""Faz A2 / A2.2 (c) - Sizinti bir oruntu mu? S0 (rastgele pencere bolmesi, 5 tohum) vs S1 (grup-bazli 5 kat), B1 / B2 / CNN-GRU.
Tum metrikler OOF dosyalarindan fa_common.state_metrics ile yeniden hesaplanir (tek kaynak).
Etki = ortalama(S0 tohumlari) - ortalama(S1 katlari); %95 GA = Welch t araligi (n=5 vs n=5).
Cikti: reports/faz_a2/02_leakage.md (+ 02_leakage_per_run.csv, 02_leakage_summary.csv)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
import numpy as np
import pandas as pd
from scipy import stats
import a2_s0 as s0

fc = s0.fc
OUT2 = s0.OUT2
MODELS = [("b1", "B1 en-yakin-merkez"), ("b2", "B2 el-yapimi ozellik + RF"), ("cnngru", "CNN-GRU")]
METRICS = [("acc", "accuracy"), ("macro_f1", "macro-F1")]
LINES = []


def out(s=""):
    LINES.append(s)
    print(s, flush=True)


def s0_file(model, seed):
    if model == "cnngru" and seed == 42:
        return fc.oof_path("cnngru", "S0", 0)                 # Faz A dosyasi (aynen)
    return s0.oof2_path(model, seed)


def metr(df):
    return fc.state_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())


W = fc.windows()
rows = []
status = {}
for model, _ in MODELS:
    # ---- S1 katlari (Faz A OOF, salt okunur)
    for f in range(fc.N_FOLDS):
        p = fc.oof_path(model, "S1", f)
        df = pd.read_csv(p)
        m = metr(df)
        rows.append({"model": model, "split": "S1", "run": f, "n": m["n"], "acc": m["acc"], "macro_f1": m["macro_f1"]})
    # ---- S0 tohumlari
    for seed in s0.SEEDS:
        p = s0_file(model, seed)
        if not p.exists():
            status[(model, seed)] = "EKSIK"
            continue
        df = pd.read_csv(p)
        te = s0.s0_test_idx(W, seed)
        exp = W.iloc[te][["session", "frame_id"]].reset_index(drop=True)
        got = df[["session", "frame_id"]].reset_index(drop=True)
        assert exp.equals(got), f"{p.name}: OOF test pencereleri tohum {seed} bolmesiyle ayni degil"
        assert (df["y_true"].to_numpy() == W.iloc[te]["y_true"].to_numpy()).all()
        m = metr(df)
        rows.append({"model": model, "split": "S0", "run": seed, "n": m["n"], "acc": m["acc"], "macro_f1": m["macro_f1"]})
        status[(model, seed)] = "ok"
R = pd.DataFrame(rows)
R.to_csv(OUT2 / "02_leakage_per_run.csv", index=False)

# ---- CNN-GRU tohum 42 capraz kontrol (Faz A kaydi)
r42 = R[(R.model == "cnngru") & (R.split == "S0") & (R.run == 42)]
if len(r42):
    a, f = float(r42["acc"].iloc[0]), float(r42["macro_f1"].iloc[0])
    assert abs(a - 0.90092) < 6e-6 and abs(f - 0.87807) < 6e-6, (a, f)
    xchk = f"CNN-GRU S0 tohum 42 yeniden hesap: acc {a:.5f} / macro-F1 {f:.5f} = Faz A cnn_runs.csv (0.90092 / 0.87807) -> ESLESTI"
else:
    xchk = "CNN-GRU tohum 42 bulunamadi"


def welch(x0, x1, level=0.95):
    n0, n1 = len(x0), len(x1)
    m0, m1 = np.mean(x0), np.mean(x1)
    v0, v1 = np.var(x0, ddof=1), np.var(x1, ddof=1)
    se = np.sqrt(v0 / n0 + v1 / n1)
    df = (v0 / n0 + v1 / n1) ** 2 / ((v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1))
    t = stats.t.ppf(0.5 + level / 2, df)
    d = m0 - m1
    tstat = d / se
    p = 2 * stats.t.sf(abs(tstat), df)
    return dict(m0=m0, s0=np.sqrt(v0), n0=n0, m1=m1, s1=np.sqrt(v1), n1=n1, d=d, se=se, df=df, lo=d - t * se, hi=d + t * se, t=tstat, p=p)


S = []
for model, label in MODELS:
    for mk, mlabel in METRICS:
        x0 = R[(R.model == model) & (R.split == "S0")][mk].to_numpy()
        x1 = R[(R.model == model) & (R.split == "S1")][mk].to_numpy()
        if len(x0) < 2:
            continue
        w = welch(x0, x1)
        w.update(model=model, metric=mk)
        S.append(w)
SM = pd.DataFrame(S)
SM.to_csv(OUT2 / "02_leakage_summary.csv", index=False)

# ---- Faz A dosyalari degismedi mi?
def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


md5_note = "snapshot dosyasi yok"
snap = Path(__file__).resolve().parent / "fazA_md5_before.txt"
if snap.exists():
    bad, n = [], 0
    for ln in snap.read_text().splitlines():
        h, rel = ln.split(None, 1)
        rel = rel.strip().lstrip("*")
        n += 1
        if md5(ROOT / rel) != h:
            bad.append(rel)
    md5_note = (f"Faz A OOF dosyalari + cnn_runs.csv + split_s0.csv: {n} dosyanin md5'i is oncesi anlik goruntuyle "
                + ("AYNI (hicbiri degismedi)" if not bad else f"**FARKLI: {bad}**"))
    assert not bad, bad

# ----------------------------------------------------------------------------- rapor
out("# Faz A2 / A2.2 — Sızıntı bir örüntü mü? (S0 rastgele pencere bölmesi vs S1 grup-bazlı katlar)")
out()
out("Üreten: `faz_a2/02_leakage.py` (B1/B2 S0: `11_baselines_s0_seeds.py`; CNN-GRU S0 tohum 1–4: `12_cnn_s0_seeds.py`). "
    "Önceden kayıtlı: `faz_a2/DESIGN_A2.md` §A2.2. Bu bir **ikincil** analizdir; ana model seçimini (B2 + D3) değiştirmez.")
out()
out("## Protokol")
out("- **S0** = sınıf-tabakalı rastgele %80/20 pencere bölmesi (`train_test_split(test_size=0.2, stratify=y_true, random_state=tohum)`), "
    "tohumlar **42 (Faz A) ve 1, 2, 3, 4**; her tohumda 12.998 eğitim / 3.250 test penceresi. "
    "Tohum 42 bölmesi Faz A `split_s0.csv` ile birebir aynı (betikte assert).")
out("- **S1** = Faz A grup-bazlı 5 kat (`StratifiedGroupKFold`, grup = (oturum, döngü)); S1 değerleri `reports/faz_a/oof/` içindeki "
    "Faz A OOF dosyalarından **kat başına** yeniden hesaplandı (havuzlanmadı). Kat başına n = 3.223–3.300.")
out("- Her bölmede B1/B2/CNN-GRU aynı hiperparametre ve aynı etiketlerle çalışır. Metrikler `fa_common.state_metrics` (accuracy; macro-F1 = test kümesinde bulunan sınıflar üzerinden).")
out("- **Tohum şeması** (DESIGN_A2'de açık bırakılan ayrıntı, burada sabitlendi ve üç modele aynı uygulandı): bölme `random_state` = tohum; "
    "iç doğrulama (CNN erken durdurma ve B1 τ seçimi) = eğitim pencerelerinden sınıf-tabakalı rastgele %10, `random_state` = tohum "
    "(01_state_cnn.py'nin S0 dalıyla aynı); model tohumu (torch başlangıç / RF `random_state`) = 20261104 + tohum. "
    "**İstisna:** CNN-GRU tohum 42, Faz A'dan olduğu gibi alındı (model tohumu 20261104; yeniden eğitilmedi). "
    "B1/B2'nin S0 sonuçları Faz A'da yoktu; 5 tohumun tümü bu aşamada yeni koşuldu (`02_state_baselines.py` kodu birebir, yalnız bölme farklı; "
    "B2 RF 300 ağaç, `class_weight=balanced`, n_jobs=3; B1 τ'yu iç doğrulama macro-F1'iyle seçer).")
out("- **Etki** = ortalama(S0, 5 tohum) − ortalama(S1, 5 kat), accuracy ve macro-F1 için; %95 GA = Welch t aralığı: "
    "d ± t(0,975; df)·√(s₀²/5 + s₁²/5), df Welch–Satterthwaite. Puan = yüzde puan.")
out("- **Donanım notu:** S1 CNN-GRU eğitimi uzak RTX 3090'da; yeni S0 tohumları (CNN-GRU 1–4) yerel GTX 1650 SUPER'de koştu; CNN-GRU S0 tohum 42 Faz A'dan (uzak 3090). "
    "GPU/cuDNN belirlenimsizliği ve PyTorch sürüm farkı tam tekrarlanabilirliği engeller; B1/B2 CPU'da deterministiktir.")
out()
out("## Çapraz kontroller")
out(f"- {xchk}")
out(f"- S0 OOF dosyalarının test pencereleri (oturum, kare) ilgili tohumun bölmesiyle birebir aynı (her dosya için assert).")
out(f"- {md5_note}.")
miss = [k for k, v in status.items() if v != "ok"]
out(f"- Eksik S0 koşuları: {miss if miss else 'yok (3 model × 5 tohum = 15 OOF dosyası mevcut)'}")
out()

out("## Özet: S0 vs S1 ve sızıntı etkisi")
out()
out("| model | metrik | S0 ort ± sd (n=5 tohum) | S1 ort ± sd (n=5 kat) | etki (puan) | %95 GA (puan) | Welch df | t | p |")
out("|---|---|---|---|---|---|---|---|---|")
lab = dict(MODELS)
mlab = dict(METRICS)
for _, r in SM.iterrows():
    out(f"| {lab[r.model]} | {mlab[r.metric]} | {100*r.m0:.2f} ± {100*r.s0:.2f} | {100*r.m1:.2f} ± {100*r.s1:.2f} | "
        f"**{100*r.d:+.2f}** | [{100*r.lo:+.2f}, {100*r.hi:+.2f}] | {r.df:.1f} | {r.t:.2f} | {r.p:.2g} |")
out()
out("Makale cümlesi için (DESIGN_A2'ye göre; hesaplanan sayılarla; etki işareti ve GA'ya göre ifade seçilmelidir):")
for _, r in SM.iterrows():
    verdict = "GA 0'ı dışlıyor" if (r.lo > 0 or r.hi < 0) else "GA 0'ı içeriyor (etki belirgin değil)"
    out(f"- {lab[r.model]}, {mlab[r.metric]}: rastgele pencere bölmesi − grup-bazlı bölme = **{100*r.d:+.2f}** puan "
        f"(Welch SE {100*r.se:.2f}; %95 GA [{100*r.lo:+.2f}, {100*r.hi:+.2f}]; {verdict}).")
out()
out("**Yorum notları (veriyle sınırlı):**")
out("- Aralıklar yaklaşıktır: S0 tohumlarının test kümeleri aynı 16.248 pencereden çekilir (örtüşür) ve S1 katlarının eğitim kümeleri birbiriyle paylaşılır; "
    "tohumlar/katlar bağımsız örnekler değildir. S0 sd'si bölme-tohumu + model-tohumu varyansını, S1 sd'si ise farklı grupların (döngülerin) zorluk farkını da içerir; ikisi aynı nicelik değildir.")
out("- n=5 vs n=5 ile df küçüktür; GA'lar t dağılımıyla (normal değil) hesaplandı.")
c = R[(R.model == "cnngru") & (R.split == "S0")].set_index("run")["acc"]
if {1, 2, 3, 4, 42} <= set(c.index):
    out(f"- CNN-GRU karıştırıcısı: S0 tohum 42 (uzak 3090, Faz A) accuracy {100*c[42]:.2f}; yerel 1650S'teki tohum 1–4 {100*c[[1,2,3,4]].min():.2f}–{100*c[[1,2,3,4]].max():.2f} "
        f"(ort {100*c[[1,2,3,4]].mean():.2f}). Tek bir 3090 koşusuyla donanım/model-tohumu etkisi ayrıştırılamaz; 5-tohum ortalaması 4 yerel + 1 uzak koşunun karışımıdır. "
        "Tohum 42-yalnız (aynı donanım) etki en alttaki ekte ayrıca verilmiştir.")
out("- B1/B2 için fark küçük ve GA 0'ı içeriyor; CNN-GRU için GA 0'ı dışlıyor. Bu, gözlenen sızıntı farkının bu veri ve bu üç model için derin modelde göründüğünü, "
    "düşük kapasiteli/elle çıkarılmış özellikli modellerde (B1: tek kare centroid; B2: 21 özellik + RF) ayırt edilemediğini söyler; nedeni (kapasite, girdi türü) bu deneyle ayrıştırılmadı.")
out()

out("## Tohum başına S0 sonuçları")
out()
out("| model | tohum | n_test | accuracy | macro-F1 | kaynak |")
out("|---|---|---|---|---|---|")
for model, _ in MODELS:
    for seed in s0.SEEDS:
        r = R[(R.model == model) & (R.split == "S0") & (R.run == seed)]
        if not len(r):
            out(f"| {lab[model]} | {seed} | - | EKSIK | EKSIK | - |")
            continue
        r = r.iloc[0]
        src = "Faz A (`cnngru__S0__f0`, uzak 3090)" if (model == "cnngru" and seed == 42) else \
              ("yerel GTX 1650S, bu aşama" if model == "cnngru" else "yerel CPU, bu aşama")
        out(f"| {lab[model]} | {seed} | {int(r.n)} | {r.acc:.4f} | {r.macro_f1:.4f} | {src} |")
out()
out("## Kat başına S1 sonuçları (Faz A OOF, kat başına yeniden hesap)")
out()
out("| model | kat | n_test | accuracy | macro-F1 |")
out("|---|---|---|---|---|")
for model, _ in MODELS:
    for f in range(fc.N_FOLDS):
        r = R[(R.model == model) & (R.split == "S1") & (R.run == f)].iloc[0]
        out(f"| {lab[model]} | {f} | {int(r.n)} | {r.acc:.4f} | {r.macro_f1:.4f} |")
out()

# ---- CNN-GRU eğitim kayıtları
runs_p = OUT2 / "12_cnn_s0_runs.csv"
out("## CNN-GRU S0 eğitim kayıtları (cihaz / epoch)")
out()
out("| koşu | cihaz | epoch (best) | best val loss | süre (s) | n_train / n_val / n_test | model tohumu |")
out("|---|---|---|---|---|---|---|")
fa_runs = pd.read_csv(ROOT / "reports" / "faz_a" / "cnn_runs.csv")
r0 = fa_runs[(fa_runs.model == "cnngru") & (fa_runs.split == "S0")].iloc[0]
out(f"| S0 tohum 42 (Faz A) | {r0.device} (uzak RTX 3090) | {int(r0.epochs_run)} ({int(r0.best_epoch)}) | {r0.best_val_loss:.5f} | {r0.wall_s:.0f} | "
    f"{int(r0.n_train)} / {int(r0.n_val)} / {int(r0.n_test)} | 20261104 |")
if runs_p.exists():
    rr = pd.read_csv(runs_p).sort_values("seed")
    for _, r in rr.iterrows():
        out(f"| S0 tohum {int(r.seed)} (A2.2) | {r.device} (yerel GTX 1650 SUPER) | {int(r.epochs_run)} ({int(r.best_epoch)}) | {r.best_val_loss:.5f} | {r.wall_s:.0f} | "
            f"{int(r.n_train)} / {int(r.n_val)} / {int(r.n_test)} | {int(r.model_seed)} |")
    cpu = rr[rr.device.astype(str).str.contains("cpu")]
    out()
    out(f"CPU'ya düşen tohum: {cpu['seed'].tolist() if len(cpu) else 'yok (tümü CUDA)'}.")
out()
out("## Sınırlamalar")
out("- CNN-GRU S1 sayıları uzak 3090'da, S0 tohum 1–4 yerel 1650S'te üretildi; donanım/belirlenimsizlik farkı etkiye karışabilir (tohum 42 S0 ile S1 aynı donanımdadır; aşağıda tohum 42-yalnız etki de verilir).")
out("- S0 ve S1 aynı pencere kümesini paylaşır ama test kümeleri farklıdır (S0 rastgele %20; S1 katları ayrık, grup-bazlı). S0'da ardışık pencereler (4/5 kare örtüşmesi) eğitim ve testte birlikte bulunur — sızıntı mekanizması budur; etki bu mekanizmanın toplam ölçüsüdür, ayrıştırılmış bir nedensel kestirim değildir.")
out()
# tohum-42-yalniz etki (ayni donanim, CNN)
out("### Ek: tohum 42 S0 ile S1 katları (CNN-GRU için aynı donanım: uzak 3090)")
for mk, ml in METRICS:
    v0 = R[(R.model == "cnngru") & (R.split == "S0") & (R.run == 42)][mk]
    v1 = R[(R.model == "cnngru") & (R.split == "S1")][mk]
    if len(v0):
        out(f"- CNN-GRU {ml}: S0(42) {100*v0.iloc[0]:.2f} − S1 ortalama {100*v1.mean():.2f} = **{100*(v0.iloc[0]-v1.mean()):+.2f}** puan (tek koşu; S1 kat sd {100*v1.std(ddof=1):.2f}).")
out()
(OUT2 / "02_leakage.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")
print("yazildi:", OUT2 / "02_leakage.md")

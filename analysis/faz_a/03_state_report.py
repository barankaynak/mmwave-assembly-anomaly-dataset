"""Faz A / 03 - durum tahmini raporu: tum mevcut OOF dosyalarini okur (cnngru/cnnonly/b1/b2 x S1/S0/S2)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probe"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fa_common as fc
from common import md_table

ft = fc.frame_table()[["session", "frame_id", "near_change"]]
NAMES = {"cnngru": "CNN-GRU", "cnnonly": "CNN-only", "b1": "B1 en-yakin-merkez", "b2": "B2 el-yapimi+RF"}
SUBS = [("(a) kutu", lambda d: d.y_true.to_numpy() > 0),
        ("(b) none", lambda d: d.y_true.to_numpy() == 0),
        ("(c) gecis", lambda d: d.near_change.to_numpy())]
CLS = ["none", "B1", "B2", "B3", "B4", "B5"]
L = []


def out(s=""):
    L.append(s)


def sd(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(a.std(ddof=1)) if len(a) > 1 else 0.0


def ms(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return f"{a.mean():.4f} ± {sd(a):.4f}" if len(a) else "nan"


def fold_f1(d):
    return d.groupby("fold")[["y_true", "y_pred"]].apply(lambda g: fc.state_metrics(g.y_true, g.y_pred)["macro_f1"])


data = {}
for sp in fc.SPLITS:
    for m in fc.MODELS:
        d = fc.load_oof(m, sp)
        if len(d):
            d = d.merge(ft, on=["session", "frame_id"], how="left", validate="m:1")
            assert d.near_change.notna().all()
            d["near_change"] = d["near_change"].astype(bool)
            data[(m, sp)] = d
print("mevcut OOF:", {f"{m}/{s}": (len(d), d.fold.nunique()) for (m, s), d in data.items()})

out("# 03 — Durum tahmini raporu (Faz A)")
out()
out("`faz_a/03_state_report.py` tarafindan uretilir; DESIGN.md protokolu. Mevcut OOF dosyalari: " +
    (", ".join(f"{NAMES[m]}/{s} ({len(d)} pencere, {d.fold.nunique()} kat)" for (m, s), d in data.items()) or "yok") + ".")
out("Etiket = ±3 kare mod yumusatmali ToF etiketi (y_true). Metrikler OOF tahminler uzerinden kat bazli hesaplanir; ortalama ± sd (ddof=1) kat sayisi uzerinden.")
out("macro-F1 yalniz y_true'da bulunan siniflar uzerinden (zero_division=0).")
rows_all = []
for sp, title in (("S1", "S1 — 5 katli StratifiedGroupKFold (ana)"), ("S2", "S2 — leave-one-session-out (6 kat)")):
    mods = [m for m in fc.MODELS if (m, sp) in data]
    if not mods:
        continue
    out()
    out(f"## {title}")
    out()
    out("### Genel (kat ortalamasi ± sd)")
    t = []
    for m in mods:
        d = data[(m, sp)]
        ks = [fc.state_metrics(g.y_true, g.y_pred) for _, g in d.groupby("fold")]
        r = {"model": NAMES[m], "kat": len(ks), "n": len(d), "accuracy": ms([k["acc"] for k in ks]),
             "macro-F1": ms([k["macro_f1"] for k in ks])}
        for c in range(6):
            r[f"F1 {CLS[c]}"] = ms([k[f"f1_{c}"] for k in ks])
        t.append(r)
        rows_all.append({"split": sp, "model": m, "subset": "tum",
                         "acc_mean": np.mean([k["acc"] for k in ks]), "acc_sd": sd([k["acc"] for k in ks]),
                         "macro_f1_mean": np.mean([k["macro_f1"] for k in ks]), "macro_f1_sd": sd([k["macro_f1"] for k in ks])})
    out(md_table(pd.DataFrame(t)))
    print(sp, "genel:", [(r["model"], r["accuracy"], r["macro-F1"]) for r in t])
    out()
    out("### Alt kumeler: (a) kutu kareleri, (b) none kareleri, (c) gecis kareleri (etiket degisimine ±2 kare)")
    out("(b) icin accuracy = none sinifinin recall'u. macro-F1 alt kumedeki mevcut siniflar uzerinden. Kat basina hesap, ortalama ± sd.")
    t = []
    for m in mods:
        d = data[(m, sp)]
        r = {"model": NAMES[m]}
        for nm, fn in SUBS:
            acc, mf = [], []
            for _, g in d.groupby("fold"):
                s = g[fn(g)]
                if len(s):
                    k = fc.state_metrics(s.y_true, s.y_pred)
                    acc.append(k["acc"])
                    mf.append(k["macro_f1"])
            r[f"{nm} n"] = int(fn(d).sum())
            r[f"{nm} acc"] = ms(acc)
            if nm != "(b) none":
                r[f"{nm} mF1"] = ms(mf)
            rows_all.append({"split": sp, "model": m, "subset": nm, "acc_mean": np.mean(acc), "acc_sd": sd(acc),
                             "macro_f1_mean": np.mean(mf) if mf else np.nan, "macro_f1_sd": sd(mf) if mf else np.nan})
        t.append(r)
        print(sp, m, {k: v for k, v in r.items() if "acc" in k})
    out(md_table(pd.DataFrame(t)))
    out()
    out("### (d) Oturum basina (OOF havuzlanmis; s01 = kutular cozunurluk altinda)")
    sess = sorted(data[(mods[0], sp)].session.unique())
    t = []
    for s in sess:
        r = {"session": s[:10] + (" **(s01)**" if s.startswith("session_01") else "")}
        for m in mods:
            g = data[(m, sp)]
            g = g[g.session == s]
            if len(g):
                k = fc.state_metrics(g.y_true, g.y_pred)
                r[f"{NAMES[m]} acc"], r[f"{NAMES[m]} mF1"] = k["acc"], k["macro_f1"]
            else:
                r[f"{NAMES[m]} acc"] = r[f"{NAMES[m]} mF1"] = np.nan
            rows_all.append({"split": sp, "model": m, "subset": f"session:{s}",
                             "acc_mean": r[f"{NAMES[m]} acc"], "macro_f1_mean": r[f"{NAMES[m]} mF1"]})
        r["n"] = int((data[(mods[0], sp)].session == s).sum())
        t.append(r)
    out(md_table(pd.DataFrame(t)))
    if "cnngru" in mods and len(mods) > 1:
        out()
        out("**Eslesik kat farki (macro-F1; fark = CNN-GRU − diger):**")
        a = fold_f1(data[("cnngru", sp)])
        for m in mods:
            if m == "cnngru":
                continue
            dd = (a - fold_f1(data[(m, sp)])).dropna()
            out(f"- {NAMES[m]}: fark {dd.mean():+.4f} (sd {sd(dd):.4f}, {len(dd)} kat). DESIGN sart 1: fark <= 0.02 ise taban 'yakin'.")
            print(sp, "fark cnngru -", m, f"{dd.mean():+.4f}")

out()
out("## Sizinti etkisi: S0 (rastgele pencere bolmesi) vs S1 (grup bazli) — CNN-GRU")
if ("cnngru", "S0") in data and ("cnngru", "S1") in data:
    d0, d1 = data[("cnngru", "S0")], data[("cnngru", "S1")]
    k0 = fc.state_metrics(d0.y_true, d0.y_pred)
    k1 = [fc.state_metrics(g.y_true, g.y_pred) for _, g in d1.groupby("fold")]
    s1a, s1f = np.mean([k["acc"] for k in k1]), np.mean([k["macro_f1"] for k in k1])
    out(md_table(pd.DataFrame([
        {"kosul": "Keras (eski makale, S0, 3250 test)", "accuracy": 0.9366, "macro-F1": np.nan},
        {"kosul": f"PyTorch port S0 (n={k0['n']})", "accuracy": k0["acc"], "macro-F1": k0["macro_f1"]},
        {"kosul": "PyTorch port S1 (kat ort.)", "accuracy": s1a, "macro-F1": s1f}]), floatfmt="{:.4f}"))
    out()
    out(f"**Sizinti etkisi (S0_pt − S1_pt): accuracy {k0['acc'] - s1a:+.4f}, macro-F1 {k0['macro_f1'] - s1f:+.4f}.**")
    out(f"Port farki (S0_pt − Keras 0.9366): accuracy {k0['acc'] - 0.9366:+.4f}.")
    print("S0_pt", k0["acc"], k0["macro_f1"], "S1_pt", s1a, s1f)
else:
    out("CNN-GRU S0 veya S1 OOF dosyasi henuz yok; CNN isi bitince bu rapor yeniden calistirilacak.")
out()
out("Port farklari (DESIGN): Conv→BN→ReLU sirasi; Keras GRU dropout 0.3/recurrent 0.2 yerine GRU cikisinda dropout 0.3; kayitli 89.4 vs 93.7 ablasyon farki esit-epoch (40 vs 120) karsilastirmadan kaynaklanir.")

out()
out("## Parametre sayisi ve CPU cikarim suresi (pencere basina)")
t = []
bj = fc.OUT / "baseline_runs.json"
if bj.exists():
    b = json.load(open(bj))
    t.append({"model": NAMES["b1"], "parametre": "5 merkez (10 deger) + tau", "CPU ms/pencere": f"{b['b1_ms']:.4f}"})
    t.append({"model": NAMES["b2"], "parametre": f"RF {b['b2_n_trees']} agac, {b['b2_n_features']} ozellik",
              "CPU ms/pencere": f"{b['b2_ms']:.3f} (tek is parcacigi, ozellik cikarimi dahil)"})
    print("latency B1/B2 ms:", b["b1_ms"], b["b2_ms"])
if t:
    out(md_table(pd.DataFrame(t)))
cr = fc.OUT / "cnn_runs.csv"
out()
if cr.exists():
    c = pd.read_csv(cr)
    print("cnn_runs.csv kolonlar:", list(c.columns))
    out("cnn_runs.csv (ham):")
    out()
    out(md_table(c.head(40)))
else:
    out("(cnn_runs.csv henuz yok; CNN parametre sayisi/gecikmesi CNN isi bitince eklenecek.)")

(fc.OUT / "03_state.md").write_text("\n".join(L) + "\n", encoding="utf-8")
pd.DataFrame(rows_all).to_csv(fc.OUT / "03_state_table.csv", index=False)
print("yazildi 03_state.md,", len(rows_all), "satir")

"""Faz A4 / A4.3 (M3) - (a) ayni donanimda sizinti etkisi, (b) orijinal Keras CNN-GRU.
Tum metrikler OOF dosyalarindan fa_common.state_metrics ile yeniden hesaplanir. Eksik dosya varsa raporda aynen yazilir (uydurma yok).
Girdi (salt okunur): reports/faz_a/oof (S1 uzak CNN-GRU, B2), reports/faz_a2/oof (S0 tohum 1-4 yerel, B2 S0), reports/faz_a2/02_leakage_summary.csv,
  reports/faz_a4/oof (cnngru__S1local, keras__S1, keras__S0), 13_cnn_s1_local_runs.csv, 14_keras_runs.csv.
Cikti: reports/faz_a4/03_leakage_matched_keras.md (+ 03_leakage_matched_keras_*.csv)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
import numpy as np
import pandas as pd
from scipy import stats
import fa_common as fc

R4 = ROOT / "reports" / "faz_a4"
R2 = ROOT / "reports" / "faz_a2"
RA = ROOT / "reports" / "faz_a"
OOF4, OOF2, OOFA = R4 / "oof", R2 / "oof", RA / "oof"
LINES: list[str] = []
MISSING: list[str] = []


def out(s=""):
    LINES.append(s)
    print(s, flush=True)


def load(p: Path):
    if not p.exists():
        MISSING.append(str(p.relative_to(ROOT)))
        return None
    return pd.read_csv(p)


def metr(df):
    return fc.state_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())


def key(df):
    return list(zip(df["session"], df["frame_id"]))


def welch(x0, x1, level=0.95):
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    n0, n1 = len(x0), len(x1)
    m0, m1 = x0.mean(), x1.mean()
    v0, v1 = x0.var(ddof=1), x1.var(ddof=1)
    se = np.sqrt(v0 / n0 + v1 / n1)
    df = (v0 / n0 + v1 / n1) ** 2 / ((v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1))
    tq = stats.t.ppf(0.5 + level / 2, df)
    d = m0 - m1
    t = d / se
    return dict(m0=m0, s0=np.sqrt(v0), n0=n0, m1=m1, s1=np.sqrt(v1), n1=n1, d=d, se=se, df=df,
                lo=d - tq * se, hi=d + tq * se, t=t, p=2 * stats.t.sf(abs(t), df))


def paired(a, b, level=0.95):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m, s = d.mean(), d.std(ddof=1)
    se = s / np.sqrt(n)
    tq = stats.t.ppf(0.5 + level / 2, n - 1)
    t = m / se
    return dict(n=n, mean=m, sd=s, se=se, lo=m - tq * se, hi=m + tq * se, t=t, p=2 * stats.t.sf(abs(t), n - 1),
                n_pos=int((d > 0).sum()))


def pp(x, nd=2, sign=False):
    v = 100 * x
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


# ----------------------------------------------------------------------------- yukle
rows = []        # her kosu icin metrik
oofs = {}        # (model, split, run) -> df


def add(model, split, run, df, src):
    if df is None:
        return
    m = metr(df)
    rows.append(dict(model=model, split=split, run=run, n=m["n"], acc=m["acc"], macro_f1=m["macro_f1"], src=src))
    oofs[(model, split, run)] = df


for f in range(5):
    add("cnngru_remote", "S1", f, load(OOFA / f"cnngru__S1__f{f}.csv.gz"), "Faz A, uzak RTX 3090")
    add("cnngru_local", "S1", f, load(OOF4 / f"cnngru__S1local__f{f}.csv.gz"), "A4.3a, yerel GTX 1650S")
    add("keras", "S1", f, load(OOF4 / f"keras__S1__f{f}.csv.gz"), "A4.3b, uzak RTX 3090")
    add("b2", "S1", f, load(OOFA / f"b2__S1__f{f}.csv.gz"), "Faz A, CPU")
for sd in (1, 2, 3, 4):
    add("cnngru_local", "S0", sd, load(OOF2 / f"cnngru__S0s{sd}__f0.csv.gz"), "A2.2, yerel GTX 1650S")
add("cnngru_remote", "S0", 42, load(OOFA / "cnngru__S0__f0.csv.gz"), "Faz A, uzak RTX 3090")
add("keras", "S0", 42, load(OOF4 / "keras__S0__f0.csv.gz"), "A4.3b, uzak RTX 3090")
add("b2", "S0", 42, load(OOF2 / "b2__S0s42__f0.csv.gz"), "A2.2, CPU")
RUNS = pd.DataFrame(rows)
RUNS.to_csv(R4 / "03_leakage_matched_keras_per_run.csv", index=False)


def vec(model, split, runs, metric):
    d = RUNS[(RUNS.model == model) & (RUNS.split == split) & (RUNS.run.isin(runs))].sort_values("run")
    return d[metric].to_numpy(), list(d["run"])


def have(model, split, runs):
    d = RUNS[(RUNS.model == model) & (RUNS.split == split) & (RUNS.run.isin(runs))]
    return len(d) == len(runs)


FOLDS = list(range(5))
ok_local = have("cnngru_local", "S1", FOLDS) and have("cnngru_local", "S0", [1, 2, 3, 4])
ok_remote = have("cnngru_remote", "S1", FOLDS) and have("cnngru_remote", "S0", [42])
ok_keras = have("keras", "S1", FOLDS) and have("b2", "S1", FOLDS)

# ----------------------------------------------------------------------------- capraz kontroller
chk = []
# (1) pencere eslesmesi
base_keys = {f: key(oofs[("b2", "S1", f)]) for f in FOLDS if ("b2", "S1", f) in oofs}
for model in ("cnngru_remote", "cnngru_local", "keras"):
    for f in FOLDS:
        if (model, "S1", f) in oofs and f in base_keys:
            assert key(oofs[(model, "S1", f)]) == base_keys[f], f"{model} S1 f{f}: test pencereleri B2 ile ayni degil"
            assert (oofs[(model, "S1", f)]["y_true"].to_numpy() == oofs[("b2", "S1", f)]["y_true"].to_numpy()).all()
chk.append("S1: cnngru_remote / cnngru_local / keras OOF dosyalarının test pencereleri (oturum, kare) ve y_true, B2 ile her katta birebir aynı (assert).")
if ("keras", "S0", 42) in oofs:
    k42 = key(oofs[("keras", "S0", 42)])
    for model in ("cnngru_remote", "b2"):
        if (model, "S0", 42) in oofs:
            assert key(oofs[(model, "S0", 42)]) == k42
            assert (oofs[(model, "S0", 42)]["y_true"].to_numpy() == oofs[("keras", "S0", 42)]["y_true"].to_numpy()).all()
    chk.append("S0 tohum 42: Keras, uzak CNN-GRU ve B2 OOF test pencereleri (n=3.250) birebir aynı (assert).")
# (2) runs.csv ile yeniden hesap
for fn, mdl, spl_name in (("13_cnn_s1_local_runs.csv", "cnngru_local", "S1local"), ("14_keras_runs.csv", "keras", None)):
    rr = load(R4 / fn)
    if rr is None:
        continue
    bad = []
    for _, r in rr.iterrows():
        spl = r["split"] if spl_name is None else "S1"
        run = int(r["fold"])
        if mdl == "keras" and spl == "S0":
            run = 42                       # S0 OOF tohum 42 ile anahtarlandi (runs.csv'de fold=0)
        hit = RUNS[(RUNS.model == mdl) & (RUNS.split == spl) & (RUNS.run == run)]
        if len(hit) != 1:
            bad.append(f"{spl} f{run}: OOF yok")
            continue
        if abs(hit["acc"].iloc[0] - r["test_acc"]) > 6e-6 or abs(hit["macro_f1"].iloc[0] - r["test_macro_f1"]) > 6e-6:
            bad.append(f"{spl} f{run}: runs.csv {r['test_acc']:.5f}/{r['test_macro_f1']:.5f} vs OOF {hit['acc'].iloc[0]:.5f}/{hit['macro_f1'].iloc[0]:.5f}")
    chk.append(f"`{fn}` ({len(rr)} koşu) ile OOF'tan yeniden hesaplanan accuracy/macro-F1: " + ("hepsi eşleşti (|fark| < 6e-6)." if not bad else "**UYUŞMAZLIK: " + "; ".join(bad) + "**"))
# (3) Faz A uzak S1 CNN-GRU ve 5-tohum sizinti ozeti
summ = load(R2 / "02_leakage_summary.csv")
mixed = {}
if summ is not None:
    for mk in ("acc", "macro_f1"):
        r = summ[(summ.model == "cnngru") & (summ.metric == mk)].iloc[0]
        mixed[mk] = dict(d=r["d"], lo=r["lo"], hi=r["hi"], m0=r["m0"], m1=r["m1"])
    if ok_remote:
        for mk in ("acc", "macro_f1"):
            m1, _ = vec("cnngru_remote", "S1", FOLDS, mk)
            assert abs(m1.mean() - mixed[mk]["m1"]) < 1e-9, "uzak S1 ortalamasi 02_leakage_summary ile tutmuyor"
        chk.append("Uzak S1 CNN-GRU kat ortalamaları (acc/macro-F1) `faz_a2/02_leakage_summary.csv` içindeki S1 ortalamasıyla birebir aynı.")

# ----------------------------------------------------------------------------- (a) sizinti etkisi
LEAK = []   # csv
est = {}    # metric -> dict(name -> d)
if ok_local:
    for mk, mlabel in (("acc", "accuracy"), ("macro_f1", "macro-F1")):
        x0, _ = vec("cnngru_local", "S0", [1, 2, 3, 4], mk)
        x1, _ = vec("cnngru_local", "S1", FOLDS, mk)
        w = welch(x0, x1)
        w.update(kind="yerel-eslesmis (yerel S0 tohum 1-4 - yerel S1)", metric=mk)
        LEAK.append(w)
        est.setdefault(mk, {})["local"] = w
        # capraz donanim: yerel S0 - uzak S1 (karisik, yalniz bilgi)
        if ok_remote:
            x1r, _ = vec("cnngru_remote", "S1", FOLDS, mk)
            w2 = welch(x0, x1r)
            w2.update(kind="CAPRAZ-donanim (yerel S0 tohum 1-4 - uzak S1) [karisik]", metric=mk)
            LEAK.append(w2)
            # yerel S1 - uzak S1 (kat bazli eslesik + Welch)
            pr = paired(x1, x1r)
            wl = welch(x1, x1r)
            LEAK.append(dict(kind="yerel S1 - uzak S1 (Welch)", metric=mk, **{k: wl[k] for k in wl}))
            est.setdefault(mk, {})["s1_loc_minus_rem_paired"] = pr
            est[mk]["s1_loc_minus_rem_welch"] = wl
if ok_remote:
    for mk in ("acc", "macro_f1"):
        s0v, _ = vec("cnngru_remote", "S0", [42], mk)
        x1r, _ = vec("cnngru_remote", "S1", FOLDS, mk)
        d = float(s0v[0] - x1r.mean())
        est.setdefault(mk, {})["remote"] = dict(d=d, s0=float(s0v[0]), m1=float(x1r.mean()), s1=float(x1r.std(ddof=1)))
        LEAK.append(dict(kind="uzak-eslesmis (S0 tohum 42 - uzak S1) [tek S0 kosusu, GA yok]", metric=mk, d=d, m0=float(s0v[0]),
                         m1=float(x1r.mean()), s1=float(x1r.std(ddof=1)), n0=1, n1=5))
for mk in mixed:
    LEAK.append(dict(kind="5-tohum karisik (S0: uzak 42 + yerel 1-4; S1: uzak) [A2.2]", metric=mk, d=mixed[mk]["d"], lo=mixed[mk]["lo"],
                     hi=mixed[mk]["hi"], m0=mixed[mk]["m0"], m1=mixed[mk]["m1"], n0=5, n1=5))
    est.setdefault(mk, {})["mixed"] = mixed[mk]
LK = pd.DataFrame(LEAK)
LK.to_csv(R4 / "03_leakage_matched_keras_leakage.csv", index=False)

# kural: baslik = iki ayni-donanim tahmininden kucuk olan
headline = {}
for mk in ("acc", "macro_f1"):
    e = est.get(mk, {})
    if "local" in e and "remote" in e:
        dl, dr = e["local"]["d"], e["remote"]["d"]
        headline[mk] = ("uzak-eşleşmiş", dr) if dr <= dl else ("yerel-eşleşmiş", dl)

# ----------------------------------------------------------------------------- (b) Keras
KER = []
kd = {}
if ok_keras:
    for mk, mlabel in (("acc", "accuracy"), ("macro_f1", "macro-F1")):
        kv, _ = vec("keras", "S1", FOLDS, mk)
        bv, _ = vec("b2", "S1", FOLDS, mk)
        pr = paired(kv, bv)
        kd[mk] = dict(keras=kv, b2=bv, paired=pr)
        KER.append(dict(metric=mk, keras_mean=kv.mean(), keras_sd=kv.std(ddof=1), b2_mean=bv.mean(), b2_sd=bv.std(ddof=1), **{f"diff_{k}": v for k, v in pr.items()}))
pd.DataFrame(KER).to_csv(R4 / "03_leakage_matched_keras_keras_vs_b2.csv", index=False)

# havuzlanmis (S1 tum OOF birlestirilmis) duyarlilik
pooled = {}
if ok_keras:
    for m in ("keras", "b2", "cnngru_remote", "cnngru_local"):
        if have(m, "S1", FOLDS):
            d = pd.concat([oofs[(m, "S1", f)] for f in FOLDS], ignore_index=True)
            pooled[m] = metr(d)

# ----------------------------------------------------------------------------- rapor
out("# Faz A4 / A4.3 — Sızıntı etkisi aynı donanımda; orijinal Keras CNN-GRU (M3)")
out()
out("Üreten: `faz_a4/03_leakage_matched_keras.py`. Önceden kayıtlı plan: `faz_a4/DESIGN_A4.md` §A4.3 (okuma kuralı ve karar kuralı sonuç görülmeden yazılmıştır). "
    "Bu bölüm **ikincil** analizdir; Faz A'nın sayıları birincil kalır. Tüm sayılar OOF dosyalarından `fa_common.state_metrics` ile yeniden hesaplandı "
    "(accuracy; macro-F1 = test kümesinde bulunan sınıflar üzerinden). Puan = yüzde puan.")
out()
out("## Koşu durumu ve çapraz kontroller")
out("- PyTorch yerel S1 işi (`job-20261105-001436-6884`): 5 kat tamam (`13_cnn_s1_local_runs.csv`), betik `faz_a4/13_cnn_s1_local.py`.")
out("- Keras uzak işi (`job-20261105-003141-626-10306`): S1 5 kat + S0 tohum 42 tamam (`14_keras_runs.csv`); TF 2.18.0 GPU + Keras 3.9.2, RTX 3090 (günlükte `Created device ... RTX 3090`). "
    "`14_keras_runs.csv`'deki `device=cpu` sütunu betikte sabit yazılmış bir etikettir (14_keras_train.py), gerçek cihaz GPU'dur (epoch başına ~6,6 s; yerel CPU'da ~210 s idi).")
for c in chk:
    out(f"- {c}")
if MISSING:
    out(f"- **EKSİK DOSYALAR:** {', '.join(MISSING)}. Bu dosyalara bağlı hücreler aşağıda boş bırakıldı / hesaplanmadı.")
else:
    out("- Eksik dosya yok.")
out()

# ---- tohum/kat tablosu
out("## Koşu başına sonuçlar (test pencereleri)")
out()
out("| model | bölme | koşu (kat / tohum) | n | accuracy | macro-F1 | kaynak |")
out("|---|---|---|---|---|---|---|")
name = {"cnngru_remote": "CNN-GRU (PyTorch, uzak 3090)", "cnngru_local": "CNN-GRU (PyTorch, yerel 1650S)", "keras": "CNN-GRU (Keras, uzak 3090)", "b2": "B2 (el yapımı özellik + RF)"}
for m in ("keras", "cnngru_remote", "cnngru_local", "b2"):
    for spl in ("S1", "S0"):
        for _, r in RUNS[(RUNS.model == m) & (RUNS.split == spl)].sort_values("run").iterrows():
            out(f"| {name[m]} | {spl} | {int(r['run'])} | {int(r['n'])} | {pp(r['acc'], 2)} | {pp(r['macro_f1'], 2)} | {r['src']} |")
out()

# ---- (a)
out("## (a) Aynı donanımda sızıntı etkisi (S0 rastgele pencere bölmesi − S1 grup-bazlı katlar), CNN-GRU PyTorch")
out()
if not ok_local:
    out("**Yerel eşleşmiş tahmin hesaplanamadı:** yerel S1 (5 kat) ve/veya yerel S0 tohum 1–4 OOF dosyaları eksik.")
out("| tahmin | metrik | S0 | S1 | etki (puan) | %95 GA (puan) | not |")
out("|---|---|---|---|---|---|---|")
label_note = {
    "yerel-eslesmis": "S0: yerel tohum 1–4 (n=4), S1: yerel 5 kat (n=5); Welch t",
    "uzak-eslesmis": "S0: uzak tohum 42 (n=1), S1: uzak 5 kat; tek S0 koşusu, GA yok (Faz A2'deki +3,16/+3,73)",
    "5-tohum": "S0: uzak 42 + yerel 1–4 (n=5), S1: uzak 5 kat (A2.2); donanım karışık",
    "CAPRAZ": "yalnız bilgi: farklı donanım",
}
for _, r in LK.iterrows():
    k = r["kind"]
    if k.startswith("yerel S1 - uzak S1"):
        continue
    nt = next((v for kk, v in label_note.items() if k.startswith(kk)), "")
    ci = f"[{pp(r['lo'], 2, True)}, {pp(r['hi'], 2, True)}]" if pd.notna(r.get("lo", np.nan)) else "—"
    s0s = f"{pp(r['m0'], 2)}" + (f" ± {pp(r['s0'], 2)}" if pd.notna(r.get("s0", np.nan)) else "")
    s1s = f"{pp(r['m1'], 2)}" + (f" ± {pp(r['s1'], 2)}" if pd.notna(r.get("s1", np.nan)) else "")
    mname = "accuracy" if r["metric"] == "acc" else "macro-F1"
    out(f"| {k} | {mname} | {s0s} | {s1s} | **{pp(r['d'], 2, True)}** | {ci} | {nt} |")
out()

if ok_local and ok_remote:
    out("**Yerel S1 − uzak S1 (aynı katlar, aynı tohumlar, farklı GPU/PyTorch yapısı):**")
    out()
    out("| metrik | yerel S1 ort ± sd | uzak S1 ort ± sd | fark (yerel − uzak, puan) | eşli kat farkı %95 GA | Welch %95 GA |")
    out("|---|---|---|---|---|---|")
    for mk, ml in (("acc", "accuracy"), ("macro_f1", "macro-F1")):
        pr = est[mk]["s1_loc_minus_rem_paired"]
        wl = est[mk]["s1_loc_minus_rem_welch"]
        out(f"| {ml} | {pp(wl['m0'])} ± {pp(wl['s0'])} | {pp(wl['m1'])} ± {pp(wl['s1'])} | **{pp(pr['mean'], 2, True)}** | "
            f"[{pp(pr['lo'], 2, True)}, {pp(pr['hi'], 2, True)}] (kat farkı sd {pp(pr['sd'])}, {pr['n_pos']}/5 katta yerel > uzak) | [{pp(wl['lo'], 2, True)}, {pp(wl['hi'], 2, True)}] |")
    out()
    out("(Aynı kat bölmesi ve aynı model tohumu kullanıldığından kat bazlı eşli fark, GPU/cuDNN/PyTorch yapısı kaynaklı belirlenimsizliği ve donanım farkını ölçer; "
        "iki yapının bit düzeyinde aynı olması beklenmez.)")
    out()

out("### Önceden yazılmış okuma kuralının uygulanması")
out("Kural (DESIGN_A4 §A4.3a): makaledeki başlık-düzeyi iddia, iki aynı-donanım tahmininden **küçük olanı** kullanır; beş tohumluk karışık tahmin ikincil kalır.")
out()
if headline:
    for mk, ml in (("acc", "accuracy"), ("macro_f1", "macro-F1")):
        e = est[mk]
        out(f"- **{ml}:** yerel-eşleşmiş {pp(e['local']['d'], 2, True)} puan (GA [{pp(e['local']['lo'], 2, True)}, {pp(e['local']['hi'], 2, True)}]) ve "
            f"uzak-eşleşmiş {pp(e['remote']['d'], 2, True)} puan → küçük olan **{headline[mk][0]}**: **{pp(headline[mk][1], 2, True)} puan** başlık değeridir. "
            f"Karışık 5-tohum tahmin {pp(e['mixed']['d'], 2, True)} (ikincil).")
    both_remote = all(v[0] == "uzak-eşleşmiş" for v in headline.values())
    out()
    out("Okuma notları (veriyle sınırlı):")
    el, er = est["acc"]["local"], est["acc"]["remote"]
    out(f"- İki eşleşmiş tahmin birbirinden {pp(abs(el['d'] - er['d']), 1)} puan (accuracy) / "
        f"{pp(abs(est['macro_f1']['local']['d'] - est['macro_f1']['remote']['d']), 1)} puan (macro-F1) ayrılıyor. Bu ayrım donanım, model tohumu ve bölme-tohumu etkilerinden hangisinden geldiği bu verilerle ayrıştırılamaz "
        "(yerel S0 tohum 1–4, uzak S0 tohum 42'den farklı bölmeler de içerir). Her iki tahmin de pozitif ve S1 kat sd'sinden (CNN-GRU için ≈1,3–1,5 puan) büyük; yön (rastgele bölme S1'den yüksek) hiçbir tahminde değişmiyor.")
    for mk, ml in (("acc", "accuracy"), ("macro_f1", "macro-F1")):
        e = est[mk]
        inside = e["local"]["lo"] <= e["remote"]["d"] <= e["local"]["hi"]
        out(f"- {ml}: uzak-eşleşmiş nokta tahmin ({pp(e['remote']['d'], 2, True)}), yerel-eşleşmiş %95 GA'nın [{pp(e['local']['lo'], 2, True)}, {pp(e['local']['hi'], 2, True)}] "
            + ("**içinde**." if inside else "**dışında** (altında)."))
    if both_remote:
        out("- Kural gereği başlık değeri **tek S0 koşusuna** dayanan uzak-eşleşmiş tahmindir (GA hesaplanamaz); makale metninde başlık sayısı yazılırken bunun tek koşu olduğu ve yerel-eşleşmiş değerin (daha büyük, GA'lı) birlikte verilmesi gerekir.")
    out("- Yerel S0 koşuları (91,4–92,2 accuracy) uzak S0 tohum 42'den (90,1) yüksek; yerel S1 ise uzak S1'den biraz düşük (yukarıdaki yerel − uzak tablosu: −0,65 / −0,74 puan, eşli GA 0'ı içeriyor). "
        "Yani yerel donanımda S0 daha yüksek, S1 daha düşük çıkmış; bu, yerel-eşleşmiş farkı büyütüyor. S0 tarafı farklı tohumlar da içerdiği için bunun donanım mı tohum mu olduğu ayrıştırılmadı.")
else:
    out("- Başlık değeri **hesaplanamadı** (yerel ya da uzak eşleşmiş tahmin için gerekli dosyalar eksik).")
out()

# ---- (b)
out("## (b) Orijinal Keras CNN-GRU (klasör 4 defteri `01_mmwave_cnn_lstm_model.ipynb`'in portu)")
out()
out("**Port sadakati (kayıt: DESIGN_A4 Sapmalar, 2026-11-05 A4.3b).** Defterden birebir: GRU(128, dropout 0,3, recurrent_dropout 0,2) → Dense(6, softmax) "
    "(Dense64 ve ek Dropout(0,4) yok); Conv(relu) → BatchNorm sırası; Adam 5e-4 + ExponentialDecay(8000; 0,92), batch 32, ≤ 120 epoch, EarlyStopping(val_loss, patience 25, restore_best), "
    "dengeli class_weight, girdi ölçekleme yok (n_param = 1.914.022, tüm koşularda aynı). **Defterden farklar:** "
    "(i) defter `validation_data=TEST` kullanıyordu → burada Faz A iç doğrulaması (S1: grup bazlı %10; S0: sınıf-tabakalı %10; indeksler `01_state_cnn.py` ile aynı, md5 doğrulandı); "
    "(ii) `ModelCheckpoint(val_accuracy)` kullanılmadı (yalnız EarlyStopping restore_best); "
    "(iii) heatmap Faz A önbelleği (kanal 1 = hız TOPLAMI; defterde hücre ortalaması) ve etiket = `fa_common` y_true; "
    "(iv) tohum = 20261104 + 1000·split_id + kat (defterde tohum yok); "
    "(v) ortam: TF 2.18.0 + Keras 3.9.2 (defteri kaydeden sürüm), GPU (RTX 3090). "
    "**İş parçacığı/donanım sapması (kayıtlı):** yerel CPU koşusu epoch başına ~210 s sürdüğü için iş uzak GPU'ya taşındı; model, veri ve ayarlar aynı kaldı. "
    "Her koşuda kayıtlı `restore check` (geri yüklenen ağırlıkla val_loss = en iyi val_loss) günlükte eşleşiyor.")
out()
if ok_keras:
    out("### S1 (grup-bazlı 5 kat): Keras ve B2, kat bazında")
    out()
    out("| kat | Keras acc | Keras macro-F1 | B2 acc | B2 macro-F1 | PyTorch uzak macro-F1 | PyTorch yerel macro-F1 | Keras − B2 macro-F1 (puan) | epoch (en iyi) |")
    out("|---|---|---|---|---|---|---|---|---|")
    kr = load(R4 / "14_keras_runs.csv")
    for i, f in enumerate(FOLDS):
        ep = ""
        if kr is not None:
            h = kr[(kr["split"] == "S1") & (kr["fold"] == f)]
            if len(h):
                ep = f"{int(h['epochs_run'].iloc[0])} ({int(h['best_epoch'].iloc[0])})"
        pr_ = vec("cnngru_remote", "S1", [f], "macro_f1")[0]
        pl_ = vec("cnngru_local", "S1", [f], "macro_f1")[0]
        out(f"| {f} | {pp(kd['acc']['keras'][i])} | {pp(kd['macro_f1']['keras'][i])} | {pp(kd['acc']['b2'][i])} | {pp(kd['macro_f1']['b2'][i])} | "
            f"{pp(pr_[0]) if len(pr_) else '—'} | {pp(pl_[0]) if len(pl_) else '—'} | {pp(kd['macro_f1']['keras'][i] - kd['macro_f1']['b2'][i], 2, True)} | {ep} |")
    out()
    out("### S1 özet (kat ortalaması ± sd, n = 5 kat)")
    out()
    out("| model | accuracy | macro-F1 | havuzlanmış (5 katın OOF'u birleşik) acc / macro-F1 |")
    out("|---|---|---|---|")
    for m in ("keras", "b2", "cnngru_remote", "cnngru_local"):
        if have(m, "S1", FOLDS):
            a, _ = vec(m, "S1", FOLDS, "acc")
            f1, _ = vec(m, "S1", FOLDS, "macro_f1")
            pm = pooled.get(m)
            out(f"| {name[m]} | {pp(a.mean())} ± {pp(a.std(ddof=1))} | {pp(f1.mean())} ± {pp(f1.std(ddof=1))} | {pp(pm['acc']) if pm else '—'} / {pp(pm['macro_f1']) if pm else '—'} |")
    out()
    pm_ = kd["macro_f1"]["paired"]
    pa_ = kd["acc"]["paired"]
    out("### Eşli kat farkı: Keras − B2 (S1)")
    out()
    out("| metrik | Keras ort | B2 ort | fark (puan) | %95 GA (t, df=4) | kat sd | Keras > B2 olan kat | t | p |")
    out("|---|---|---|---|---|---|---|---|---|")
    for mk, ml in (("macro_f1", "macro-F1"), ("acc", "accuracy")):
        p_ = kd[mk]["paired"]
        out(f"| {ml} | {pp(kd[mk]['keras'].mean())} | {pp(kd[mk]['b2'].mean())} | **{pp(p_['mean'], 2, True)}** | [{pp(p_['lo'], 2, True)}, {pp(p_['hi'], 2, True)}] | {pp(p_['sd'])} | {p_['n_pos']}/5 | {p_['t']:.2f} | {p_['p']:.3f} |")
    out()
    # karar kurali
    diff = pm_["mean"] * 100
    fired = diff > 2.0
    out("### Önceden yazılmış karar kuralının LİTERAL uygulanması")
    out("Kural (DESIGN_A4 §A4.3b; Faz A ölçütü 1): B2'nin S1 macro-F1'i Keras'ınkine 2 puan yakın **ya da üstündeyse** \"basit taban yeterli\" hükmü Keras için de geçerlidir; "
        "Keras B2'yi **2 puandan fazla** geçerse hüküm değişir. Karşılaştırma ölçüsü: S1 macro-F1'in kat ortalamaları (Faz A ölçütünün kullandığı büyüklük); Keras − B2 = "
        f"{pp(kd['macro_f1']['keras'].mean(), 2)} − {pp(kd['macro_f1']['b2'].mean(), 2)} = **{diff:+.2f} puan**.")
    out()
    if fired:
        out(f"- **KURAL TETİKLENDİ:** Keras, B2'yi {diff:.2f} puan (> 2,00) geçiyor → hüküm değişir; makale ana model iddiasını Keras CNN-GRU lehine düzeltmelidir (OVERNIGHT notu: bu gece makale yeniden yazılmaz, `paper/ALT_keras.md` taslağı hazırlanır).")
    else:
        out(f"- **KURAL TETİKLENMEDİ.** Keras − B2 = {diff:+.2f} puan; eşik +2,00. B2, Keras'ın {'önünde' if diff < 0 else 'en fazla 2 puan gerisinde'}"
            f" ({'B2 ' + pp(-pm_['mean'], 2) + ' puan önde' if diff < 0 else ''}). \"Basit taban yeterli\" hükmü Keras için de geçerli (bu ölçütün macro-F1 şartı bakımından).")
    out(f"- Duyarlılık (aynı sonuç mu?): kat bazlı eşli GA [{pp(pm_['lo'], 2, True)}, {pp(pm_['hi'], 2, True)}], Keras'ın B2'yi geçtiği kat sayısı {pm_['n_pos']}/5"
        + (f"; havuzlanmış macro-F1 farkı {100 * (pooled['keras']['macro_f1'] - pooled['b2']['macro_f1']):+.2f} puan" if 'keras' in pooled and 'b2' in pooled else "")
        + f". Kural eşiği (2 puan) {'aşılıyor' if fired else 'aşılmıyor'}: GA üst sınırı {pp(pm_['hi'], 2, True)}"
        + (" eşiği aşıyor (belirsizlik payı 5 kat ile büyük)" if (not fired and pm_['hi'] * 100 > 2) else "") + ".")
    out("- Kapsam: Faz A ölçütü 1'in yalnız birinci şartı (durum macro-F1) Keras için sınanabilir; ikinci şart (D3 döngü-AUROC farkı) Keras anomali boru hattına sokulmadığı için (DESIGN_A4 \"Kapsam dışı\") değerlendirilmedi.")
    out()

# ---- Keras S0 / sizinti / eski makale
out("### Keras: S0 tohum 42, sızıntı etkisi ve eski makalenin %93,66'sı")
out()
if have("keras", "S0", [42]):
    ks = RUNS[(RUNS.model == "keras") & (RUNS.split == "S0")].iloc[0]
    out("| model | S0 tohum 42 acc | S0 tohum 42 macro-F1 |")
    out("|---|---|---|")
    for m in ("keras", "cnngru_remote", "b2"):
        h = RUNS[(RUNS.model == m) & (RUNS.split == "S0") & (RUNS.run == 42)]
        if len(h):
            out(f"| {name[m]} | {pp(h['acc'].iloc[0])} | {pp(h['macro_f1'].iloc[0])} |")
    out("| eski makale (Keras defteri, rastgele bölme; değer hakem raporunda yazarların notu olarak anılıyor, burada yeniden üretilmedi) | 93,66 | — |")
    out()
    if ok_keras:
        for mk, ml in (("acc", "accuracy"), ("macro_f1", "macro-F1")):
            kv = kd[mk]["keras"]
            d = float(ks[mk] - kv.mean())
            out(f"- **Keras sızıntı etkisi ({ml}):** S0(42) {pp(ks[mk])} − S1 ort. {pp(kv.mean())} = **{pp(d, 2, True)} puan** (tek S0 koşusu; S1 kat sd {pp(kv.std(ddof=1))}; GA yok). "
                f"Karşılaştırma: PyTorch CNN-GRU uzak-eşleşmiş "
                + (f"{pp(est[mk]['remote']['d'], 2, True)}" if "remote" in est.get(mk, {}) else "—") + ", yerel-eşleşmiş "
                + (f"{pp(est[mk]['local']['d'], 2, True)}" if "local" in est.get(mk, {}) else "—") + ".")
    b2s = RUNS[(RUNS.model == "b2") & (RUNS.split == "S0") & (RUNS.run == 42)]
    if len(b2s):
        out(f"- **S0 tohum 42'de sıralama (Keras vs B2):** Keras {pp(ks['acc'])} / {pp(ks['macro_f1'])} (acc / macro-F1), B2 {pp(b2s['acc'].iloc[0])} / {pp(b2s['macro_f1'].iloc[0])} → "
            f"Keras − B2 = {pp(ks['acc'] - b2s['acc'].iloc[0], 2, True)} / {pp(ks['macro_f1'] - b2s['macro_f1'].iloc[0], 2, True)} puan. "
            + ("Rastgele bölmede Keras önde, grup-bazlı S1'de B2 önde → \"sıralama ters dönüyor\" bulgusu Keras ile de görülüyor (tek S0 koşusu; yön olarak)." if (ok_keras and ks['macro_f1'] > b2s['macro_f1'].iloc[0] and kd['macro_f1']['paired']['mean'] < 0)
               else "Yukarıdaki S1 farkıyla birlikte okunmalıdır."))
    r42 = RUNS[(RUNS.model == "cnngru_remote") & (RUNS.split == "S0") & (RUNS.run == 42)]
    if len(r42):
        out(f"- **Keras portu ile PyTorch portu aynı S0 bölmesinde:** Keras {pp(ks['acc'])} vs PyTorch {pp(r42['acc'].iloc[0])} → Keras − PyTorch = {pp(ks['acc'] - r42['acc'].iloc[0], 2, True)} puan accuracy "
            f"({pp(ks['macro_f1'] - r42['macro_f1'].iloc[0], 2, True)} macro-F1).")
    out(f"- **Eski makalenin %93,66'sı ile karşılaştırma:** Keras portu S0 tohum 42'de {pp(ks['acc'])} → eski değerin **{pp(0.9366 - ks['acc'], 2)} puan altında**; PyTorch portu {pp(r42['acc'].iloc[0]) if len(r42) else '—'} idi "
        f"(hakem raporunda belirtilen ~3,6 puanlık açık). Yani Keras'a geçmek açığın yalnız {pp(ks['acc'] - r42['acc'].iloc[0]) if len(r42) else '—'} puanını kapatıyor; "
        "katman sırası / dropout yerleşimi açıklaması (yazarların varsayımı) bu deneyle **desteklenmiyor**. Kalan açık bu deneyde izole edilmedi; "
        "eski defterin test kümesini doğrulama olarak kullanması ve `ModelCheckpoint(val_accuracy)` ile en iyi test epoch'unu seçmesi (iyimser bir seçim) ile heatmap kanal 1 tanımı, bölme ve etiket farkları adaydır, "
        "ama hiçbiri burada ayrı ayrı sınanmadı (aday açıklamalardır, bulgu değil).")
else:
    out("**Keras S0 tohum 42 OOF dosyası eksik; bu bölüm hesaplanamadı.**")
out()

out("## Sınırlar ve dürüstlük notları")
out("- Yerel-eşleşmiş tahmin n=4 (S0) vs n=5 (S1) ile Welch aralığıdır; S0 tohumlarının test kümeleri aynı 16.248 pencereden çekildiği için örtüşür, S1 katlarının eğitim kümeleri paylaşılır; aralıklar yaklaşıktır (A2.2'deki not geçerli).")
out("- Uzak-eşleşmiş tahmin tek S0 koşusudur; makaledeki başlık sayısı bu kurala göre seçilirse bunun belirsizliği bilinmeyen bir tek-koşu belirsizliğidir. Her iki eşleşmiş tahmin ve GA'sı birlikte verilmelidir.")
out("- Keras: S0 için tek koşu (tohum 42), S1 için 5 kat; eşli GA df=4 ile geniştir. Karar kuralı nokta tahmin (kat ortalaması farkı) üzerinden literal uygulandı; GA ve kat sayıları duyarlılık olarak verildi, kuralın yerine geçmez.")
out("- Keras GPU koşusu cuDNN belirlenimsizliği içerir; yerel CPU ile bit düzeyinde aynı sonuç beklenmez. Keras modeli anomali boru hattına sokulmadı (kapsam dışı).")
out("- Seçici raporlama yok: A4.3'te tanımlanan tüm sayılar yukarıdadır; eksik koşu yoktur" + (" (ancak: " + ", ".join(MISSING) + " eksik)." if MISSING else "."))

(R4 / "03_leakage_matched_keras.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")
print("\n[done] wrote", R4 / "03_leakage_matched_keras.md", file=sys.stderr)

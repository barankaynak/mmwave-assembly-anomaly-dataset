"""Faz A4.2 (M2) - Tempo tespitinin kanal dokumu ve oracle'in slow'daki davranisi.  Brief: faz_a4/DESIGN_A4.md.

Girdi: reports/faz_a/04_cycle_scores.csv.gz (role == test).  Birim = (dongu, f) cifti (Faz A 'tespit' tanimi);
GA = dongu-kumeli bootstrap (dongu basina 5 kat uzerinden ortalama, dongu yeniden orneklenir).
Kontrol: a2_cycles ile dongu / ziyaret dizileri yeniden kurulup CSV'deki visit_seq ve D1/D2 ile bire bir karsilastirilir.
Calistirma: bash tools/local_run.sh faz_a4/02_tempo_channels.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
sys.path.insert(0, str(ROOT / "faz_a2"))
import fa_common as fc  # noqa: E402
import a2_cycles as ac  # noqa: E402

OUT = ROOT / "reports" / "faz_a4"
OUT.mkdir(parents=True, exist_ok=True)
TOL = 1e-9
RNG = np.random.RandomState(20261105)
NB = 4000
CELLS = [("b2", "pred"), ("tof", "tof"), ("tof", "pred"), ("b2", "tof")]   # ilk ikisi brief'in birincil hucreleri
PRIMARY = CELLS[:2]
KEY = ["state_source", "segmentation", "session", "cycle_idx"]

S = pd.read_csv(ROOT / "reports" / "faz_a" / "04_cycle_scores.csv.gz")
S["visit_seq"] = S["visit_seq"].fillna("").astype(str)   # bos dizi (ziyaret yok) -> ""
T = S[S["role"] == "test"].copy()
T["has5"] = T["visit_seq"].str.split("-").apply(lambda l: "5" in l)
T["a1"] = T["p1"] <= 0.05 + TOL            # D1 kanali: Bonferroni esigi min(p1,p2) <= alpha/2 = 0.05
T["a2"] = T["p2"] <= 0.05 + TOL
T["a3"] = T["p3"] <= 0.10 + TOL
assert (T["a3"] == (T["a1"] | T["a2"])).all(), "D3 alarmi != D1|D2 (alpha=0.10)"
T["b2_10"] = T["p2"] <= 0.10 + TOL          # onceden yazilmis ikincil tempo tanimi: D2 tek basina, alpha = 0.10
T["b2_05"] = T["p2"] <= 0.05 + TOL
T["b1_10"] = T["p1"] <= 0.10 + TOL
T["ch"] = np.select([T.a1 & T.a2, T.a1, T.a2], ["both", "D1_only", "D2_only"], "none")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cboot(d, col):
    """Dongu-kumeli bootstrap: dongu basina f uzerinden ortalama; dongu yeniden orneklenir. Doner (nokta, lo, hi)."""
    g = d.groupby(["session", "cycle_idx"])[col].mean().to_numpy(float)
    if len(g) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.randint(0, len(g), size=(NB, len(g)))
    bs = g[idx].mean(axis=1)
    return (float(d[col].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def fmt(p, lo=None, hi=None, k=None, n=None):
    s = f"{p:.3f}"
    if lo is not None and not np.isnan(lo):
        s += f" [{lo:.3f}; {hi:.3f}]"
    if k is not None:
        s = f"{k}/{n} = " + s
    return s


def cell(src, seg):
    return T[(T.state_source == src) & (T.segmentation == seg)]


# ===================================================================== 0. kontrol: a2_cycles ile yeniden kur
print("== 0. kontrol (a2_cycles ile yeniden insa) ==")
LAB = ac.source_labels(sources=("tof", "b2"), verbose=False)
check_rows = []
for src, seg in PRIMARY:
    C, _ = ac.build_cycles(src, seg, LAB[src])
    C["vs"] = C["seq"].apply(lambda s: "-".join(map(str, s)))
    m = cell(src, seg)[["session", "cycle_idx", "visit_seq", "atype"]].drop_duplicates()
    mm = m.merge(C[["session", "cycle_idx", "vs", "atype"]], on=["session", "cycle_idx"], how="outer", indicator=True,
                 suffixes=("", "_ac"))
    n_both = int((mm["_merge"] == "both").sum())
    n_only_csv = int((mm["_merge"] == "left_only").sum())
    n_ok = int(((mm["_merge"] == "both") & (mm["visit_seq"].astype(str) == mm["vs"]) & (mm["atype"] == mm["atype_ac"])).sum())
    print(f"  {src} x {seg}: CSV test dongusu {len(m)}, a2_cycles ile eslesen {n_both}, visit_seq+atype ayni {n_ok}, yalniz CSV'de {n_only_csv}")
    assert n_only_csv == 0 and n_ok == n_both == len(m)
    check_rows.append((src, seg, len(m), n_ok))

# ===================================================================== 1. kanal dokumu (D3 alarmi alan fast/slow)
print("\n== 1. D3 alarmi (alpha = 0.10; esik min(p1,p2) <= 0.05) kanal dokumu ==")
rows = []
for src, seg in CELLS:
    d = cell(src, seg)
    for at in ("fast", "slow", "tempo"):
        q = d[d.atype.isin(["fast", "slow"])] if at == "tempo" else d[d.atype == at]
        n = len(q)
        n3 = int(q.a3.sum())
        r = dict(state_source=src, segmentation=seg, atype=at, n_pairs=n, n_cycles=q.groupby(["session", "cycle_idx"]).ngroups,
                 n_D3=n3)
        for ch in ("D1_only", "D2_only", "both", "none"):
            k = int((q.ch == ch).sum())
            r[f"n_{ch}"] = k
            r[f"share_of_D3_{ch}"] = (k / n3 if (n3 and ch != "none") else np.nan)
            r[f"rate_{ch}"] = k / n
        for nm, col in (("D1alarm", "a1"), ("D2alarm", "a2"), ("D3alarm", "a3")):
            p, lo, hi = cboot(q, col)
            r[f"{nm}"], r[f"{nm}_lo"], r[f"{nm}_hi"] = p, lo, hi
        rows.append(r)
CH = pd.DataFrame(rows)
CH.to_csv(OUT / "02_tempo_channel_split.csv", index=False)
for src, seg in PRIMARY:
    print(f"-- {src} x {seg}")
    print(CH[(CH.state_source == src) & (CH.segmentation == seg)][
        ["atype", "n_pairs", "n_cycles", "n_D3", "n_D1_only", "n_D2_only", "n_both", "n_none", "D3alarm", "D1alarm", "D2alarm"]
    ].to_string(index=False))

# ===================================================================== 2. kisalmis dizi (Box5 yok)
print("\n== 2. Kisalmis dizi (ziyaret dizisinde Box5 yok) ==")
rows = []
for src, seg in CELLS:
    d = cell(src, seg)
    for at in ("normal", "fast", "slow", "seq135", "seq145"):
        q = d[d.atype == at]
        for nm, qq in (("tum", q), ("kisalmis(Box5 yok)", q[~q.has5]), ("tam(Box5 var)", q[q.has5])):
            n = len(qq)
            r = dict(state_source=src, segmentation=seg, atype=at, grup=nm, n_pairs=n,
                     n_cycles=qq.groupby(["session", "cycle_idx"]).ngroups,
                     share_trunc_pairs=(len(q[~q.has5]) / len(q) if len(q) else np.nan) if nm == "tum" else np.nan)
            for cn, col in (("D1alarm", "a1"), ("D2alarm", "a2"), ("D3alarm", "a3"), ("D2alone_a10", "b2_10")):
                k = int(qq[col].sum())
                r[f"k_{cn}"] = k
                r[cn] = k / n if n else np.nan
                lo, hi = wilson(k, n)
                r[cn + "_wlo"], r[cn + "_whi"] = lo, hi
            rows.append(r)
TR = pd.DataFrame(rows)
TR.to_csv(OUT / "02_tempo_truncation.csv", index=False)
for src, seg in PRIMARY:
    print(f"-- {src} x {seg}")
    print(TR[(TR.state_source == src) & (TR.segmentation == seg) & TR.atype.isin(["normal", "fast", "slow"])][
        ["atype", "grup", "n_pairs", "n_cycles", "share_trunc_pairs", "D1alarm", "D2alarm", "D3alarm", "D2alone_a10"]].to_string(index=False))

# ===================================================================== 3. ikincil tempo tanimi: D2 tek basina
print("\n== 3. Ikincil tempo tanimi: D2 tek basina alpha = 0.10 (ve referans: D2 tek basina 0.05, D3 0.10) ==")
rows = []
for src, seg in CELLS:
    d = cell(src, seg)
    nrm = d[d.atype == "normal"]
    r0 = dict(state_source=src, segmentation=seg)
    for nm, col in (("D2_a10", "b2_10"), ("D2_a05", "b2_05"), ("D3_a10", "a3"), ("D1_a10", "b1_10")):
        k, n = int(nrm[col].sum()), len(nrm)
        lo, hi = wilson(k, n)
        r0[f"FAR_{nm}"], r0[f"FAR_{nm}_wlo"], r0[f"FAR_{nm}_whi"] = k / n, lo, hi
        r0[f"nNormal"] = n
    for at in ("fast", "slow", "tempo"):
        q = d[d.atype.isin(["fast", "slow"])] if at == "tempo" else d[d.atype == at]
        for nm, col in (("D2_a10", "b2_10"), ("D2_a05", "b2_05"), ("D3_a10", "a3")):
            p, lo, hi = cboot(q, col)
            r0[f"det_{at}_{nm}"], r0[f"det_{at}_{nm}_lo"], r0[f"det_{at}_{nm}_hi"] = p, lo, hi
    rows.append(r0)
SEC = pd.DataFrame(rows)
SEC.to_csv(OUT / "02_tempo_secondary_D2alone.csv", index=False)
for _, r in SEC.iterrows():
    print(f"-- {r.state_source} x {r.segmentation}: normal n={int(r.nNormal)}  FAR D2@0.10={r.FAR_D2_a10:.3f} "
          f"[{r.FAR_D2_a10_wlo:.3f}; {r.FAR_D2_a10_whi:.3f}]; FAR D3@0.10={r.FAR_D3_a10:.3f}")
    for at in ("fast", "slow", "tempo"):
        print(f"   {at:6s}: D2 tek basina a=0.10: {r[f'det_{at}_D2_a10']:.3f} [{r[f'det_{at}_D2_a10_lo']:.3f}; {r[f'det_{at}_D2_a10_hi']:.3f}]"
              f" | D2 a=0.05: {r[f'det_{at}_D2_a05']:.3f} | D3 a=0.10: {r[f'det_{at}_D3_a10']:.3f} [{r[f'det_{at}_D3_a10_lo']:.3f}; {r[f'det_{at}_D3_a10_hi']:.3f}]")

# ===================================================================== 4. oracle (ToF x tof) slow dokumu
print("\n== 4. Oracle (ToF x tof) slow: D1 / D2 / D3 ayri ayri ve p dagilimi ==")
orows, prow = [], []
for seg in ("tof", "pred"):
    d = cell("tof", seg)
    q = d[d.atype == "slow"]
    nrm = d[d.atype == "normal"]
    print(f"-- ToF x {seg}: slow n_pairs={len(q)}, dongu={q.groupby(['session','cycle_idx']).ngroups}")
    for nm, col in (("D1 (p1<=0.05)", "a1"), ("D2 (p2<=0.05)", "a2"), ("D3 (alpha=0.10)", "a3"), ("D2 tek basina (p2<=0.10)", "b2_10"),
                    ("D1 tek basina (p1<=0.10)", "b1_10")):
        p, lo, hi = cboot(q, col)
        orows.append(dict(segmentation=seg, olcut=nm, rate=p, lo=lo, hi=hi, k=int(q[col].sum()), n=len(q),
                          normal_FAR=float(nrm[col].mean())))
        print(f"   {nm:26s}: {int(q[col].sum())}/{len(q)} = {p:.3f} [{lo:.3f}; {hi:.3f}]   (normal FAR {nrm[col].mean():.3f})")
    ch = q["ch"].value_counts()
    print("   kanal:", {k: int(ch.get(k, 0)) for k in ("D1_only", "D2_only", "both", "none")})
    for v in ("p1", "p2", "p3"):
        x = q[v].to_numpy(float)
        qs = np.percentile(x, [0, 10, 25, 50, 75, 90, 100])
        bins = [0, 0.05 + TOL, 0.10 + TOL, 0.20, 0.50, 1.0 + TOL]
        hb = np.histogram(x, bins=bins)[0]
        prow.append(dict(segmentation=seg, p=v, n=len(x), min=qs[0], q10=qs[1], q25=qs[2], median=qs[3], q75=qs[4], q90=qs[5],
                         max=qs[6], n_le05=int(hb[0]), n_05_10=int(hb[1]), n_10_20=int(hb[2]), n_20_50=int(hb[3]), n_gt50=int(hb[4]),
                         n_eq1=int((x >= 1 - TOL).sum())))
        print(f"   {v}: min {qs[0]:.3f} q10 {qs[1]:.3f} q25 {qs[2]:.3f} med {qs[3]:.3f} q75 {qs[4]:.3f} q90 {qs[5]:.3f} max {qs[6]:.3f} | "
              f"<=.05:{hb[0]} (.05,.10]:{hb[1]} (.10,.20]:{hb[2]} (.20,.50]:{hb[3]} >.50:{hb[4]} (p==1: {(x >= 1 - TOL).sum()})")
    # kat bazinda ve dongu bazinda tutarlilik
    pf = q.groupby("f")[["a1", "a2", "a3"]].mean()
    print("   kat bazinda tespit (f: D1/D2/D3):", {int(f): tuple(round(v, 2) for v in r) for f, r in pf.iterrows()})
    pc = q.groupby(["session", "cycle_idx"])["a3"].agg(["sum", "count"])
    print(f"   dongu bazinda D3: 5/5 kat alarm {int((pc['sum'] == pc['count']).sum())}, hic alarm yok {int((pc['sum'] == 0).sum())}, "
          f"karisik {int(((pc['sum'] > 0) & (pc['sum'] < pc['count'])).sum())} (dongu {len(pc)})")
    # betimleyici: alarm / kacirilan dongular
    for nm, qq in (("alarm(D3)", q[q.a3]), ("kacirilan(D3)", q[~q.a3])):
        print(f"   {nm:14s}: n={len(qq)} dur_s medyan {qq.dur_s.median():.2f} [IQR {qq.dur_s.quantile(.25):.2f}; {qq.dur_s.quantile(.75):.2f}] "
              f"D0 med {qq.D0.median():.2f}  D1 med {qq.D1.median():.2f}  D2 med {qq.D2.median():.2f}  n_visits med {qq.n_visits.median():.0f}  "
              f"visit_seq: {qq.visit_seq.value_counts().head(3).to_dict()}")
    print(f"   normal: dur_s medyan {nrm.dur_s.median():.2f} [IQR {nrm.dur_s.quantile(.25):.2f}; {nrm.dur_s.quantile(.75):.2f}]  "
          f"D0 med {nrm.D0.median():.2f}  D1 med {nrm.D1.median():.2f}  D2 med {nrm.D2.median():.2f}")
    print(f"   slow: dur_s medyan {q.dur_s.median():.2f} [IQR {q.dur_s.quantile(.25):.2f}; {q.dur_s.quantile(.75):.2f}]")
pd.DataFrame(orows).to_csv(OUT / "02_oracle_slow_channels.csv", index=False)
pd.DataFrame(prow).to_csv(OUT / "02_oracle_slow_pdist.csv", index=False)

# ---- D2 adim atfi (hangi adim en yuksek z'yi veriyor): oracle slow, ToF x tof ve ToF x pred
print("\n== 4b. Oracle slow: D2'yi belirleyen adim (en buyuk |z|) ==")
steprows = []
for seg in ("tof", "pred"):
    C, _ = ac.build_cycles("tof", seg, LAB["tof"])
    cl = C.to_dict("records")
    sc = cell("tof", seg)
    for f in range(ac.N_FOLDS):
        ro = fc.anom_roles(f)
        tr, ca, te = ac._split_roles(cl, ro)
        par = ac.fit_params(tr)
        for c in te:
            if c["atype"] != "slow":
                continue
            z = {k: abs(v - par.d2[k][0]) / par.d2[k][1] for k, v in ac.steps_of(c) if k in par.d2}
            top = max(z, key=z.get) if z else None
            row = sc[(sc.f == f) & (sc.session == c["session"]) & (sc.cycle_idx == c["cycle_idx"])]
            assert len(row) == 1
            row = row.iloc[0]
            assert abs((max(z.values()) if z else 0.0) - row.D2) < 1e-9, "D2 yeniden uretilemedi"
            steprows.append(dict(segmentation=seg, f=f, session=c["session"], cycle_idx=c["cycle_idx"], top_step=top,
                                 top_z=z[top] if top else 0.0, D2=row.D2, p2=row.p2, a2=bool(row.a2), a3=bool(row.a3),
                                 z_dwell1=z.get("dwell_1", np.nan), z_gap12=z.get("gap_1_2", np.nan), z_dwell2=z.get("dwell_2", np.nan),
                                 z_gap25=z.get("gap_2_5", np.nan), z_dwell5=z.get("dwell_5", np.nan)))
SR = pd.DataFrame(steprows)
SR.to_csv(OUT / "02_oracle_slow_steps.csv", index=False)
print("(D2 degerleri CSV ile birebir yeniden uretildi)")
for seg in ("tof", "pred"):
    q = SR[SR.segmentation == seg]
    t = pd.crosstab(q.top_step, q.a2.map({True: "D2 alarm", False: "D2 alarm yok"}))
    print(f"-- ToF x {seg}: en buyuk z veren adim x D2 alarmi (cift sayisi)")
    print(t.to_string())
    print(f"   adim bazinda z medyani (slow): " + ", ".join(
        f"{c}={q[c].median():.2f}" for c in ("z_dwell1", "z_gap12", "z_dwell2", "z_gap25", "z_dwell5")))

# ---- 4c. kalibrasyon kumesinin D2 uclari: slow tespiti kat bazinda hep-ya-da-hic mi?  (cal skorlari; alarm durumu DEGIL)
print("\n== 4c. Kat bazinda D2 kalibrasyon ustleri ve slow D2 skorlari ==")
crow = []
for src, seg in CELLS:
    d = S[(S.state_source == src) & (S.segmentation == seg)]
    for f in range(ac.N_FOLDS):
        cal = d[(d.role == "cal") & (d.f == f)]["D2"].to_numpy(float)
        cs = np.sort(cal)[::-1]
        te = cell(src, seg); te = te[te.f == f]
        for at in ("fast", "slow"):
            x = te[te.atype == at]
            if not len(x):
                continue
            crow.append(dict(state_source=src, segmentation=seg, atype=at, f=f, n_cal=len(cal), cal_D2_top1=cs[0], cal_D2_top2=cs[1],
                             cal_D2_top3=cs[2], min_p=1 / (len(cal) + 1), n_test=len(x), D2_med=float(x.D2.median()),
                             D2_q25=float(x.D2.quantile(.25)), D2_q75=float(x.D2.quantile(.75)),
                             n_above_top1=int((x.D2.to_numpy() >= cs[0] - 1e-12).sum()), n_above_top2=int((x.D2.to_numpy() >= cs[1] - 1e-12).sum()),
                             det_D2_a05=float(x.a2.mean()), det_D2_a10=float(x.b2_10.mean()), det_D3=float(x.a3.mean()),
                             det_D1=float(x.a1.mean())))
CF = pd.DataFrame(crow)
CF.to_csv(OUT / "02_tempo_perfold_cal.csv", index=False)
for src, seg in PRIMARY:
    print(f"-- {src} x {seg}")
    print(CF[(CF.state_source == src) & (CF.segmentation == seg)].drop(columns=["state_source", "segmentation"]).round(3).to_string(index=False))
print("-- ToF x pred / B2 x tof (slow)")
print(CF[(CF.atype == "slow") & (CF.state_source + CF.segmentation).isin(["tofpred", "b2tof"])].round(3).to_string(index=False))

# ---- 4d. kalibrasyondaki asiri D2 uclari kimin?  (yalniz cal-rolu skorlari; alarm durumu / natural_dev bakilmaz)
print("\n== 4d. ToF x tof kalibrasyon D2 asiri degerleri ==")
OUTL = []
dd = S[(S.state_source == "tof") & (S.segmentation == "tof") & (S.role == "cal")]
for f in range(ac.N_FOLDS):
    x = dd[dd.f == f].sort_values("D2", ascending=False)
    top = x.iloc[0]
    OUTL.append(dict(f=f, session=top.session, cycle_idx=int(top.cycle_idx), tof_fold=int(top.tof_fold), D2=float(top.D2), next_D2=float(x.iloc[1].D2),
                     dur_s=float(top.dur_s), visit_seq=top.visit_seq))
    print(f"  f={f}: en yuksek cal D2 = {top.D2:.3f} (dongu {top.session}/{int(top.cycle_idx)}, ToF kati {int(top.tof_fold)}, dur {top.dur_s:.2f} s, dizi {top.visit_seq}); "
          f"sonraki {x.iloc[1].D2:.3f}")
OUTL = pd.DataFrame(OUTL)
OUTL.to_csv(OUT / "02_oracle_cal_outliers.csv", index=False)

# ===================================================================== 5. markdown
print("\n== 5. rapor ==")


def md_table(df, cols, names=None):
    names = names or cols
    out = "| " + " | ".join(names) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return out


def pct(x):
    return f"{x:.3f}"


L = []
L.append("# A4.2 (M2) - Tempo tespitinin kanal dokumu ve oracle'in slow'daki davranisi (2026-11-05)\n")
L.append("Brief: `faz_a4/DESIGN_A4.md` (A4.2). Betik: `faz_a4/02_tempo_channels.py`. Girdi: `reports/faz_a/04_cycle_scores.csv.gz` (role = test).\n"
         "Birim = (dongu, f) cifti (Faz A 'tespit' tanimi; anomali donguleri her katta test). GA = dongu-kumeli bootstrap "
         f"(dongu basina 5 kat ortalamasi, {NB} tekrar); normal FAR icin Wilson. Kanal esigi: D3 alarmi (alpha = 0.10) <=> min(p1,p2) <= 0.05; "
         "yani 'D1 kanali' = p1 <= 0.05, 'D2 kanali' = p2 <= 0.05, D3 alarmi = bunlarin birlesimi (betikte dogrulandi, her satirda).\n"
         "Kontrol: a2_cycles ile ToF x tof ve B2 x pred donguleri yeniden kuruldu; CSV'deki visit_seq ve atype birebir ayni "
         + "; ".join(f"{s} x {g}: {k}/{n}" for s, g, n, k in check_rows) + ". Oracle slow icin D2 degerleri adim adim yeniden uretildi (fark < 1e-9).\n")

# --- 1
L.append("## 1. D3 alarmi alan fast/slow donguleri: hangi kanal?\n")
for src, seg in PRIMARY:
    nm = "B2 x pred" if src == "b2" else "ToF (oracle) x tof"
    L.append(f"### {nm}\n")
    rr = CH[(CH.state_source == src) & (CH.segmentation == seg)].copy()
    rr["tur"] = rr.atype.map({"fast": "fast", "slow": "slow", "tempo": "tempo (fast+slow)"})
    rr["n"] = rr.apply(lambda r: f"{int(r.n_pairs)} ({int(r.n_cycles)} dongu)", axis=1)
    rr["D3 alarm"] = rr.apply(lambda r: f"{int(r.n_D3)} ({r.D3alarm:.3f} [{r.D3alarm_lo:.3f}; {r.D3alarm_hi:.3f}])", axis=1)
    for c in ("D1_only", "D2_only", "both"):
        rr[c] = rr.apply(lambda r, c=c: f"{int(r['n_' + c])} ({100 * r['share_of_D3_' + c]:.1f}%)" if r.n_D3 else "-", axis=1)
    rr["alarm yok"] = rr.n_none
    L.append(md_table(rr, ["tur", "n", "D3 alarm", "D1_only", "D2_only", "both", "alarm yok"],
                      ["tur", "n cift", "D3 alarm (oran [GA])", "yalniz D1 (D3 alarmlarinin %'si)", "yalniz D2", "ikisi birden", "alarm yok"]))
    L.append("")
L.append("Diger iki hucre (ToF x pred, B2 x tof) icin ayni dokum: `02_tempo_channel_split.csv`.\n")

# --- 2
L.append("## 2. Fast donguler: kisalmis dizi (ziyaret dizisinde Box5 yok) ve D1 ile alarm\n")
for src, seg in PRIMARY:
    nm = "B2 x pred" if src == "b2" else "ToF (oracle) x tof"
    L.append(f"### {nm}\n")
    rr = TR[(TR.state_source == src) & (TR.segmentation == seg) & TR.atype.isin(["fast", "slow", "normal"])].copy()
    rr["n"] = rr.apply(lambda r: f"{int(r.n_pairs)} ({int(r.n_cycles)} dongu)", axis=1)
    for c in ("D1alarm", "D2alarm", "D3alarm"):
        rr[c + "_s"] = rr.apply(lambda r, c=c: f"{int(r['k_' + c])}/{int(r.n_pairs)} = {r[c]:.3f}" if r.n_pairs else "-", axis=1)
    rr["trunc"] = rr.apply(lambda r: f"{r.share_trunc_pairs:.3f}" if not np.isnan(r.share_trunc_pairs) else "", axis=1)
    L.append(md_table(rr, ["atype", "grup", "n", "trunc", "D1alarm_s", "D2alarm_s", "D3alarm_s"],
                      ["tur", "grup", "n cift", "kisalmis payi", "D1 alarm (p1<=0.05)", "D2 alarm (p2<=0.05)", "D3 alarm"]))
    L.append("")
L.append("`kisalmis payi` yalniz 'tum' satirinda: tur icindeki (dongu, f) ciftlerinin Box5'siz olan kismi. Hem normal hem anomali icin "
         "Wilson GA'li tum oranlar: `02_tempo_truncation.csv` (seq135/seq145 dahil; diger hucreler dahil).\n")

# --- 3
L.append("## 3. Onceden yazilmis ikincil tempo tanimi: D2 tek basina, alpha = 0.10 ('tempo kanali')\n")
L.append("Makalede 'tempo deviations detected' bu sayiyla nitelenir. Alarm = p2 <= 0.10 (D3'un Bonferroni esiginden farkli: alpha bolunmez). "
         "Referans olarak D2 tek basina alpha = 0.05 ve D3 alpha = 0.10 de verilir.\n")
sec = []
for src, seg in CELLS:
    r = SEC[(SEC.state_source == src) & (SEC.segmentation == seg)].iloc[0]
    nm = {"b2pred": "B2 x pred", "toftof": "ToF x tof", "tofpred": "ToF x pred", "b2tof": "B2 x tof"}[src + seg]
    for at in ("fast", "slow", "tempo"):
        sec.append(dict(hucre=nm, tur=at,
                        d2=f"{r[f'det_{at}_D2_a10']:.3f} [{r[f'det_{at}_D2_a10_lo']:.3f}; {r[f'det_{at}_D2_a10_hi']:.3f}]",
                        d2b=f"{r[f'det_{at}_D2_a05']:.3f}",
                        d3=f"{r[f'det_{at}_D3_a10']:.3f} [{r[f'det_{at}_D3_a10_lo']:.3f}; {r[f'det_{at}_D3_a10_hi']:.3f}]"))
L.append(md_table(pd.DataFrame(sec), ["hucre", "tur", "d2", "d2b", "d3"],
                  ["hucre", "tur", "D2 tek basina a=0.10 (tespit [GA])", "D2 tek basina a=0.05", "D3 a=0.10 (Faz A tespit)"]))
L.append("")
far = []
for src, seg in CELLS:
    r = SEC[(SEC.state_source == src) & (SEC.segmentation == seg)].iloc[0]
    nm = {"b2pred": "B2 x pred", "toftof": "ToF x tof", "tofpred": "ToF x pred", "b2tof": "B2 x tof"}[src + seg]
    far.append(dict(h=nm, n=int(r.nNormal), a=f"{r.FAR_D2_a10:.3f} [{r.FAR_D2_a10_wlo:.3f}; {r.FAR_D2_a10_whi:.3f}]",
                    c=f"{r.FAR_D2_a05:.3f}", d=f"{r.FAR_D3_a10:.3f} [{r.FAR_D3_a10_wlo:.3f}; {r.FAR_D3_a10_whi:.3f}]"))
L.append("Normal (yanlis alarm) karsiligi, ayni test ciftleri uzerinden (Wilson %95 GA):\n")
L.append(md_table(pd.DataFrame(far), ["h", "n", "a", "c", "d"],
                  ["hucre", "n normal", "FAR D2 tek basina a=0.10", "FAR D2 a=0.05", "FAR D3 a=0.10"]))
L.append("")

# --- 4
L.append("## 4. Oracle (ToF x tof) slow'da neden ~%60'ta kaliyor? (betimleyici)\n")
q = cell("tof", "tof")
qs = q[q.atype == "slow"]
oo = pd.DataFrame(orows)
for seg in ("tof", "pred"):
    L.append(f"### ToF x {seg}" + (" (brief'in hucresi)" if seg == "tof" else " (ek: ayni oracle etiketler, tahmin-segmentasyonu)") + "\n")
    o = oo[oo.segmentation == seg].copy()
    o["oran"] = o.apply(lambda r: f"{int(r.k)}/{int(r.n)} = {r.rate:.3f} [{r.lo:.3f}; {r.hi:.3f}]", axis=1)
    o["nf"] = o.normal_FAR.map(pct)
    L.append(md_table(o, ["olcut", "oran", "nf"], ["olcut", "slow tespit [GA]", "normal FAR"]))
    L.append("")
    pp = pd.DataFrame(prow)
    pp = pp[pp.segmentation == seg].copy()
    for c in ("min", "q10", "q25", "median", "q75", "q90", "max"):
        pp[c] = pp[c].map(lambda x: f"{x:.3f}")
    L.append("Slow ciftlerinde p dagilimi (p1, p2, p3 = min(1, 2 min(p1,p2))). Alarm esikleri: p1, p2 icin <= 0.05; p3 icin <= 0.10 (yani p3 icin alarm kutusu (.05,.10] sutunu ve altidir; <=.05 sutunu p3 icin anlamsizdir):\n")
    L.append(md_table(pp, ["p", "n", "min", "q10", "q25", "median", "q75", "q90", "max", "n_le05", "n_05_10", "n_10_20", "n_20_50", "n_gt50", "n_eq1"],
                      ["p", "n", "min", "q10", "q25", "medyan", "q75", "q90", "max", "<=.05", "(.05,.10]", "(.10,.20]", "(.20,.50]", ">.50", "p=1"]))
    L.append("")
    sr = SR[SR.segmentation == seg]
    t = pd.crosstab(sr.top_step, sr.a2.map({True: "D2 alarm", False: "D2 alarm yok"}))
    for c in ("D2 alarm", "D2 alarm yok"):
        if c not in t:
            t[c] = 0
    t = t.reset_index().rename(columns={"top_step": "D2'yi belirleyen adim"})
    L.append("D2'yi belirleyen adim (slow ciftleri, en buyuk |z|), D2 alarmina gore:\n")
    L.append(md_table(t, ["D2'yi belirleyen adim", "D2 alarm", "D2 alarm yok"]))
    L.append("")
L.append("Betimleyici ayrintilar (kat bazinda tespit, dongu bazinda tutarlilik, alarm/kacirilan donguler, sure ve skor medyanlari) stdout'tadir "
         "ve asagida ozetlenmistir.\n")

# alt-bolum: stdout'tan ozet icin yeniden hesapla
q = cell("tof", "tof")
qs = q[q.atype == "slow"]
pf = qs.groupby("f")[["a1", "a2", "a3"]].mean()
L.append("Kat bazinda slow tespiti (ToF x tof; D1 / D2 / D3): " + "; ".join(
    f"f={int(f)}: {r.a1:.2f} / {r.a2:.2f} / {r.a3:.2f}" for f, r in pf.iterrows()) + ".\n")
pc = qs.groupby(["session", "cycle_idx"])["a3"].agg(["sum", "count"])
L.append(f"Dongu bazinda D3: {int((pc['sum'] == pc['count']).sum())} dongu 5/5 katta alarm, {int((pc['sum'] == 0).sum())} dongu hic alarm yok, "
         f"{int(((pc['sum'] > 0) & (pc['sum'] < pc['count'])).sum())} dongu karisik (toplam {len(pc)} slow dongusu).\n")
nr = q[q.atype == "normal"]
for nm, qq in (("alarm (D3)", qs[qs.a3]), ("kacirilan (D3)", qs[~qs.a3])):
    L.append(f"- {nm}: n = {len(qq)} cift; dongu suresi medyan {qq.dur_s.median():.2f} s [IQR {qq.dur_s.quantile(.25):.2f}; {qq.dur_s.quantile(.75):.2f}]; "
             f"D0 medyan {qq.D0.median():.2f}, D1 medyan {qq.D1.median():.2f}, D2 medyan {qq.D2.median():.2f}; "
             f"ziyaret dizisi {qq.visit_seq.value_counts().head(3).to_dict()}.")
L.append(f"- normal (referans): dongu suresi medyan {nr.dur_s.median():.2f} s [IQR {nr.dur_s.quantile(.25):.2f}; {nr.dur_s.quantile(.75):.2f}]; "
         f"D0 medyan {nr.D0.median():.2f}, D2 medyan {nr.D2.median():.2f}.\n")
L.append("### Kat bazinda: slow tespiti kalibrasyon kumesine bagli (D2 skoru ve kalibrasyon ustleri)\n")
L.append("D2 alarmi p2 <= 0.05 gerektirir; n_cal ~ 21 oldugundan bu, test D2 skorunun o kattaki kalibrasyon kumesinin **en yuksek** D2'sini asmasi demektir "
         "(en kucuk p = 1/(n_cal+1)). Slow icin kat bazinda (D2 skoru medyani, kalibrasyon ilk uc D2, en yuksegi asan cift sayisi, tespit):\n")
for src, seg in PRIMARY:
    nm = "B2 x pred" if src == "b2" else "ToF (oracle) x tof"
    cf = CF[(CF.state_source == src) & (CF.segmentation == seg)].copy()
    for at in ("slow", "fast"):
        c2 = cf[cf.atype == at].copy()
        c2["a"] = c2.apply(lambda r: f"{r.D2_med:.2f} [{r.D2_q25:.2f}; {r.D2_q75:.2f}]", axis=1)
        c2["b"] = c2.apply(lambda r: f"{r.cal_D2_top1:.2f} / {r.cal_D2_top2:.2f} / {r.cal_D2_top3:.2f}", axis=1)
        c2["c"] = c2.apply(lambda r: f"{int(r.n_above_top1)}/{int(r.n_test)}", axis=1)
        for k in ("det_D1", "det_D2_a05", "det_D2_a10", "det_D3"):
            c2[k] = c2[k].map(lambda x: f"{x:.2f}")
        L.append(f"**{nm}, {at}**\n")
        L.append(md_table(c2, ["f", "n_cal", "a", "b", "c", "det_D1", "det_D2_a05", "det_D3", "det_D2_a10"],
                          ["f", "n_cal", "test D2 medyan [IQR]", "cal D2 ilk 3", "D2 >= cal max", "D1 (p1<=.05)", "D2 (p2<=.05)", "D3 a=0.10", "D2 tek basina a=0.10"]))
        L.append("")
ol = OUTL.copy()
L.append("ToF x tof: her katin kalibrasyon kumesindeki en yuksek D2 ve sahibi dongu (yalniz kalibrasyon-rolu skorlari; alarm durumuna bakilmadi):\n")
L.append(md_table(ol.round(3), ["f", "session", "cycle_idx", "tof_fold", "D2", "next_D2", "dur_s", "visit_seq"],
                  ["f", "oturum", "dongu", "ToF kati", "en yuksek cal D2", "sonraki cal D2", "dongu suresi (s)", "dizi"]))
L.append("")
L.append("**Betimleyici ozet (oracle slow):** ToF x tof slow tespiti 0.600 = {f=0,1,4: 52/52 cift} + {f=2,3: 0/52}. f=2 ve f=3 kalibrasyon kumelerinin ikisi de ayni tek normal dongu icerir (session_02 dongu 51, 25.0 s, D2 = 11.3 / 11.1; sonraki en yuksek cal D2 4.0 / 5.0). p2 <= 0.05 icin D2 >= kalibrasyon maksimumu gerekir; bu iki katta hicbir slow dongusu 11'i asmaz (n_above_top1 = 0/52), ikinci en yuksegi asan 46/52 ve 27/52 cift vardir (D2 tek basina alpha = 0.10: 0.885 ve 0.519). Diger uc katta ilk uc cal D2 2.0-2.8 ve butun slow donguleri bunu asar. Yani %60, slow sinyalinin zayifligindan degil, **iki kalibrasyon kumesindeki tek bir uc degerden** gelir (tek ornek; olcut: n_cal ~ 20, D3 'tum kalibrasyon skorlarini asma' kurali).\n")
L.append("Not: slow/fast anomali donguleri her katta test oldugu icin, bir katin kalibrasyon kumesi (20 civari normal dongu) o kattaki tespit oranini tek basina belirler; "
         "bu yuzden dongu-kumeli bootstrap GA'si bu degiskenligi (kalibrasyon kumesi degiskenligini) YANSITMAZ, yalniz dongulerin ornekleme degiskenligini yansitir. "
         "Oracle slow'da D3 = 0.600 [0.600; 0.600] bu durumun uc ornegidir: her dongu 3/5 katta alarm verir, bootstrap'in varyansi sifirdir. "
         "Gercek belirsizlik kat-duzeyidir (5 kat); kat bazinda aralik tablolarda.\n")
L.append("**Not (betimleyici, nedensellik test edilmedi):** yukaridaki sayilar slow kacirmalarinin hangi kanalda ve hangi p araliginda "
         "yigildigini gosterir; 'neden' sorusuna tek bir mekanizma onerilmez.\n")

(OUT / "02_tempo_channels.md").write_text("\n".join(L), encoding="utf-8")
print("yazildi:", OUT / "02_tempo_channels.md")

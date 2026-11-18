"""Faz A / 04 - DONGU DUZEYI ANOMALI PROTOKOLU (DESIGN.md "Anomali tespiti").

Girdi : frame_table() (ToF oracle = y_true), reports/faz_a/oof/{b1,b2,cnngru,cnnonly}__S1__f*.csv.gz (y_pred),
        isteğe bagli reports/faz_a/05_old_cycle_scores.csv.gz (eski boru hatti; asagidaki sozlesme).
Cikti : reports/faz_a/04_anomaly.md, 04_anomaly_table.csv, 04_anomaly_perfold.csv, 04_cycle_scores.csv.gz, 04_cycle_counts.csv

CYCLE KEY (stage 05 ile hizalama): (state_source, segmentation, session, cycle_idx, start_frame_id)
  state_source in {tof, cnngru, cnnonly, b1, b2};  segmentation in {pred, tof}
  segmentation == "tof"  : cycle_idx = ToF loop_id,                start_frame_id = loop'un ilk kare frame_id'si
  segmentation == "pred" : cycle_idx = oturumdaki (debounce sonrasi) tahmini Box1 girisinin 0-tabanli sirasi,
                           start_frame_id = Box1 girisinin ilk kare frame_id'si; dongu bir sonraki Box1 girisine kadar.
  04_cycle_scores.csv.gz satirlari (key, f, role) ayrintisindadir; her f icin normal s02 donguleri tam bir rol alir
  (train/cal/test), anomali (s03-s06) donguleri her f'de 'test'. 05'in yazacagi dosya:
  05_old_cycle_scores.csv.gz : key sutunlari + f + old_tft (max |delta|, Denk. 9) + old_z (max |Z|); (cal ve test rolleri yeterli)
"""
import sys
import math
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probe"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

import fa_common as fc
from common import md_table

DT = 0.1                     # kare periyodu (s)
MIN_BOX = 2                  # debounce: kutu run'i >= 2 kare
MAX_GAP = 1                  # debounce: <= 1 karelik none boslugu ayni kutular arasinda birlestirilir
EPS = 0.05                   # log(d + EPS): sifir uzunluklu gecis icin
SIG_MIN = 0.05               # log-olcekte sigma tabani
MIN_REF = 3                  # D2: bir adimin referansi icin en az egitim gozlemi
ALPHAS = (0.10, 0.05)
THR_TFT, THR_Z = 0.5, 1.0    # eski sabit esikler
USED = {"normal_crossfit", "anomaly_test_all_folds"}
ATYPES = ["seq135", "seq145", "fast", "slow"]
AGROUPS = {"seq135": ["seq135"], "seq145": ["seq145"], "fast": ["fast"], "slow": ["slow"],
           "seq(135+145)": ["seq135", "seq145"], "tempo(fast+slow)": ["fast", "slow"],
           "tum anomali": ["seq135", "seq145", "fast", "slow"]}
SRC = {"tof": "ToF oracle", "cnngru": "CNN-GRU", "cnnonly": "CNN-only", "b1": "B1", "b2": "B2"}
SEGN = {"pred": "tahmini Box1 girisleri", "tof": "ToF dongu sinirlari"}
VNAME = {"D0": "D0 sure", "D1": "D1 Markov", "D2": "D2 adim-suresi", "D3": "D3 min-p",
         "OLD_tft": "OLD TFT", "OLD_z": "OLD Z", "OLD3": "OLD min-p"}

LINES = []


def out(s=""):
    LINES.append(s)
    print(s)


def sd(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def mean(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(a.mean()) if len(a) else float("nan")


def ms(a, d=3):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if not len(a):
        return "nan"
    return f"{a.mean():.{d}f} ± {(a.std(ddof=1) if len(a) > 1 else float('nan')):.{d}f}"


# ------------------------------------------------------------------ frame table / loops
ft = fc.frame_table()
assert ft.groupby("session")["frame_id"].apply(lambda s: bool((np.diff(s.to_numpy()) > 0).all())).all()
AL = fc.anom_loops()
SESS = [s for s in sorted(ft["session"].unique()) if s != "session_01_training"]
OFF = {s: int(ft.index[ft["session"] == s][0]) for s in SESS}
assert all((ft["row"].to_numpy()[ft["session"] == s] == np.arange(len(ft))[ft["session"] == s]).all() for s in SESS)
SE = {s: ft[ft["session"] == s].reset_index(drop=True) for s in SESS}          # per-session frames
ALX = {(r.session, int(r.loop_id)): r for r in AL.itertuples(index=False)}


# ------------------------------------------------------------------ state sources
def runs_of(lab):
    lab = np.asarray(lab)
    if len(lab) == 0:
        return []
    idx = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    ends = np.r_[idx[1:], len(lab)]
    return [(int(lab[i]), int(i), int(e)) for i, e in zip(idx, ends)]


def debounce_labels(lab, min_box=MIN_BOX, max_gap=MAX_GAP):
    """Kare sayisi uzerinden: (1) ayni kutu arasindaki <= max_gap karelik none boslugunu birlestir;
    (2) < min_box karelik kutu run'ini none yap; (1)-(2) sabit noktaya kadar tekrar edilir."""
    lab = np.asarray(lab).copy()
    for _ in range(50):
        changed = False
        r = runs_of(lab)
        for i in range(1, len(r) - 1):
            l, s, e = r[i]
            if l == 0 and (e - s) <= max_gap and r[i - 1][0] == r[i + 1][0] and r[i - 1][0] > 0:
                lab[s:e] = r[i - 1][0]
                changed = True
        for l, s, e in runs_of(lab):
            if l > 0 and (e - s) < min_box:
                lab[s:e] = 0
                changed = True
        if not changed:
            break
    return lab


def source_labels():
    """{source: {session: int array of per-frame labels}}; frames without a window prediction (first 4 of each session) -> none."""
    res = {"tof": {s: SE[s]["y_true"].to_numpy().astype(int) for s in SESS}}
    for m in ("cnngru", "cnnonly", "b1", "b2"):
        oof = fc.load_oof(m, "S1")
        if len(oof) == 0:
            print(f"[uyari] {m}: S1 OOF yok -> atlandi")
            continue
        nf = oof["fold"].nunique()
        if len(oof) != 16248 or nf != 5:
            print(f"[uyari] {m}: S1 OOF eksik/yarim (n={len(oof)}, kat={nf}) -> atlandi")
            continue
        mm = ft[["session", "frame_id"]].merge(oof[["session", "frame_id", "y_pred"]], on=["session", "frame_id"],
                                                how="left", validate="1:1")
        miss = int(mm["y_pred"].isna().sum())
        print(f"[bilgi] {m}: kare basina tahmin; pencere olmayan (none sayilan) kare = {miss} (beklenen 6*4=24)")
        y = mm["y_pred"].fillna(0).astype(int).to_numpy()
        res[m] = {s: y[ft["session"].to_numpy() == s] for s in SESS}
    return res


# ------------------------------------------------------------------ cycles
def visits_of(lab, tt):
    n = len(lab)
    vs = []
    for l, s, e in runs_of(lab):
        if l > 0:
            en = tt[e] if e < n else tt[n - 1] + DT
            vs.append((l, s, e, float(tt[s]), float(en)))
    return vs


def cycle_from(vs):
    seq = tuple(v[0] for v in vs)
    dw = [v[4] - v[3] for v in vs]
    gp = [max(0.0, vs[i + 1][3] - vs[i][4]) for i in range(len(vs) - 1)]
    return seq, dw, gp


def build_cycles(source, seg, labs):
    rows, cnt = [], []
    for s in SESS:
        df = SE[s]
        tt = df["t"].to_numpy()
        fid = df["frame_id"].to_numpy()
        lid = df["loop_id"].to_numpy()
        lab = debounce_labels(labs[s])
        vs = visits_of(lab, tt)
        vstart = np.array([v[1] for v in vs], dtype=int)
        n_ref = int(((AL["session"] == s) & AL["use"].isin(USED)).sum())
        dropped_unm = dropped_open = n_entries = 0
        if seg == "tof":
            for a in AL[(AL["session"] == s) & AL["use"].isin(USED)].itertuples(index=False):
                i0, i1 = int(a.first_row) - OFF[s], int(a.last_row) - OFF[s]
                sel = [vs[k] for k in np.flatnonzero((vstart >= i0) & (vstart <= i1))]
                seq, dw, gp = cycle_from(sel)
                rows.append(dict(state_source=source, segmentation=seg, session=s, cycle_idx=int(a.loop_id),
                                 start_frame_id=int(fid[i0]), end_frame_id=int(fid[i1]), start_idx=i0, end_idx=i1 + 1,
                                 dur_s=float(tt[i1] + DT - tt[i0]), anchor_loop=int(a.loop_id), atype=a.scenario,
                                 fold=int(a.fold), natural_dev=bool(a.natural_dev), seq=seq, dw=dw, gp=gp))
        else:
            ent = [k for k, v in enumerate(vs) if v[0] == 1]
            n_entries = len(ent)
            for j, k in enumerate(ent):
                if j + 1 >= len(ent):
                    dropped_open += 1                         # son (kapanmamis) dongu
                    continue
                k2 = ent[j + 1]
                i0, i1 = vs[k][1], vs[k2][1]                  # [i0, i1) ; son kare i1-1
                lp = int(lid[i0])
                a = ALX.get((s, lp))
                if a is None or a.use not in USED:
                    dropped_unm += 1
                    continue
                seq, dw, gp = cycle_from(vs[k:k2])
                rows.append(dict(state_source=source, segmentation=seg, session=s, cycle_idx=j,
                                 start_frame_id=int(fid[i0]), end_frame_id=int(fid[i1 - 1]), start_idx=i0, end_idx=i1,
                                 dur_s=float(tt[i1] - tt[i0]), anchor_loop=lp, atype=a.scenario,
                                 fold=int(a.fold), natural_dev=bool(a.natural_dev), seq=seq, dw=dw, gp=gp))
        c = [r for r in rows if r["session"] == s]
        per_loop = Counter(r["anchor_loop"] for r in c)
        used_loops = AL[(AL["session"] == s) & AL["use"].isin(USED)]["loop_id"].astype(int).tolist()
        k = [per_loop.get(l, 0) for l in used_loops]
        cnt.append(dict(state_source=source, segmentation=seg, session=s, atype=(c[0]["atype"] if c else ""), n_tof_ref=n_ref, n_pred_box1_entries=n_entries if seg == "pred" else np.nan,
                        n_used=len(c), n_dropped_unmapped_loop=dropped_unm, n_dropped_open_last=dropped_open,
                        n_natural_dev=int(sum(r["natural_dev"] for r in c)),
                        n_loops_with_0=int(sum(1 for x in k if x == 0)), n_loops_with_1=int(sum(1 for x in k if x == 1)),
                        n_loops_with_2plus=int(sum(1 for x in k if x >= 2))))
    return pd.DataFrame(rows), cnt


# ------------------------------------------------------------------ scores
def d0_fit(tr):
    x = np.log(np.array([c["dur_s"] for c in tr], float))
    return float(x.mean()), float(max(x.std(ddof=1), SIG_MIN))


def d0_score(c, par):
    return abs(math.log(c["dur_s"]) - par[0]) / par[1]


def d1_fit(tr):
    cnt = {k: Counter() for k in ("S", 1, 2, 3, 4, 5)}
    for c in tr:
        path = ["S"] + list(c["seq"]) + ["E"]
        for a, b in zip(path[:-1], path[1:]):
            cnt[a][b] += 1
    return cnt


def d1_score(c, cnt):
    path = ["S"] + list(c["seq"]) + ["E"]
    return max(-math.log((cnt[a][b] + 1) / (sum(cnt[a].values()) + 6)) for a, b in zip(path[:-1], path[1:]))


def steps_of(c):
    st = [(f"dwell_{b}", math.log(d + EPS)) for b, d in zip(c["seq"], c["dw"])]
    st += [(f"gap_{a}_{b}", math.log(g + EPS)) for a, b, g in zip(c["seq"][:-1], c["seq"][1:], c["gp"])]
    return st


def d2_fit(tr):
    acc = {}
    for c in tr:
        for k, v in steps_of(c):
            acc.setdefault(k, []).append(v)
    return {k: (float(np.mean(v)), float(max(np.std(v, ddof=1), SIG_MIN)), len(v)) for k, v in acc.items() if len(v) >= MIN_REF}


def d2_score(c, ref):
    z = [abs(v - ref[k][0]) / ref[k][1] for k, v in steps_of(c) if k in ref]
    return (max(z) if z else 0.0), len(z)


def conf_p(cal_scores, s):
    cs = np.asarray(cal_scores, float)
    s = np.asarray(s, float)
    return (1.0 + (cs[None, :] >= s[:, None] - 1e-12).sum(axis=1)) / (len(cs) + 1.0)


def score_group(C):
    """Her f icin D0/D1/D2 + conformal p; satir = (dongu, f)."""
    res = []
    ncal_log = []
    cl = C.to_dict("records")
    for f in range(fc.N_FOLDS):
        ro = fc.anom_roles(f)
        tr = [c for c in cl if c["atype"] == "normal" and c["fold"] in ro["train"]]
        ca = [c for c in cl if c["atype"] == "normal" and c["fold"] in ro["cal"]]
        te = [c for c in cl if (c["atype"] == "normal" and c["fold"] == f) or c["atype"] != "normal"]
        if len(tr) < 3 or len(ca) < 3:
            print(f"[uyari] f={f}: egitim={len(tr)} kal={len(ca)} yetersiz")
            continue
        p0, p1, p2 = d0_fit(tr), d1_fit(tr), d2_fit(tr)

        def sc(c, role):
            s2, nref = d2_score(c, p2)
            return dict(state_source=c["state_source"], segmentation=c["segmentation"], session=c["session"],
                        cycle_idx=c["cycle_idx"], start_frame_id=c["start_frame_id"], end_frame_id=c["end_frame_id"],
                        f=f, role=role, atype=c["atype"], natural_dev=c["natural_dev"], tof_fold=c["fold"],
                        anchor_loop=c["anchor_loop"], dur_s=c["dur_s"], n_visits=len(c["seq"]),
                        visit_seq="-".join(map(str, c["seq"])), D0=d0_score(c, p0), D1=d1_score(c, p1), D2=s2, d2_nref=nref)
        rtr = [sc(c, "train") for c in tr]
        rca = [sc(c, "cal") for c in ca]
        rte = [sc(c, "test") for c in te]
        d = pd.DataFrame(rtr + rca + rte)
        cal = d[d["role"] == "cal"]
        ncal_log.append(dict(f=f, n_cal=len(cal), n_train=len(tr)))
        for k, nm in (("0", "D0"), ("1", "D1"), ("2", "D2")):
            m = d["role"] == "test"
            d.loc[m, "p" + k] = conf_p(cal[nm].to_numpy(), d.loc[m, nm].to_numpy())
        d["p3"] = np.minimum(1.0, 2.0 * d[["p1", "p2"]].min(axis=1))
        res.append(d)
    return pd.concat(res, ignore_index=True), ncal_log


# ------------------------------------------------------------------ variants / evaluation
def add_old(S):
    """Eski boru hatti skorlarini (varsa) birlestir; conformal p'leri ayni cal protokoluyle hesapla."""
    p = fc.OUT / "05_old_cycle_scores.csv.gz"
    if not p.exists():
        return S, False
    old = pd.read_csv(p)
    key = ["state_source", "segmentation", "session", "cycle_idx", "start_frame_id", "f"]
    need = key + ["old_tft", "old_z"]
    if not set(need) <= set(old.columns):
        print(f"[uyari] {p.name}: beklenen sutunlar yok ({need}) -> atlandi")
        return S, False
    old = old[need].drop_duplicates(key)
    S = S.merge(old, on=key, how="left", validate="m:1")
    nm = int(S.loc[S["role"] == "test", "old_tft"].notna().sum())
    print(f"[bilgi] OLD birlesti: test satiri eslesen = {nm}/{int((S['role'] == 'test').sum())}")
    if nm == 0:
        return S, False
    for (sr, sg, f), idx in S.groupby(["state_source", "segmentation", "f"]).groups.items():
        d = S.loc[idx]
        cal = d[(d["role"] == "cal")]
        te = d[(d["role"] == "test") & d["old_tft"].notna()]
        for col, pc in (("old_tft", "p_otft"), ("old_z", "p_oz")):
            cs = cal[col].dropna().to_numpy()
            if len(cs) >= 3 and len(te):
                S.loc[te.index, pc] = conf_p(cs, te[col].to_numpy())
    S["p_o3"] = np.minimum(1.0, 2.0 * S[["p_otft", "p_oz"]].min(axis=1, skipna=False))
    S["fx_tft"] = np.where(S["old_tft"].notna(), (S["old_tft"] > THR_TFT).astype(float), np.nan)
    S["fx_z"] = np.where(S["old_z"].notna(), (S["old_z"] > THR_Z).astype(float), np.nan)
    S["fx_or"] = np.where(S["old_tft"].notna() & S["old_z"].notna(), ((S["old_tft"] > THR_TFT) | (S["old_z"] > THR_Z)).astype(float), np.nan)
    return S, True


def variants(has_old):
    v = {"D0": ("D0", "p0", None), "D1": ("D1", "p1", None), "D2": ("D2", "p2", None), "D3": (None, "p3", None)}
    if has_old:
        v.update({"OLD_tft": ("old_tft", "p_otft", "fx_tft"), "OLD_z": ("old_z", "p_oz", "fx_z"), "OLD3": (None, "p_o3", "fx_or")})
    return v


def vscore(d, sc, pc):
    return d[sc].to_numpy(float) if sc else (1.0 - d[pc].to_numpy(float))


def auc_ap(neg, pos):
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    if len(neg) < 2 or len(pos) < 1:
        return float("nan"), float("nan")
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def evaluate(S, has_old):
    V = variants(has_old)
    perfold, summ = [], []
    T = S[S["role"] == "test"]
    for (sr, sg), g in T.groupby(["state_source", "segmentation"], sort=False):
        for vn, (sc, pc, fx) in V.items():
            if pc not in g or g[pc].notna().sum() == 0:
                continue
            for f, gf in g.groupby("f"):
                neg_d = gf[gf["atype"] == "normal"]
                for an, members in AGROUPS.items():
                    pos_d = gf[gf["atype"].isin(members)]
                    if vn.startswith("OLD") and (neg_d[pc].isna().all() or pos_d[pc].isna().all()):
                        continue
                    a, ap = auc_ap(vscore(neg_d, sc, pc), vscore(pos_d, sc, pc))
                    perfold.append(dict(state_source=sr, segmentation=sg, variant=vn, atype=an, f=f,
                                        n_neg=len(neg_d), n_pos=len(pos_d), auroc=a, auprc=ap,
                                        prevalence=len(pos_d) / max(1, len(pos_d) + len(neg_d))))
            # pooled FAR / detection
            neg = g[(g["atype"] == "normal") & g[pc].notna()]
            alarms = {f"a{int(a * 100):02d}": (lambda d, a=a: (d[pc].to_numpy(float) <= a + 1e-9)) for a in ALPHAS}
            if fx:
                alarms["fixed"] = lambda d: d[fx].to_numpy(float) > 0.5
            for an in ["normal"] + list(AGROUPS):
                d = neg if an == "normal" else g[g["atype"].isin(AGROUPS[an]) & g[pc].notna()]
                row = dict(state_source=sr, segmentation=sg, variant=vn, atype=an, n_pairs=len(d))
                for ak, fn in alarms.items():
                    if len(d) == 0:
                        row[f"rate_{ak}"] = float("nan")
                        continue
                    row[f"rate_{ak}"] = float(fn(d).mean())
                if an == "normal":
                    nonat = d[~d["natural_dev"]]
                    row["n_natdev"] = int(d["natural_dev"].sum())
                    for ak, fn in alarms.items():
                        row[f"rate_excl_nat_{ak}"] = float(fn(nonat).mean()) if len(nonat) else float("nan")
                summ.append(row)
    return pd.DataFrame(perfold), pd.DataFrame(summ)


# ------------------------------------------------------------------ main
LAB = source_labels()
SOURCES = list(LAB)
cycles_all, counts_all, scores_all, ncal_all = [], [], [], []
for src in SOURCES:
    for seg in ("pred", "tof"):
        C, cnt = build_cycles(src, seg, LAB[src])
        counts_all += cnt
        cycles_all.append(C)
        Sg, ncl = score_group(C)
        scores_all.append(Sg)
        for x in ncl:
            x.update(state_source=src, segmentation=seg)
        ncal_all += ncl
CNT = pd.DataFrame(counts_all)
S = pd.concat(scores_all, ignore_index=True)
NCAL = pd.DataFrame(ncal_all)
S, HAS_OLD = add_old(S)
PF, SM = evaluate(S, HAS_OLD)
CNT.to_csv(fc.OUT / "04_cycle_counts.csv", index=False)
S.to_csv(fc.OUT / "04_cycle_scores.csv.gz", index=False, compression="gzip")
PF.to_csv(fc.OUT / "04_anomaly_perfold.csv", index=False)

# ---- tidy summary table: AUROC/AUPRC fold-wise mean +- sd, pooled rates
rows = []
V = variants(HAS_OLD)
for (sr, sg, vn, an), g in PF.groupby(["state_source", "segmentation", "variant", "atype"], sort=False):
    r = dict(state_source=sr, segmentation=sg, variant=vn, atype=an, n_folds=int(g["auroc"].notna().sum()),
             auroc_mean=mean(g["auroc"]), auroc_sd=sd(g["auroc"]), auprc_mean=mean(g["auprc"]), auprc_sd=sd(g["auprc"]),
             prevalence=mean(g["prevalence"]), n_neg_per_fold=mean(g["n_neg"]), n_pos_per_fold=mean(g["n_pos"]))
    rows.append(r)
TAB = pd.DataFrame(rows)
SMg = SM.copy()
farcols = [c for c in SMg.columns if c.startswith("rate_") or c in ("n_pairs", "n_natdev")]
nm = SMg[SMg["atype"] == "normal"].rename(columns={c: ("far_" + c[5:] if c.startswith("rate_") else "far_" + c) for c in farcols})
nm = nm.drop(columns=["atype"])
an_ = SMg[SMg["atype"] != "normal"].rename(columns={"n_pairs": "det_n_pairs", **{c: "det_" + c[5:] for c in farcols if c.startswith("rate_")}})
an_ = an_[[c for c in an_.columns if "excl_nat" not in c and c != "n_natdev"]]
TAB = TAB.merge(an_, on=["state_source", "segmentation", "variant", "atype"], how="left")
TAB = TAB.merge(nm, on=["state_source", "segmentation", "variant"], how="left")
for a in ALPHAS:
    k = f"a{int(a * 100):02d}"
    TAB[f"far_se_{k}"] = np.sqrt(a * (1 - a) / TAB["far_n_pairs"])
    TAB[f"far_within2se_{k}"] = (TAB[f"far_{k}"] - a).abs() <= 2 * TAB[f"far_se_{k}"]
TAB.to_csv(fc.OUT / "04_anomaly_table.csv", index=False)


# ------------------------------------------------------------------ report
def srcs_present():
    return [s for s in ("tof", "cnngru", "cnnonly", "b1", "b2") if s in SOURCES]


def cell(sr, sg, vn, an, col):
    g = TAB[(TAB.state_source == sr) & (TAB.segmentation == sg) & (TAB.variant == vn) & (TAB.atype == an)]
    return float(g[col].iloc[0]) if len(g) else float("nan")


def pf_vals(sr, sg, vn, an, col="auroc"):
    g = PF[(PF.state_source == sr) & (PF.segmentation == sg) & (PF.variant == vn) & (PF.atype == an)].sort_values("f")
    return g[col].to_numpy(float), g["f"].to_numpy()


out("# 04 — Dongu duzeyi anomali protokolu (Faz A)")
out()
out("`faz_a/04_cycle_anomaly.py` tarafindan uretilir; DESIGN.md 'Anomali tespiti' protokolu. Tum sayilar stdout'a da yazilir. Sapma yok (asagidaki 'Kesinlestirilen parametreler' DESIGN'da acik birakilan ayrintilardir).")
out()
out("## Protokol ve dongu anahtari")
out(f"- Durum kaynaklari: **ToF oracle** (`y_true`) + S1 OOF `y_pred`: {', '.join(SRC[s] for s in srcs_present() if s != 'tof')}"
    + ("" if len(srcs_present()) == 5 else f" (bu kosuda OOF'u olmayan/yarim olanlar atlandi: {', '.join(SRC[s] for s in ('cnngru','cnnonly','b1','b2') if s not in SOURCES)})") + ".")
out("- OOF'ta pencere = son kare; her oturumun ilk 4 karesinde pencere yok -> **none sayilir** (ilk kismi dongu zaten atilir).")
out(f"- **Debounce** (kare sayisiyla): once ayni kutu arasindaki <= {MAX_GAP} karelik none boslugu birlestirilir, sonra < {MIN_BOX} karelik kutu run'i none yapilir; sabit noktaya kadar tekrarlanir. Ziyaret = kutu run'i; kalis = run suresi (kare zamanlarindan), gecis = ardisik ziyaretler arasi bosluk (>= 0).")
out("- **Segmentasyon:** `pred` = debounce edilmis Box1 girisi -> bir sonraki Box1 girisi (ToF kullanilmaz); ilk (Box1 oncesi) ve son (kapanmamis) kismi dongu atilir. `tof` = `anom_loops` donguleri (`use` in normal_crossfit/anomaly_test_all_folds; kismi/tamamlanmamis donguler atilir).")
out("- **Kat esleme (pred):** tahmini dongunun kati, Box1-giris karesini iceren ToF dongusunun katidir; o dongu kullanilmayan bir dongu ise (parca/tamamlanmamis) tahmini dongu atilir ve 'unmapped' sayilir. Etiket/natural_dev de o ToF dongusunden gelir.")
out("- Oturum etiketleri: s02 normal; s03 seq135, s04 seq145 (sira anomalisi); s05 fast, s06 slow (tempo anomalisi); s01 kullanilmaz.")
out("- **Dongu anahtari (05 ile hizalama):** `(state_source, segmentation, session, cycle_idx, start_frame_id)`; `segmentation=tof` icin cycle_idx = ToF loop_id, start_frame_id = loop'un ilk karesi; `segmentation=pred` icin cycle_idx = oturumdaki tahmini Box1 girisinin 0-tabanli sirasi (debounce sonrasi, atilanlar dahil), start_frame_id = giris karesi. `04_cycle_scores.csv.gz` satiri = (anahtar, f, role) ve `start_frame_id`/`end_frame_id` (dahil) ile `dur_s` icerir; 05 ayni dongu sinirlariyla OLD skorlarini `05_old_cycle_scores.csv.gz` icinde anahtar + `f` + `old_tft` + `old_z` olarak yazmalidir.")
out()
out("### Kesinlestirilen parametreler (DESIGN'da acik birakilanlar)")
out(f"- **D0:** log(dongu suresi) icin |z|; mu, sigma s02 egitim donguleri; sigma tabani {SIG_MIN}.")
out("- **D1:** durumlar {1..5, END}, ek olarak bir **START** baglami (ilk ziyaretin olasiligi da puanlanir; pred segmentasyonda ilk ziyaret hep Box1'dir). Laplace alpha = 1: p = (sayi + 1)/(satir toplami + 6); egitimde hic gorulmeyen baglam -> duz 1/6. Skor = max −log p.")
out(f"- **D2:** adim anahtarlari `dwell_i` ve `gap_i_j` (ardisik ziyaretler); log(d + {EPS} s) ile (sifir uzunluklu gecisler icin); referans = egitimde >= {MIN_REF} gozlem, sigma tabani {SIG_MIN}; referansi olmayan adim atlanir; hic referansli adimi olmayan dongunun D2 skoru **0** (`d2_nref` sutunu sayiyi tutar). Donguye baslangic/bitis boslugu (START→i, son→END) adim sayilmaz.")
out("- **D3:** 2 · min(p1, p2) (Bonferroni), 1 ile sinirli; AUROC icin skor = 1 − p3.")
out("- **Conformal:** p = (1 + #{cal >= s})/(n_cal + 1); alarm p <= alpha; cal = `anom_roles(f)['cal']` (2 kat, s02 normal). Esitlikte (cal >= s) muhafazakar tarafta kalir.")
out()

# counts
out("## Dongu sayilari (2×2 hucre × oturum)")
out("Hucre = durum kaynagi × segmentasyon. Deger: **kullanilan dongu** (ToF referansi = `anom_loops` kullanilan dongu sayisi). pred segmentasyonda ayrica atilan: unmapped (Box1 girisi kullanilmayan bir ToF dongusunde) / acik son dongu.")
sess_names = {s: s.split("_")[1] for s in SESS}
crow = []
for src in srcs_present():
    for seg in ("tof", "pred"):
        d = CNT[(CNT.state_source == src) & (CNT.segmentation == seg)].set_index("session")
        r = {"kaynak": SRC[src], "segmentasyon": seg}
        for s in SESS:
            r["s" + sess_names[s]] = f"{int(d.loc[s,'n_used'])} ({int(d.loc[s,'n_tof_ref'])})"
        r["toplam"] = int(d["n_used"].sum())
        r["normal(s02) dogal-sapma"] = int(d.loc[SESS[0], "n_natural_dev"])
        crow.append(r)
out(md_table(pd.DataFrame(crow)))
out()
out("### pred segmentasyon: ToF dongusu basina tahmini dongu sayisi (0 = kacirilan, >= 2 = bolunen) ve atilanlar")
crow = []
for src in srcs_present():
    d = CNT[(CNT.state_source == src) & (CNT.segmentation == "pred")]
    crow.append({"kaynak": SRC[src], "Box1 giris (toplam)": int(d["n_pred_box1_entries"].sum()),
                 "kullanilan": int(d["n_used"].sum()), "ToF ref": int(d["n_tof_ref"].sum()),
                 "loop'ta 0": int(d["n_loops_with_0"].sum()), "loop'ta 1": int(d["n_loops_with_1"].sum()),
                 "loop'ta >=2": int(d["n_loops_with_2plus"].sum()),
                 "atilan unmapped": int(d["n_dropped_unmapped_loop"].sum()), "atilan acik-son": int(d["n_dropped_open_last"].sum())})
out(md_table(pd.DataFrame(crow)))
out()
ref_tot = int(CNT[(CNT.state_source == "tof") & (CNT.segmentation == "tof")]["n_used"].sum())
out(f"Not: ToF×ToF hucresi dogrudan `anom_loops` kullanilan dongu sayisidir (toplam {ref_tot}; s02 normal = {int(CNT[(CNT.state_source=='tof')&(CNT.segmentation=='tof')&(CNT.session==SESS[0])]['n_used'].iloc[0])}). Oturum basina ~50 dongu beklenir.")
out()

# oracle diagnosis + visit-sequence distribution
yt = ft["y_true"].to_numpy()
usedL = AL[AL["use"].isin(USED)]
dg = {}
for r in usedL.itertuples():
    y = yt[int(r.first_row):int(r.last_row) + 1]
    d = dg.setdefault(r.session, dict(n=0, nob1=0, nob5=0, strad=0))
    d["n"] += 1
    d["nob1"] += int(not (y == 1).any())
    d["nob5"] += int(not (y == 5).any())
    d["strad"] += int(len(y) > 0 and r.first_row > 0 and y[0] == 1 and yt[int(r.first_row) - 1] == 1
                      and ft["session"].iloc[int(r.first_row) - 1] == r.session)
out("### Oracle (ToF) tanisi: `loop_id` sinirlari ile 10 Hz yumusatilmis etiket (oturum bazinda)")
out(md_table(pd.DataFrame([{"oturum": k.split("_")[1], "kullanilan loop": v["n"], "hic Box1 karesi yok": v["nob1"],
                            "hic Box5 karesi yok": v["nob5"], "ilk kare onceki loop'tan suren Box1 run'inda": v["strad"]}
                           for k, v in dg.items()])))
out("±3 kare mod yumusatma ve 10 Hz ornekleme ~<4 karelik dokunuslari siler; `anom_loops.box_order` ise 30 Hz ham ToF'tan gelir, bu yuzden kare-duzeyi dizi bazi donguler icin 1-2-5'ten (ya da oturumun beklenen dizisinden) eksik cikar; `loop_id` ayrica Box1 girisinde arttigindan Box1 run'i dongu sinirini asabilir. Sonuc (tablo): s05 (fast) oracle dizisi 49 dongunun cogunda 1-2 (Box5 kare duzeyinde gorunmuyor; olasi neden: cok kisa Box5 dokunusu yumusatmada siliniyor), s03/s04'te de az sayida 1-3 / 1-4 / (s04: 4) kisaltma var. Bu yuzden **D1'in fast AUROC'u (~0.93) bir hiz sinyali degil, kare-duzeyi kisaltma ('2'den sonra END') artefaktidir** ve D1'in tempo sutunlari 'sira' kanalinin yan etkisi olarak okunmali; s03/s04 icin ise sira anomalisi sinyali (1→3, 1→4, 3→5, 4→5 gecisleri egitimde hic yok) gecerlidir. ToF×pred hucresinde 'loop'ta 0' sayisi ayrica her oturumun son (kapanmamis) dongusunu icerir.")
out()
CY = pd.concat(cycles_all, ignore_index=True)
CY["seqs"] = CY["seq"].map(lambda t: "-".join(map(str, t)))
out("### Ziyaret dizisi dagilimi: oturum × hucre (tum durum kaynaklari; her oturum icin beklenen dizi: s02 1-2-5, s03 1-3-5, s04 1-4-5, s05/s06 1-2-5)")
EXPV = {"session_02_normal_seq125": "1-2-5", "session_03_normal_seq135": "1-3-5", "session_04_normal_seq145": "1-4-5",
        "session_05_anomaly_fast": "1-2-5", "session_06_anomaly_slow": "1-2-5"}
rows = []
for src in srcs_present():
    for seg in ("tof", "pred"):
        for s_ in SESS:
            g = CY[(CY.state_source == src) & (CY.segmentation == seg) & (CY.session == s_)]
            vc = g["seqs"].value_counts()
            rows.append({"kaynak": SRC[src], "seg": seg, "oturum": s_.split("_")[1], "n": len(g),
                         "beklenen dizi orani": f"{(g['seqs'] == EXPV.get(s_, '')).mean():.2f}",
                         "farkli dizi": len(vc), "en sik 3 dizi": "; ".join(f"{k or 'ø'}×{v}" for k, v in vc.head(3).items())})
out(md_table(pd.DataFrame(rows)))
out()

# n_cal / min p
out("## Conformal ayrintisi: kalibrasyon boyutu ve ulasilabilir en kucuk p")
nc = NCAL.groupby(["state_source", "segmentation"]).agg(n_cal_min=("n_cal", "min"), n_cal_mean=("n_cal", "mean"),
                                                         n_cal_max=("n_cal", "max"), n_train_mean=("n_train", "mean")).reset_index()
nc["en kucuk p (1/(n+1)), n=min"] = 1.0 / (nc["n_cal_min"] + 1)
nc["D3 en kucuk p (2/(n+1)), n=min"] = np.minimum(1.0, 2.0 / (nc["n_cal_min"] + 1))
nc["state_source"] = nc["state_source"].map(SRC)
out(md_table(nc, "{:.3f}"))
out()
mn = float(NCAL["n_cal"].min())
out(f"Yapisal uyari (on-kayitli kural degistirilmedi): D3 = 2·min(p1, p2) oldugundan alpha'da alarm icin min(p1, p2) <= alpha/2 gerekir. n_cal ≈ 20'de en kucuk p ≈ 0.048; bu yuzden **D3 alpha = 0.10'da (esik 0.05) yalniz kalibrasyondaki tum skorlari asan donguler icin alarm verebilir ve alpha = 0.05'te (esik 0.025) hic alarm veremez** (gerekli n_cal >= 39; burada min n_cal = {int(mn)}). n_cal < 19 olan katlarda (ornegin B1×pred) D3 alpha = 0.10'da da hic alarm veremez (en kucuk D3 p = 2/(n_cal+1) > 0.10). D0 skoru tof segmentasyonunda durum kaynagindan bagimsizdir (dongu suresi ToF dongu sinirlarindan gelir), bu yuzden o hucrelerde D0 satirlari ozdestir. D1/D2/D0 tek basina alpha = 0.05'te alarm verebilir (p = 1/(n_cal+1) <= 0.05 icin n_cal >= 19). p-degerleri ayriktir ve cok sayida esitlik icerir (ozellikle D1: normal dongulerin cogu ayni sirayi paylasir) -> D3'un AUROC'u kaba cozunurludur; AUPRC/AUROC'ta esitlikler 0.5 sayilir.")
out()

# AUROC / AUPRC tables
ORDER_AT = list(AGROUPS)


def metric_table(seg, col_m, col_s):
    rows = []
    for src in srcs_present():
        for vn in V:
            if TAB[(TAB.state_source == src) & (TAB.segmentation == seg) & (TAB.variant == vn)].empty:
                continue
            r = {"kaynak": SRC[src], "skor": VNAME[vn]}
            for an in ORDER_AT:
                g = TAB[(TAB.state_source == src) & (TAB.segmentation == seg) & (TAB.variant == vn) & (TAB.atype == an)]
                r[an] = f"{g[col_m].iloc[0]:.3f} ± {g[col_s].iloc[0]:.3f}" if len(g) and not np.isnan(g[col_m].iloc[0]) else "nan"
            rows.append(r)
    return pd.DataFrame(rows)


out("## AUROC (normal test donguleri vs anomali turu; kat bazli ortalama ± sd, 5 kat)")
out(f"Her katta ~{PF[(PF.atype=='tum anomali')&(PF.variant=='D0')]['n_neg'].mean():.1f} normal test dongusu (anomali: tip basina ~{PF[(PF.atype=='seq135')&(PF.variant=='D0')]['n_pos'].mean():.0f}); AUROC SE ≈ 0.03–0.05 oldugundan 0.02 farklari cozulemez.")
for seg in ("tof", "pred"):
    out()
    out(f"### Segmentasyon = {SEGN[seg]} (`{seg}`)")
    out(md_table(metric_table(seg, "auroc_mean", "auroc_sd")))
out()
out("## AUPRC (pozitif = anomali; kat bazli ortalama ± sd)")
pv = PF[(PF.variant == "D0")].groupby("atype")["prevalence"].mean()
out("Taban cizgisi (prevalans, rastgele skor AUPRC'si) tur basina: " + ", ".join(f"{k}={pv[k]:.2f}" for k in ORDER_AT if k in pv.index) + ". Normal azinlikta oldugundan AUPRC yuksek gorunur; AUROC ile birlikte okunmali.")
for seg in ("tof", "pred"):
    out()
    out(f"### Segmentasyon = {SEGN[seg]} (`{seg}`)")
    out(md_table(metric_table(seg, "auprc_mean", "auprc_sd")))
out()

# alarm tables
for a in ALPHAS:
    k = f"a{int(a * 100):02d}"
    out(f"## Calisma noktasi alpha = {a:.2f}{' (BIRINCIL)' if a == 0.10 else ' (ikincil)'}: gozlenen FAR ve tespit orani")
    out(f"FAR = havuzlanmis normal test donguleri (her normal dongu bir kez test); SE = sqrt(alpha(1−alpha)/n_test); 'gecerli' = |FAR − alpha| <= 2·SE (on-kayitli olcut; FAR < alpha − 2·SE ise 'muhafazakar': kural gecerli ama gucsuz, esitlikler p'yi buyutur; FAR > alpha + 2·SE ise 'asiri': gecersiz). 'FAR (dogal sapma haric)' s02'deki ToF sirasi 1-2-5 olmayan donguleri disarida birakir. Tespit orani = alarm verilen (dongu, f) ciftlerinin orani (anomali donguleri her f'de test edilir; = f ortalamasi).")
    for seg in ("tof", "pred"):
        out()
        out(f"### Segmentasyon = `{seg}`")
        rows = []
        for src in srcs_present():
            for vn in V:
                g = TAB[(TAB.state_source == src) & (TAB.segmentation == seg) & (TAB.variant == vn) & (TAB.atype == "tum anomali")]
                if g.empty:
                    continue
                g = g.iloc[0]
                r = {"kaynak": SRC[src], "skor": VNAME[vn], "n_test": int(g["far_n_pairs"]), "FAR": f"{g['far_' + k]:.3f}",
                     "SE": f"{g['far_se_' + k]:.3f}", "gecerli": ("evet" if g["far_within2se_" + k] else ("HAYIR (muhafazakar)" if g["far_" + k] < a else "HAYIR (asiri)")),
                     "FAR (dogal sapma haric)": f"{g['far_excl_nat_' + k]:.3f}"}
                for an in ["seq135", "seq145", "fast", "slow", "tum anomali"]:
                    gg = TAB[(TAB.state_source == src) & (TAB.segmentation == seg) & (TAB.variant == vn) & (TAB.atype == an)]
                    r["tespit " + an] = f"{gg['det_' + k].iloc[0]:.3f}" if len(gg) else "nan"
                rows.append(r)
        out(md_table(pd.DataFrame(rows)))
    out()

# OLD
if HAS_OLD:
    out("## OLD boru hatti: sabit esik (TFT > 0.5, Z > 1.0) ve ayni conformal prosedur")
    rows = []
    for src in srcs_present():
        for seg in ("tof", "pred"):
            for vn in ("OLD_tft", "OLD_z", "OLD3"):
                g = TAB[(TAB.state_source == src) & (TAB.segmentation == seg) & (TAB.variant == vn) & (TAB.atype == "tum anomali")]
                if g.empty:
                    continue
                g = g.iloc[0]
                rows.append({"kaynak": SRC[src], "seg": seg, "skor": VNAME[vn], "AUROC(tum)": f"{g['auroc_mean']:.3f} ± {g['auroc_sd']:.3f}",
                             "FAR sabit": f"{g['far_fixed']:.3f}" if "far_fixed" in g and not np.isnan(g["far_fixed"]) else "-",
                             "tespit sabit": f"{g['det_fixed']:.3f}" if "det_fixed" in g and not np.isnan(g["det_fixed"]) else "-",
                             "FAR a=0.10": f"{g['far_a10']:.3f}", "tespit a=0.10": f"{g['det_a10']:.3f}"})
    out(md_table(pd.DataFrame(rows)))
    out()
else:
    out("## OLD boru hatti")
    out("`05_old_cycle_scores.csv.gz` henuz yok -> OLD karsilastirmasi bu kosuda atlandi (05 yazildiginda 04 yeniden kosulur).")
    out()

# paired differences vs CNN-GRU
out("## Eslesik kat farki: durum kaynagi X − CNN-GRU (D3 AUROC; karar olcutu 2'nin 2. kosulu)")
if "cnngru" not in SOURCES:
    out("CNN-GRU S1 OOF'u henuz tam degil -> eslesik fark uretilemedi (kat bazli AUROC'lar `04_anomaly_perfold.csv`'de; CNN OOF'u gelince 04 yeniden kosulur).")
else:
    rows = []
    for seg in ("tof", "pred"):
        for src in srcs_present():
            if src == "cnngru":
                continue
            for an in ["seq135", "seq145", "fast", "slow"]:
                a, fa = pf_vals(src, seg, "D3", an)
                b, fb = pf_vals("cnngru", seg, "D3", an)
                if len(a) != len(b) or len(a) < 2:
                    continue
                dd = a - b
                dd = dd[~np.isnan(dd)]
                if len(dd) < 2:
                    continue
                m, s_ = float(dd.mean()), float(dd.std(ddof=1))
                h = float(stats.t.ppf(0.975, len(dd) - 1) * s_ / math.sqrt(len(dd)))
                ok = (m - h <= 0 <= m + h) or abs(m) <= 0.05
                rows.append({"seg": seg, "X": SRC[src], "anomali": an, "fark (X−CNN-GRU)": f"{m:+.3f}",
                             "%95 GA": f"[{m - h:+.3f}, {m + h:+.3f}]", "0'i iceriyor ya da |fark|<=0.05": "evet" if ok else "HAYIR"})
    out(md_table(pd.DataFrame(rows)))
out()

# gates
out("## Dogrulama kapilari (ToF oracle hucresi)")
for seg in ("tof", "pred"):
    if "tof" in SOURCES:
        out(f"- Oracle × `{seg}`: D1 AUROC seq135 = {cell('tof',seg,'D1','seq135','auroc_mean'):.3f}, seq145 = {cell('tof',seg,'D1','seq145','auroc_mean'):.3f} (beklenen ≈ 1); "
            f"D0 AUROC fast = {cell('tof',seg,'D0','fast','auroc_mean'):.3f}, slow = {cell('tof',seg,'D0','slow','auroc_mean'):.3f}; "
            f"D2 AUROC fast = {cell('tof',seg,'D2','fast','auroc_mean'):.3f}, slow = {cell('tof',seg,'D2','slow','auroc_mean'):.3f}; "
            f"FAR@0.10: D0 {cell('tof',seg,'D0','tum anomali','far_a10'):.3f} / D1 {cell('tof',seg,'D1','tum anomali','far_a10'):.3f} / D2 {cell('tof',seg,'D2','tum anomali','far_a10'):.3f} / D3 {cell('tof',seg,'D3','tum anomali','far_a10'):.3f}.")
out()
out("Yorum ve karar olcutleri (B1/B2 vs CNN-GRU hukmu) bu asamada verilmez; yalniz protokolun sayilari uretilir.")

(fc.OUT / "04_anomaly.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")
print("[tamam] yazildi: 04_anomaly.md, 04_anomaly_table.csv, 04_anomaly_perfold.csv, 04_cycle_scores.csv.gz, 04_cycle_counts.csv")

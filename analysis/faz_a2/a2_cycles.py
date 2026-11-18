"""Faz A2 - yeniden kullanilabilir DONGU / ANOMALI modulu (faz_a/04_cycle_anomaly.py mantiginin fonksiyon hali).

IMPORT YAN ETKISI YOKTUR: veri/etiket tablolari ilk kullanimda tembel (lru_cache) yuklenir.
Faz A'nin sozlesmeleri BIREBIR korunur: START baglami, Laplace alpha = 1 (6 durum), log(d + 0.05), sigma tabani 0.05,
min-ref 3, referansli adimi olmayan dongunun D2 = 0, esitlikte `cal >= s` (1e-12 toleransi), debounce (min_box 2, max_gap 1).

Kullanim ozeti
--------------
    import sys; sys.path.insert(0, str(ROOT/'faz_a2')); import a2_cycles as ac

    LAB = ac.source_labels()                         # {source: {session: per-frame int labels}}  (tof + S1 OOF'lar)
    C, cnt = ac.build_cycles('b2', 'pred', LAB['b2'])   # DataFrame (bir satir = bir dongu) + sayim listesi
    S, ncal = ac.score_group(C)                      # Faz A protokolu: her f icin roller, D0/D1/D2, p0..p3 (standart conformal)
    S = ac.score_all()                               # tum kaynak x segmentasyon (Faz A 04_cycle_scores ile ayni sutunlar)

    # keyfi normal egitim dongu listesinden parametre ogrenme / puanlama
    par = ac.fit_params(list_of_cycle_dicts)         # Params(d0, d1, d2)
    df  = ac.score_cycles(list_of_cycle_dicts, par)  # DataFrame: D0, D1, D2, d2_nref (+ kimlik sutunlari)

    # conformal: standart / duzgunlestirilmis / cross-conformal
    ac.conf_p(cal, s); ac.conf_p_smoothed(cal, s, u); ac.smoothed_draws(cal, s, n_draws, seed)
    cf = ac.cross_scores(C, f); ac.cross_p(cf, 'D1'); ac.cross_p_draws(cf, 'D1', 200, seed)
    ac.d3(p1, p2)                                    # min(1, 2 min(p1, p2))
    ac.auc_ap(neg, pos); ac.alarm(p, alpha); ac.evaluate(S) -> (PF, SM); ac.summarize(PF, SM) -> TAB
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "faz_a"))
sys.path.insert(0, str(_ROOT / "probe"))
import fa_common as fc  # noqa: E402

# ------------------------------------------------------------------ constants (Faz A 04 ile ayni)
DT = 0.1                     # kare periyodu (s)
MIN_BOX = 2                  # debounce: kutu run'i >= 2 kare
MAX_GAP = 1                  # debounce: <= 1 karelik none boslugu ayni kutular arasinda birlestirilir
EPS = 0.05                   # log(d + EPS)
SIG_MIN = 0.05               # log-olcekte sigma tabani
MIN_REF = 3                  # D2: bir adimin referansi icin en az egitim gozlemi
TIE_TOL = 1e-12              # conformal esitlik toleransi (Faz A: cal >= s - 1e-12)
ALPHAS = (0.10, 0.05)
USED = {"normal_crossfit", "anomaly_test_all_folds"}
ATYPES = ["seq135", "seq145", "fast", "slow"]
AGROUPS = {"seq135": ["seq135"], "seq145": ["seq145"], "fast": ["fast"], "slow": ["slow"],
           "seq(135+145)": ["seq135", "seq145"], "tempo(fast+slow)": ["fast", "slow"],
           "tum anomali": ["seq135", "seq145", "fast", "slow"]}
SOURCES = ("tof", "cnngru", "cnnonly", "b1", "b2")
MODEL_SOURCES = ("cnngru", "cnnonly", "b1", "b2")
SEGS = ("pred", "tof")
VARIANTS = ("D0", "D1", "D2", "D3")
N_FOLDS = fc.N_FOLDS


# ------------------------------------------------------------------ tembel baglam
@lru_cache(maxsize=1)
def ctx() -> SimpleNamespace:
    """ft (frame_table), AL (anom_loops), SESS (s01 haric), OFF (oturum ilk satiri), SE (oturum basina kare tablosu), ALX."""
    ft = fc.frame_table()
    assert ft.groupby("session")["frame_id"].apply(lambda s: bool((np.diff(s.to_numpy()) > 0).all())).all()
    AL = fc.anom_loops()
    SESS = [s for s in sorted(ft["session"].unique()) if s != "session_01_training"]
    OFF = {s: int(ft.index[ft["session"] == s][0]) for s in SESS}
    assert all((ft["row"].to_numpy()[ft["session"] == s] == np.arange(len(ft))[ft["session"] == s]).all() for s in SESS)
    SE = {s: ft[ft["session"] == s].reset_index(drop=True) for s in SESS}
    ALX = {(r.session, int(r.loop_id)): r for r in AL.itertuples(index=False)}
    return SimpleNamespace(ft=ft, AL=AL, SESS=SESS, OFF=OFF, SE=SE, ALX=ALX)


# ------------------------------------------------------------------ durum kaynaklari
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


def labels_from_oof(oof: pd.DataFrame, strict: bool = True) -> dict | None:
    """Bir OOF DataFrame'inden (fa_common.OOF_COLS; pencere = son kare) {session: kare basina y_pred}.
    Penceresi olmayan kareler (oturum basina ilk 4) -> none (0). strict=True: 16,248 satir ve 5 kat degilse None."""
    c = ctx()
    if len(oof) == 0:
        return None
    if strict and (len(oof) != 16248 or oof["fold"].nunique() != 5):
        return None
    mm = c.ft[["session", "frame_id"]].merge(oof[["session", "frame_id", "y_pred"]], on=["session", "frame_id"],
                                             how="left", validate="1:1")
    y = mm["y_pred"].fillna(0).astype(int).to_numpy()
    ses = c.ft["session"].to_numpy()
    return {s: y[ses == s] for s in c.SESS}


def source_labels(sources=SOURCES, oof_loader=None, verbose=True) -> dict:
    """{source: {session: int array}}.  'tof' = frame_table y_true (oracle); modeller = S1 OOF y_pred.
    oof_loader(model) -> OOF DataFrame (varsayilan fc.load_oof(model, 'S1')). Eksik/yarim OOF atlanir (Faz A gibi)."""
    c = ctx()
    res = {}
    if "tof" in sources:
        res["tof"] = {s: c.SE[s]["y_true"].to_numpy().astype(int) for s in c.SESS}
    for m in sources:
        if m == "tof":
            continue
        oof = oof_loader(m) if oof_loader is not None else fc.load_oof(m, "S1")
        lab = labels_from_oof(oof)
        if lab is None:
            if verbose:
                print(f"[uyari] {m}: S1 OOF yok/yarim -> atlandi")
            continue
        res[m] = lab
    return res


# ------------------------------------------------------------------ dongu insasi
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


def build_cycles(source: str, seg: str, labs: dict, sessions=None):
    """Bir (kaynak, segmentasyon) icin dongu tablosu. labs = {session: per-frame label}. Donus: (DataFrame, sayim listesi).
    seg='tof'  : anom_loops (use in USED) ToF dongu sinirlari; seg='pred': debounce edilmis Box1 girisinden bir sonrakine.
    Satirlar: state_source, segmentation, session, cycle_idx, start_frame_id, end_frame_id, start_idx, end_idx, dur_s,
    anchor_loop, atype, fold (ToF dongusunun s02 kati; anomali/oturum disi -1), natural_dev, seq (tuple), dw, gp."""
    c = ctx()
    AL, ALX, OFF, SE = c.AL, c.ALX, c.OFF, c.SE
    rows, cnt = [], []
    for s in (c.SESS if sessions is None else sessions):
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
        cs = [r for r in rows if r["session"] == s]
        per_loop = Counter(r["anchor_loop"] for r in cs)
        used_loops = AL[(AL["session"] == s) & AL["use"].isin(USED)]["loop_id"].astype(int).tolist()
        k = [per_loop.get(l, 0) for l in used_loops]
        cnt.append(dict(state_source=source, segmentation=seg, session=s, atype=(cs[0]["atype"] if cs else ""),
                        n_tof_ref=n_ref, n_pred_box1_entries=n_entries if seg == "pred" else np.nan,
                        n_used=len(cs), n_dropped_unmapped_loop=dropped_unm, n_dropped_open_last=dropped_open,
                        n_natural_dev=int(sum(r["natural_dev"] for r in cs)),
                        n_loops_with_0=int(sum(1 for x in k if x == 0)), n_loops_with_1=int(sum(1 for x in k if x == 1)),
                        n_loops_with_2plus=int(sum(1 for x in k if x >= 2))))
    return pd.DataFrame(rows), cnt


# ------------------------------------------------------------------ D0 / D1 / D2
def d0_fit(tr):
    x = np.log(np.array([c["dur_s"] for c in tr], float))
    sd = float(x.std(ddof=1)) if len(x) > 1 else SIG_MIN
    return float(x.mean()), float(max(sd, SIG_MIN))


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


class Params(NamedTuple):
    d0: tuple
    d1: dict
    d2: dict


def _records(cycles):
    if isinstance(cycles, pd.DataFrame):
        return cycles.to_dict("records")
    return list(cycles)


def fit_params(train_cycles) -> Params:
    """D0/D1/D2 parametrelerini KEYFI bir normal egitim dongu listesinden ogren (liste of dict ya da DataFrame)."""
    tr = _records(train_cycles)
    return Params(d0_fit(tr), d1_fit(tr), d2_fit(tr))


_ID_COLS = ("state_source", "segmentation", "session", "cycle_idx", "start_frame_id", "end_frame_id")


def score_cycle(c: dict, par: Params) -> dict:
    s2, nref = d2_score(c, par.d2)
    return dict(D0=d0_score(c, par.d0), D1=d1_score(c, par.d1), D2=s2, d2_nref=nref)


def score_cycles(cycles, par: Params) -> pd.DataFrame:
    """Her dongu icin D0, D1, D2, d2_nref (+ kimlik/etiket sutunlari). Satir sirasi girdiyle ayni."""
    cl = _records(cycles)
    rows = []
    for c in cl:
        r = {k: c[k] for k in _ID_COLS if k in c}
        r.update(atype=c.get("atype"), natural_dev=c.get("natural_dev"), tof_fold=c.get("fold"),
                 anchor_loop=c.get("anchor_loop"), dur_s=c["dur_s"], n_visits=len(c["seq"]),
                 visit_seq="-".join(map(str, c["seq"])))
        r.update(score_cycle(c, par))
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ conformal
def conf_p(cal_scores, s):
    """Faz A standardi: p = (1 + #{cal >= s}) / (n_cal + 1)  (esitlikte muhafazakar; toleransli >=)."""
    cs = np.asarray(cal_scores, float)
    s = np.asarray(s, float)
    return (1.0 + (cs[None, :] >= s[:, None] - TIE_TOL).sum(axis=1)) / (len(cs) + 1.0)


def _gt_eq(cal_scores, s):
    cs = np.asarray(cal_scores, float)[None, :]
    s = np.asarray(s, float)[:, None]
    gt = (cs > s + TIE_TOL).sum(axis=1)
    eq = (np.abs(cs - s) <= TIE_TOL).sum(axis=1)
    return gt, eq


def conf_p_smoothed(cal_scores, s, u):
    """Duzgunlestirilmis (randomize) p = (#{cal > s} + U (#{cal = s} + 1)) / (n_cal + 1); u: skaler ya da len(s) vektor, U ~ Unif(0,1).
    u = 1 -> standart conf_p ile ayni."""
    gt, eq = _gt_eq(cal_scores, s)
    return (gt + np.asarray(u, float) * (eq + 1.0)) / (len(np.asarray(cal_scores)) + 1.0)


def smoothed_draws(cal_scores, s, n_draws: int = 200, seed: int = 0):
    """(n_draws, n_test) duzgunlestirilmis p; her cekilis her test dongusu icin bagimsiz U; RandomState(seed)."""
    gt, eq = _gt_eq(cal_scores, s)
    U = np.random.RandomState(seed).uniform(size=(n_draws, len(gt)))
    return (gt[None, :] + U * (eq[None, :] + 1.0)) / (len(np.asarray(cal_scores)) + 1.0)


def d3(p1, p2):
    """Birincil kural: D3 = min(1, 2 min(p1, p2)) (Bonferroni)."""
    return np.minimum(1.0, 2.0 * np.minimum(np.asarray(p1, float), np.asarray(p2, float)))


def alarm(p, alpha):
    return np.asarray(p, float) <= alpha + 1e-9


# ------------------------------------------------------------------ Faz A protokolu: roller + puanlama
def _split_roles(cl, ro):
    tr = [c for c in cl if c["atype"] == "normal" and c["fold"] in ro["train"]]
    ca = [c for c in cl if c["atype"] == "normal" and c["fold"] in ro["cal"]]
    te = [c for c in cl if (c["atype"] == "normal" and c["fold"] in ro["test"]) or c["atype"] != "normal"]
    return tr, ca, te


def score_group(C: pd.DataFrame, roles_fn=None, min_cycles: int = 3, verbose=True):
    """Faz A `score_group`: her f icin roller (varsayilan fc.anom_roles), egit -> D0/D1/D2; p0,p1,p2 (standart conformal; yalniz
    test satirlari), p3 = D3. Satir = (dongu, f). Donus: (DataFrame, [dict(f, n_cal, n_train)])."""
    roles_fn = roles_fn or fc.anom_roles
    res, ncal_log = [], []
    cl = C.to_dict("records")
    for f in range(N_FOLDS):
        ro = roles_fn(f)
        tr, ca, te = _split_roles(cl, ro)
        if len(tr) < min_cycles or len(ca) < min_cycles:
            if verbose:
                print(f"[uyari] f={f}: egitim={len(tr)} kal={len(ca)} yetersiz")
            continue
        par = fit_params(tr)
        parts = []
        for role, lst in (("train", tr), ("cal", ca), ("test", te)):
            d = score_cycles(lst, par)
            d.insert(6, "f", f)
            d.insert(7, "role", role)
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        cal = d[d["role"] == "cal"]
        ncal_log.append(dict(f=f, n_cal=len(cal), n_train=len(tr)))
        for k, nm in (("0", "D0"), ("1", "D1"), ("2", "D2")):
            m = d["role"] == "test"
            d.loc[m, "p" + k] = conf_p(cal[nm].to_numpy(), d.loc[m, nm].to_numpy())
        d["p3"] = np.minimum(1.0, 2.0 * d[["p1", "p2"]].min(axis=1))
        res.append(d)
    return pd.concat(res, ignore_index=True), ncal_log


@lru_cache(maxsize=1)
def _default_labels():
    return source_labels()


def score_all(sources=None, segs=SEGS, labels=None, roles_fn=None, verbose=False):
    """Tum kaynak x segmentasyon icin skor tablosu (Faz A 04_cycle_scores.csv.gz'in OLD olmayan sutunlari ile ayni).
    Donus: (S, CNT, NCAL)."""
    LAB = labels if labels is not None else _default_labels()
    scores, cnts, ncals = [], [], []
    for src in (sources or list(LAB)):
        for seg in segs:
            C, cnt = build_cycles(src, seg, LAB[src])
            cnts += cnt
            Sg, ncl = score_group(C, roles_fn=roles_fn, verbose=verbose)
            for x in ncl:
                x.update(state_source=src, segmentation=seg)
            ncals += ncl
            scores.append(Sg)
    return pd.concat(scores, ignore_index=True), pd.DataFrame(cnts), pd.DataFrame(ncals)


# ------------------------------------------------------------------ cross-conformal (DESIGN_A2 A2.1)
def cross_scores(C: pd.DataFrame, f: int, folds=None) -> dict:
    """Cross-conformal bilesenleri (test kati f). Kalan katlar R = tum katlar \\ {f}; her k in R icin:
    parametreler R \\ {k} katlarindaki normal donguler (~3 kat) ile ogrenilir; k kati (kat-disi kalibrasyon) ve test dongulerinin
    (f kati normal + tum anomaliler) skorlari model k ile hesaplanir.
    Donus: dict(
      test = DataFrame (test dongulerinin kimlik sutunlari; satir sirasi = S matrislerinin sirasi),
      S    = {'D0'|'D1'|'D2': (n_test, K) test skorlari, sutun j = model k_j},
      cal  = {'D0'|'D1'|'D2': [k basina kalibrasyon skor dizisi]},
      folds_k = [k...], n_cal = toplam kalibrasyon dongusu, n_train = [model basina egitim dongusu sayisi])."""
    folds = list(range(N_FOLDS)) if folds is None else list(folds)
    cl = C.to_dict("records")
    R = [k for k in folds if k != f]
    te = [c for c in cl if (c["atype"] == "normal" and c["fold"] == f) or c["atype"] != "normal"]
    S = {v: np.zeros((len(te), len(R))) for v in ("D0", "D1", "D2")}
    cal = {v: [] for v in ("D0", "D1", "D2")}
    ntr = []
    for j, k in enumerate(R):
        tr = [c for c in cl if c["atype"] == "normal" and c["fold"] in R and c["fold"] != k]
        ck = [c for c in cl if c["atype"] == "normal" and c["fold"] == k]
        par = fit_params(tr)
        ntr.append(len(tr))
        sk = score_cycles(ck, par)
        st = score_cycles(te, par)
        for v in ("D0", "D1", "D2"):
            cal[v].append(sk[v].to_numpy(float))
            S[v][:, j] = st[v].to_numpy(float)
    testdf = pd.DataFrame([{k: c[k] for k in _ID_COLS if k in c} | dict(atype=c["atype"], natural_dev=c["natural_dev"],
                                                                        tof_fold=c["fold"]) for c in te])
    return dict(f=f, test=testdf, S=S, cal=cal, folds_k=R, n_cal=int(sum(len(a) for a in cal["D0"])), n_train=ntr)


def _cross_counts(cf, v):
    n = cf["S"][v].shape[0]
    gt = np.zeros(n)
    eq = np.zeros(n)
    for j in range(len(cf["folds_k"])):
        g, e = _gt_eq(cf["cal"][v][j], cf["S"][v][:, j])
        gt += g
        eq += e
    return gt, eq


def cross_p(cf: dict, v: str, u=None):
    """Cross-conformal p, variant v in D0/D1/D2. u=None -> standart: p = (1 + sum_k #{cal_k >= s^(k)}) / (n_cal + 1).
    u (skaler ya da n_test vektor) -> duzgunlestirilmis: (sum #{>} + u (sum #{=} + 1)) / (n_cal + 1)."""
    gt, eq = _cross_counts(cf, v)
    if u is None:
        return (1.0 + gt + eq) / (cf["n_cal"] + 1.0)
    return (gt + np.asarray(u, float) * (eq + 1.0)) / (cf["n_cal"] + 1.0)


def cross_p_draws(cf: dict, v: str, n_draws: int = 200, seed: int = 0):
    """(n_draws, n_test) duzgunlestirilmis cross-conformal p cekilisleri; RandomState(seed)."""
    gt, eq = _cross_counts(cf, v)
    U = np.random.RandomState(seed).uniform(size=(n_draws, len(gt)))
    return (gt[None, :] + U * (eq[None, :] + 1.0)) / (cf["n_cal"] + 1.0)


# ------------------------------------------------------------------ metrikler
def auc_ap(neg, pos):
    from sklearn.metrics import roc_auc_score, average_precision_score
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    if len(neg) < 2 or len(pos) < 1:
        return float("nan"), float("nan")
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _variants():
    return {"D0": ("D0", "p0"), "D1": ("D1", "p1"), "D2": ("D2", "p2"), "D3": (None, "p3")}


def vscore(d, sc, pc):
    """AUROC skoru: ham skor (D0/D1/D2) ya da 1 - p (D3)."""
    return d[sc].to_numpy(float) if sc else (1.0 - d[pc].to_numpy(float))


def evaluate(S: pd.DataFrame):
    """Faz A `evaluate` (OLD haric): (PF = kat bazli AUROC/AUPRC, SM = havuzlanmis alarm oranlari)."""
    V = _variants()
    perfold, summ = [], []
    T = S[S["role"] == "test"]
    for (sr, sg), g in T.groupby(["state_source", "segmentation"], sort=False):
        for vn, (sc, pc) in V.items():
            if pc not in g or g[pc].notna().sum() == 0:
                continue
            for f, gf in g.groupby("f"):
                neg_d = gf[gf["atype"] == "normal"]
                for an, members in AGROUPS.items():
                    pos_d = gf[gf["atype"].isin(members)]
                    a, ap = auc_ap(vscore(neg_d, sc, pc), vscore(pos_d, sc, pc))
                    perfold.append(dict(state_source=sr, segmentation=sg, variant=vn, atype=an, f=f,
                                        n_neg=len(neg_d), n_pos=len(pos_d), auroc=a, auprc=ap,
                                        prevalence=len(pos_d) / max(1, len(pos_d) + len(neg_d))))
            neg = g[(g["atype"] == "normal") & g[pc].notna()]
            alarms = {f"a{int(a * 100):02d}": (lambda d, a=a: (d[pc].to_numpy(float) <= a + 1e-9)) for a in ALPHAS}
            for an in ["normal"] + list(AGROUPS):
                d = neg if an == "normal" else g[g["atype"].isin(AGROUPS[an]) & g[pc].notna()]
                row = dict(state_source=sr, segmentation=sg, variant=vn, atype=an, n_pairs=len(d))
                for ak, fn in alarms.items():
                    row[f"rate_{ak}"] = float(fn(d).mean()) if len(d) else float("nan")
                if an == "normal":
                    nonat = d[~d["natural_dev"]]
                    row["n_natdev"] = int(d["natural_dev"].sum())
                    for ak, fn in alarms.items():
                        row[f"rate_excl_nat_{ak}"] = float(fn(nonat).mean()) if len(nonat) else float("nan")
                summ.append(row)
    return pd.DataFrame(perfold), pd.DataFrame(summ)


def _mean(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(a.mean()) if len(a) else float("nan")


def _sd(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def summarize(PF: pd.DataFrame, SM: pd.DataFrame) -> pd.DataFrame:
    """Faz A 04_anomaly_table.csv ile ayni sutunlar (OLD / sabit-esik sutunlari haric)."""
    rows = []
    for (sr, sg, vn, an), g in PF.groupby(["state_source", "segmentation", "variant", "atype"], sort=False):
        rows.append(dict(state_source=sr, segmentation=sg, variant=vn, atype=an, n_folds=int(g["auroc"].notna().sum()),
                         auroc_mean=_mean(g["auroc"]), auroc_sd=_sd(g["auroc"]), auprc_mean=_mean(g["auprc"]),
                         auprc_sd=_sd(g["auprc"]), prevalence=_mean(g["prevalence"]),
                         n_neg_per_fold=_mean(g["n_neg"]), n_pos_per_fold=_mean(g["n_pos"])))
    TAB = pd.DataFrame(rows)
    farcols = [c for c in SM.columns if c.startswith("rate_") or c in ("n_pairs", "n_natdev")]
    nm = SM[SM["atype"] == "normal"].rename(columns={c: ("far_" + c[5:] if c.startswith("rate_") else "far_" + c) for c in farcols})
    nm = nm.drop(columns=["atype"])
    an_ = SM[SM["atype"] != "normal"].rename(columns={"n_pairs": "det_n_pairs", **{c: "det_" + c[5:] for c in farcols if c.startswith("rate_")}})
    an_ = an_[[c for c in an_.columns if "excl_nat" not in c and c != "n_natdev"]]
    TAB = TAB.merge(an_, on=["state_source", "segmentation", "variant", "atype"], how="left")
    TAB = TAB.merge(nm, on=["state_source", "segmentation", "variant"], how="left")
    for a in ALPHAS:
        k = f"a{int(a * 100):02d}"
        TAB[f"far_se_{k}"] = np.sqrt(a * (1 - a) / TAB["far_n_pairs"])
        TAB[f"far_within2se_{k}"] = (TAB[f"far_{k}"] - a).abs() <= 2 * TAB[f"far_se_{k}"]
    return TAB

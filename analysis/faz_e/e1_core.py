"""Phase E - E1 core: evidence-weighted, multi-match edit distance of a cycle's visits to the learned route.

Implements faz_e/DESIGN_E.md section 2 (frozen; sha256 of the part above "## Sapmalar" is checked by the scripts):
  2.2  visits = maximal runs of the DEBOUNCED label b > 0; mass E_v = sum_{t in run} p_b(t) (frames of posterior evidence);
       collapse = consecutive visits of the same box separated only by none frames merge, masses add.
  2.3  route R = modal collapsed visit sequence of the fold's training normal cycles;
       mu_b = mean over training normal cycles of the whole-cycle posterior mass of box b (transit frames included).
  2.4  S_E = multi-match DP (A = step open, B = step deleted / i = 0) with traceback; the unit cases of 2.4 are asserted
       when this module is imported (tiny synthetic visit lists, see _unit_tests()).
  2.6  explain(): deleted route steps with mu, charged visits with box, first/last frame and mass.
  3    score_roles(): Phase A fold roles (test f, cal f+1,f+2, train f+3,f+4), p_E = conformal p of a2_cycles.conf_p.

The plain (non-multi-match) DP is available only as align(..., multimatch=False) for the synthetic tabulation of
section 4 (01_e1_synthetic.py). score_roles() - the only path used by the one-shot script - always uses the frozen
multi-match score.

Import side effects: none besides importing faz_a2/a2_cycles.py (itself side-effect free; it loads no data) and
running the unit tests.
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
import a2_cycles as ac  # noqa: E402  (Phase A cycles/debounce/D1/D2/conformal; no data is loaded at import)

DESIGN = ROOT / "faz_e" / "DESIGN_E.md"
# sha256 of the bytes of DESIGN_E.md before the "## Sapmalar" heading, as logged in paper/review/CHANGELOG_A4.md (item 12)
DESIGN_FROZEN_SHA256 = "d8c2cafd4a788f0b87e8e222e2ec8ae4e6e8dc74c81b687a885b7207df168d76"
N_CLASSES = 6
INF = float("inf")
SEP = "-"


# ------------------------------------------------------------------ provenance
def design_frozen_sha256(path: Path = DESIGN) -> str:
    b = Path(path).read_bytes()
    return hashlib.sha256(b[: b.index(b"## Sapmalar")]).hexdigest()


def assert_design_frozen() -> str:
    h = design_frozen_sha256()
    if h != DESIGN_FROZEN_SHA256:
        raise SystemExit(f"[E1] DESIGN_E.md frozen part changed: sha256 {h} != {DESIGN_FROZEN_SHA256}. Refusing to run.")
    return h


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ 2.2 visits, mass, collapse
class Visit(NamedTuple):
    box: int        # 1..5
    start: int      # first frame index (session index for real cycles, composite index for synthetic ones)
    end: int        # exclusive end index
    mass: float     # sum of p_box over the frames of the (debounced) run(s)
    n_parts: int = 1  # number of debounced runs merged by collapse


def label_runs(lab):
    """(label, start, end) runs of an int label array (same as a2_cycles.runs_of)."""
    return ac.runs_of(lab)


def visits_with_mass(lab_deb, P, offset: int = 0) -> list:
    """Maximal runs of debounced label b > 0 with evidence E = sum_{t in run} p_b(t). P: (n, 6) posteriors."""
    lab = np.asarray(lab_deb)
    P = np.asarray(P, dtype=float)
    assert P.ndim == 2 and P.shape == (len(lab), N_CLASSES), (P.shape, len(lab))
    out = []
    for b, s, e in label_runs(lab):
        if b > 0:
            m = float(np.sum(P[s:e, b]))
            assert np.isfinite(m), "posterior mass is not finite (frame without a window prediction inside a visit?)"
            out.append(Visit(int(b), int(s) + offset, int(e) + offset, m, 1))
    return out


def collapse(visits, lab=None, offset: int = 0) -> list:
    """Merge consecutive visits of the same box (separated only by none frames); masses add.
    lab (optional): the debounced label array the visit indices refer to (index - offset); then the frames between two
    merged visits are asserted to be none."""
    out = []
    for v in visits:
        if out and out[-1].box == v.box:
            w = out[-1]
            if lab is not None:
                gap = np.asarray(lab)[w.end - offset: v.start - offset]
                assert (gap == 0).all(), "collapse: frames between same-box visits are not all none"
            out[-1] = Visit(w.box, w.start, v.end, w.mass + v.mass, w.n_parts + v.n_parts)
        else:
            out.append(v)
    return out


def reading(visits) -> str:
    return SEP.join(str(v.box) for v in visits)


def cycle_mass(P) -> np.ndarray:
    """Whole-cycle posterior mass per class (length 6), transit frames included."""
    P = np.asarray(P, dtype=float)
    m = P.sum(axis=0)
    assert m.shape == (N_CLASSES,) and np.isfinite(m).all()
    return m


# ------------------------------------------------------------------ 2.3 route and reference masses
def fit_route_mu(train) -> tuple:
    """train: list of dicts with 'vis_c' (collapsed visits) and 'mass' (whole-cycle mass vector).
    Returns (route tuple, mu dict {box: mean whole-cycle mass}, Counter of collapsed readings)."""
    cnt = Counter(tuple(v.box for v in c["vis_c"]) for c in train)
    top = cnt.most_common()
    assert top, "no training cycles"
    assert len(top) == 1 or top[0][1] > top[1][1], f"modal collapsed training reading is not unique: {top[:3]}"
    route = tuple(int(b) for b in top[0][0])
    mu = {b: float(np.mean([float(c["mass"][b]) for c in train])) for b in sorted(set(route))}
    return route, mu, cnt


# ------------------------------------------------------------------ 2.4 multi-match DP with traceback
def align(visits, route, mu, multimatch: bool = True):
    """Evidence-weighted edit distance of the (collapsed) visits v_1..v_n to the route r_1..r_m.
    A[i,j]: step i matched and open after visits 1..j; B[i,j]: step i deleted (or i = 0). B[0,0] = 0, others +inf.
      insertion          A[i,j] <- A[i,j-1] + E_j ;  B[i,j] <- B[i,j-1] + E_j
      match/continuation A[i,j] <- min(A[i-1,j-1], B[i-1,j-1], A[i,j-1])   if b_j = r_i, i >= 1
                         (the last term = multi-match; dropped when multimatch=False -> 'plain' variant)
      deletion           B[i,j] <- min(A[i-1,j], B[i-1,j]) + mu_{r_i}       (i >= 1, also at j = 0)
    S = min(A[m,n], B[m,n]). Returns (S, ops); ops (route order) are tuples
      ('match'|'cont', step i, visit index j, 0.0), ('ins', step i open/deleted before, visit index j, E_j),
      ('del', step i, None, mu_{r_i}); step i is 1-based, visit index j is 0-based."""
    n, m = len(visits), len(route)
    A = np.full((m + 1, n + 1), INF)
    B = np.full((m + 1, n + 1), INF)
    pa, pb = {}, {}
    B[0, 0] = 0.0
    for j in range(n + 1):
        for i in range(m + 1):
            if i >= 1 and j >= 1:
                E = visits[j - 1].mass
                cands = []
                if visits[j - 1].box == route[i - 1]:
                    cands.append((A[i - 1, j - 1], ("A", i - 1, j - 1, "match")))
                    cands.append((B[i - 1, j - 1], ("B", i - 1, j - 1, "match")))
                    if multimatch:
                        cands.append((A[i, j - 1], ("A", i, j - 1, "cont")))
                cands.append((A[i, j - 1] + E, ("A", i, j - 1, "ins")))
                best, ptr = cands[0]
                for val, p in cands[1:]:
                    if val < best:
                        best, ptr = val, p
                A[i, j] = best
                pa[(i, j)] = ptr
            if i == 0 and j == 0:
                continue
            cands = []
            if i >= 1:
                mu_i = float(mu[route[i - 1]])
                cands.append((A[i - 1, j] + mu_i, ("A", i - 1, j, "del")))
                cands.append((B[i - 1, j] + mu_i, ("B", i - 1, j, "del")))
            if j >= 1:
                cands.append((B[i, j - 1] + visits[j - 1].mass, ("B", i, j - 1, "ins")))
            best, ptr = cands[0]
            for val, p in cands[1:]:
                if val < best:
                    best, ptr = val, p
            B[i, j] = best
            pb[(i, j)] = ptr
    if A[m, n] <= B[m, n]:
        st, S = "A", float(A[m, n])
    else:
        st, S = "B", float(B[m, n])
    assert np.isfinite(S)
    ops = []
    i, j = m, n
    while not (st == "B" and i == 0 and j == 0):
        pst, pi, pj, op = (pa if st == "A" else pb)[(i, j)]
        if op in ("match", "cont"):
            ops.append((op, i, j - 1, 0.0))
        elif op == "ins":
            ops.append(("ins", i, j - 1, float(visits[j - 1].mass)))
        else:
            ops.append(("del", i, None, float(mu[route[i - 1]])))
        st, i, j = pst, pi, pj
    ops.reverse()
    tot = float(sum(o[3] for o in ops))
    assert abs(tot - S) <= 1e-9 * max(1.0, abs(S)), (tot, S)
    return S, ops


def score(visits_collapsed, route, mu) -> float:
    """Frozen S_E (multi-match)."""
    return align(visits_collapsed, route, mu, multimatch=True)[0]


# ------------------------------------------------------------------ 2.6 explanation
def explain(S, ops, visits, route, mu, frame_ids=None, fmt="{:.2f}") -> str:
    """'S_E = 8.05: route step Box 2 missing (mu_2 = 8.05); reading 1-5' (+ charged visits with frames and mass).
    frame_ids: optional array mapping a visit index (start/end-1) to a frame id."""
    def fid(k):
        return int(frame_ids[k]) if frame_ids is not None else int(k)
    parts = []
    for op, i, j, c in ops:
        if op == "del":
            b = route[i - 1]
            parts.append(f"route step Box {b} missing (mu_{b} = {fmt.format(c)})")
        elif op == "ins":
            v = visits[j]
            parts.append(f"extra visit Box {v.box} frames {fid(v.start)}-{fid(v.end - 1)} (mass {fmt.format(v.mass)})")
    rd = reading(visits) if visits else "(no visit)"
    if not parts:
        return f"S_E = {fmt.format(S)}: reading {rd} follows route {SEP.join(map(str, route))}"
    return f"S_E = {fmt.format(S)}: " + "; ".join(parts) + f"; reading {rd}"


# ------------------------------------------------------------------ pipeline helpers (shared by 01 and the one-shot)
def e1_inputs(deb_session, P_session, start_idx: int, end_idx: int) -> dict:
    """E1 inputs of one cycle [start_idx, end_idx) of a session: visits (session indices), collapsed visits, mass."""
    deb = np.asarray(deb_session)
    lab = deb[start_idx:end_idx]
    Pc = np.asarray(P_session, dtype=float)[start_idx:end_idx]
    vs = visits_with_mass(lab, Pc, offset=start_idx)
    vc = collapse(vs, lab=deb, offset=0)
    return dict(vis=vs, vis_c=vc, mass=cycle_mass(Pc))


def score_roles(cycles, roles_fn=None, n_folds: int = 5, frame_ids_by_session=None):
    """Phase A protocol for E1. cycles: list of dicts (a2_cycles.build_cycles records + e1_inputs keys).
    For every f: roles from roles_fn (default fa_common.anom_roles via a2_cycles), route/mu fitted on the training
    normal cycles, S_E (frozen multi-match) for train/cal/test, p_E = conf_p(cal S_E, s) on the test rows.
    Returns (rows DataFrame, fold-info list)."""
    import pandas as pd
    roles_fn = roles_fn or ac.fc.anom_roles
    rows, finfo = [], []
    for f in range(n_folds):
        ro = roles_fn(f)
        tr, ca, te = ac._split_roles(cycles, ro)
        route, mu, cnt = fit_route_mu(tr)
        part = []
        for role, lst in (("train", tr), ("cal", ca), ("test", te)):
            for c in lst:
                S, ops = align(c["vis_c"], route, mu, multimatch=True)
                fids = None if frame_ids_by_session is None else frame_ids_by_session[c["session"]]
                part.append(dict(session=c["session"], cycle_idx=int(c["cycle_idx"]),
                                 start_frame_id=int(c["start_frame_id"]), end_frame_id=int(c["end_frame_id"]),
                                 f=f, role=role, atype=c["atype"], natural_dev=bool(c["natural_dev"]),
                                 tof_fold=int(c["fold"]), reading=reading(c["vis"]), reading_c=reading(c["vis_c"]),
                                 n_deleted=sum(o[0] == "del" for o in ops), n_inserted=sum(o[0] == "ins" for o in ops),
                                 S_E=S, explain=explain(S, ops, c["vis_c"], route, mu, frame_ids=fids)))
        d = pd.DataFrame(part)
        cal = d.loc[d["role"] == "cal", "S_E"].to_numpy(float)
        m = d["role"] == "test"
        d["p_E"] = np.nan
        d.loc[m, "p_E"] = ac.conf_p(cal, d.loc[m, "S_E"].to_numpy(float))
        rows.append(d)
        finfo.append(dict(f=f, route=SEP.join(map(str, route)), **{f"mu_{b}": mu[b] for b in sorted(mu)},
                          n_train=len(tr), n_cal=len(ca), n_test=int(m.sum()), cal_max=float(cal.max()) if len(cal) else np.nan,
                          n_cal_pos=int((cal > 0).sum()), train_readings=dict(cnt)))
    return pd.concat(rows, ignore_index=True), finfo


# ------------------------------------------------------------------ shared checks / tabulation (01 exercises them on s01/s02)
GATE_FLOAT_COLS = ("D0", "D1", "D2", "d2_nref", "dur_s", "p0", "p1", "p2", "p3")
GATE_ID_COLS = ("start_frame_id", "end_frame_id", "visit_seq", "atype")


def compare_to_reference(rebuilt, ref, key, tol: float = 1e-9) -> tuple:
    """Row-by-row comparison of a rebuilt Phase A score table with reports/faz_a/04_cycle_scores rows.
    Returns (ok, info dict)."""
    M = rebuilt.merge(ref, on=list(key), how="outer", suffixes=("", "_ref"), indicator=True)
    info = dict(n_rebuilt=len(rebuilt), n_ref=len(ref), n_matched=int((M["_merge"] == "both").sum()))
    ok = info["n_rebuilt"] == info["n_ref"] == info["n_matched"]
    for col in GATE_ID_COLS:
        eq = M[col].astype(str) == M[col + "_ref"].astype(str)
        info[f"mismatch_{col}"] = int((~eq).sum())
        ok &= bool(eq.all())
    for col in GATE_FLOAT_COLS:
        a, b = M[col].to_numpy(float), M[col + "_ref"].to_numpy(float)
        nan_ok = bool((np.isnan(a) == np.isnan(b)).all())
        d = float(np.nanmax(np.abs(a - b))) if (~np.isnan(a)).any() else 0.0
        info[f"maxdiff_{col}"] = d
        info[f"nan_pattern_ok_{col}"] = nan_ok
        ok &= nan_ok and d <= tol
    return bool(ok), info


ATYPE_ORDER = ("normal", "seq135", "seq145", "fast", "slow")
AUC_GROUPS = {"seq135": ["seq135"], "seq145": ["seq145"], "seq(135+145)": ["seq135", "seq145"], "fast": ["fast"],
              "slow": ["slow"], "tempo(fast+slow)": ["fast", "slow"]}


def _al(p, a):
    return np.asarray(p, float) <= a + 1e-9


def oneshot_tables(T, alpha: float = 0.10) -> dict:
    """Tabulation of test rows (one row per (cycle, f)) for section 5. Required columns: session, cycle_idx, f, atype,
    natural_dev, reading_c, S_E, p_E, p2, p3, D3p. Only NORMAL (s02) natural_dev rows are dropped (their count is returned,
    never their status; plan section 5.5); natural_dev rows of the anomaly sessions stay in the rates, as in Phase A.
    Alarm path: p_E channel = 2 p_E <= alpha, p_2 channel = 2 p_2 <= alpha (D3' alarm = either)."""
    import pandas as pd
    nat_normal = T["natural_dev"].astype(bool) & (T["atype"] == "normal")
    n_nat = int(nat_normal.sum())
    T = T[~nat_normal].copy()
    T["a_D3"] = _al(T["p3"], alpha)
    T["a_D3p"] = _al(T["D3p"], alpha)
    T["a_pE"] = _al(T["p_E"], alpha)
    T["a_pE05"] = _al(T["p_E"], alpha / 2)
    T["ch_E"] = _al(2 * T["p_E"], alpha)
    T["ch_2"] = _al(2 * T["p2"], alpha)
    assert (T["a_D3p"] == (T["ch_E"] | T["ch_2"])).all()
    rates = []
    for at in [a for a in ATYPE_ORDER if a in set(T["atype"])] + sorted(set(T["atype"]) - set(ATYPE_ORDER)):
        g = T[T["atype"] == at]
        r = dict(atype=at, n_pairs=len(g), n_cycles=int(g[["session", "cycle_idx"]].drop_duplicates().shape[0]),
                 phaseA_D3=g["a_D3"].mean(), E1_D3p=g["a_D3p"].mean(), pE_alone=g["a_pE"].mean(), pE_005=g["a_pE05"].mean(),
                 path_pE_only=(g["ch_E"] & ~g["ch_2"]).mean(), path_p2_only=(~g["ch_E"] & g["ch_2"]).mean(),
                 path_both=(g["ch_E"] & g["ch_2"]).mean(), S_E_min=g["S_E"].min(), S_E_p10=g["S_E"].quantile(0.10),
                 S_E_med=g["S_E"].median(), S_E_max=g["S_E"].max())
        if at == "normal":
            r["far_se"] = float(np.sqrt(alpha * (1 - alpha) / max(1, len(g))))
        rates.append(r)
    auc = []
    for f, gf in T.groupby("f"):
        neg = gf.loc[gf["atype"] == "normal", "S_E"].to_numpy(float)
        for gname, members in AUC_GROUPS.items():
            pos = gf.loc[gf["atype"].isin(members), "S_E"].to_numpy(float)
            if len(pos) == 0:
                continue
            a, ap = ac.auc_ap(neg, pos)
            auc.append(dict(group=gname, f=int(f), n_neg=len(neg), n_pos=len(pos), auroc=a, auprc=ap))
    auc = pd.DataFrame(auc)
    auc_sum = (auc.groupby("group", sort=False).agg(n_folds=("auroc", "count"), auroc_mean=("auroc", "mean"),
                                                    auroc_sd=("auroc", "std"), auprc_mean=("auprc", "mean")).reset_index()
               if len(auc) else pd.DataFrame())
    readings = (T.groupby(["atype", "reading_c"]).agg(n_pairs=("S_E", "size"), S_E_med=("S_E", "median"),
                                                     pE_005=("a_pE05", "mean"), E1_D3p=("a_D3p", "mean"),
                                                     phaseA_D3=("a_D3", "mean")).reset_index())
    cyc = T.drop_duplicates(["session", "cycle_idx"])
    readings = readings.merge(cyc.groupby(["atype", "reading_c"]).size().rename("n_cycles").reset_index(),
                              on=["atype", "reading_c"], how="left")
    perfold = (T.groupby(["atype", "f"]).agg(n=("S_E", "size"), E1_D3p=("a_D3p", "mean"), phaseA_D3=("a_D3", "mean"),
                                             pE_005=("a_pE05", "mean"), S_E_min=("S_E", "min")).reset_index())
    normal_alarm_sets = dict(
        phaseA_D3=sorted(map(int, T.loc[(T["atype"] == "normal") & T["a_D3"], "cycle_idx"])),
        E1_D3p=sorted(map(int, T.loc[(T["atype"] == "normal") & T["a_D3p"], "cycle_idx"])))
    return dict(rates=pd.DataFrame(rates), auc=auc, auc_sum=auc_sum, readings=readings, perfold=perfold,
                normal_alarm_sets=normal_alarm_sets, n_natural_dev_dropped=n_nat, rows=T)


# ------------------------------------------------------------------ unit tests (run at import)
def _V(spec):
    """[(box, mass), ...] -> visits with consecutive dummy frame indices."""
    out, k = [], 0
    for b, m in spec:
        out.append(Visit(int(b), k, k + 2, float(m), 1))
        k += 3
    return out


def _close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def _unit_tests() -> int:
    R = (1, 2, 5)
    n_ok = 0
    # --- set A: masses chosen so that every row of the 2.4 table is the minimising path (asserted exactly)
    mu = {1: 6.0, 2: 3.0, 5: 8.0}
    E1, E2, E2b, E3, E5 = 4.0, 2.0, 1.5, 1.09, 4.5
    table = [
        ([(1, E1), (2, E2), (5, E5)], 0.0),                       # 1-2-5
        ([(1, E1), (2, E2), (2, E2b), (5, E5)], 0.0),             # 1-2-2-5 (uncollapsed; multi-match)
        ([(1, E1), (5, E5)], mu[2]),                              # 1-5
        ([(1, E1), (5, E5), (2, E2)], mu[2] + E2),                # 1-5-2
        ([(1, E1), (2, E2), (3, E3), (2, E2b), (5, E5)], E3),     # 1-2-3-2-5
        ([(1, E1), (2, E2), (5, E5), (2, E2b)], E2b),             # 1-2-5-2 (trailing)
        ([(1, E1), (2, E2)], mu[5]),                              # 1-2
        ([(2, E2), (5, E5)], mu[1]),                              # 2-5
        ([(1, E1), (3, E3), (5, E5)], mu[2] + E3),                # 1-3-5
    ]
    for spec, exp in table:
        S, ops = align(_V(spec), R, mu)
        assert _close(S, exp), ("unit case", reading(_V(spec)), S, exp)
        n_ok += 1
    # collapse makes 1-2-2-5 the route itself
    assert reading(collapse(_V([(1, E1), (2, E2), (2, E2b), (5, E5)]))) == "1-2-5"
    # traceback content: 1-5 -> one deletion of step 2; 1-3-5 -> deletion of step 2 + insertion of the Box-3 visit
    _, ops = align(_V([(1, E1), (5, E5)]), R, mu)
    assert [o[0] for o in ops] == ["match", "del", "match"] and ops[1][1] == 2
    _, ops = align(_V([(1, E1), (3, E3), (5, E5)]), R, mu)
    assert sorted(o[0] for o in ops) == ["del", "ins", "match", "match"]
    # --- set B: realistic masses; the two rows whose minimising branch depends on the masses obey the general identity
    mu = {1: 8.37, 2: 8.05, 5: 5.64}
    E1, E2, E2b, E3, E5 = 7.5, 4.9, 3.1, 1.09, 5.2
    for spec, exp in [
        ([(1, E1), (5, E5), (2, E2)], min(mu[2] + E2, mu[5] + E5)),   # swap: min of the two edit paths
        ([(1, E1), (2, E2), (5, E5), (2, E2b)], min(E2b, E5 + mu[5])),  # far repeat
        ([(1, E1), (2, E2), (3, E3), (2, E2b), (5, E5)], E3),
        ([(1, E1), (5, E5)], mu[2]),
        ([(1, E1), (2, E2)], mu[5]),
        ([(2, E2), (5, E5)], mu[1]),
        ([(1, E1), (3, E3), (5, E5)], mu[2] + E3),
    ]:
        assert _close(align(_V(spec), R, mu)[0], exp), ("set B", reading(_V(spec)))
        n_ok += 1
    # --- plain variant (synthetic tabulation only): mid-dwell flicker charges flicker + smaller half
    S_plain = align(_V([(1, E1), (2, E2), (3, E3), (2, E2b), (5, E5)]), R, mu, multimatch=False)[0]
    assert _close(S_plain, E3 + min(E2, E2b))
    # --- visits/mass/collapse on a tiny label array
    lab = np.array([0, 1, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 5, 5, 0])
    P = np.zeros((len(lab), N_CLASSES))
    P[np.arange(len(lab)), lab] = 0.8
    P[:, 0] += 0.2
    P[lab == 0, 0] = 1.0
    vs = visits_with_mass(lab, P)
    assert reading(vs) == "1-2-2-5" and _close(vs[1].mass, 1.6) and _close(vs[2].mass, 2.4)
    vc = collapse(vs, lab=lab)
    assert reading(vc) == "1-2-5" and _close(vc[1].mass, 4.0) and vc[1].start == 6 and vc[1].end == 13 and vc[1].n_parts == 2
    assert _close(score(vc, R, {1: 1.0, 2: 1.0, 5: 1.0}), 0.0)
    return n_ok


UNIT_TESTS_PASSED = _unit_tests()

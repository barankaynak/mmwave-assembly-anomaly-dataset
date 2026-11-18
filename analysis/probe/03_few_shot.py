"""03 — Q3: how many NORMAL cycles are enough to calibrate?

Calibration windows = ALL contiguous windows of n complete loops (Q2 selection rules, i.e. Q2.loop_selection) of the
normal session (s02), n in {1,3,5,10,20}, sliding by 1 loop. Evaluation = the remaining complete s02 loops, minus a guard
gap of GUARD loops on each side of the window (adjacent loops are temporally autocorrelated: hand pose / box drift /
operator rhythm carry over, so the immediate neighbours are not independent of the calibration window).
All splits are by loop_id; no random frame/window split anywhere.

Q1 metrics (frame centroids = proxy iii, s01-chosen MeanShift quantile):
  (a) supervised "ToF-calibrated": nearest-centroid accuracy on evaluation box frames, centroids from the calibration
      loops' box frames (ToF labels); cross-session: same centroids -> s05 / s06 box frames.
  (b) unsupervised: MeanShift on the calibration loops' frames (no labels) -> K_found, NMI on evaluation box frames.
Q2 metrics: per-segment d/T mean (calibration loops) with bootstrap 95% CI (resample loops) + distance to the
  full-s02 estimate; tempo T (median) from n loops of s05/s06 vs the full-session value.

Writes reports/probe/q3.md, q3_table.csv, q3_curves.png.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

Q1 = importlib.import_module("01_dwell_modes")
Q2 = importlib.import_module("02_tempo_scaling")

# ------------------------------------------------------------------ pre-declared constants (fixed BEFORE looking at results)
N_GRID = (1, 3, 5, 10, 20)
REF_N = 20                 # accuracy reference for the claim rule
GUARD = 1                  # guard loops on each side of the calibration window
ACC_TOL = 0.02             # claim rule (a): |acc(n) - acc(20)| <= 2 percentage points (mean over windows)
CI_HW_MAX = 0.02           # claim rule (b): bootstrap-CI half-width of the mean d/T < 0.02 for EVERY segment (mean over windows)
CI_LEVEL = 0.95
N_BOOT = 2000
SEED = 20261104
MIN_N_FOR_CI = 2           # a bootstrap over 1 loop is degenerate (zero width) -> CI undefined, criterion (b) NOT met
TEMPO_TOL = 0.10           # descriptive only (not part of the claim rule): share of windows with |T_med error| <= 10%
EXPECTED_K = (3, 4)        # scenario sessions: 3 boxes (+ possibly a merge zone) -> Q1's EXPECTED_K_SCEN
SEGS = Q2.SEGMENTS
SEG_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
SEG_LS = ["-", "-", "-", "--", "--", "--"]

MD: list[str] = []


def emit(text: str = "") -> None:
    MD.append(text)
    print(text, flush=True)


def pm(m, s, f="{:.3f}") -> str:
    if not (np.isfinite(m) and np.isfinite(s)):
        return "nan"
    return (f + " ± " + f).format(m, s)


def sid(s: str) -> str:
    return "s" + s.split("_")[1]


# ------------------------------------------------------------------ loops / windows
def kept_loops(session: str):
    """Q2 selection rules -> (ids ndarray, T Series by loop_id, d/T DataFrame [loop_id x SEGS], selection report)."""
    seg, rep = Q2.loop_selection(session, "tof")
    wide = seg.pivot(index="loop_id", columns="segment", values="d")[SEGS]
    T = seg.drop_duplicates("loop_id").set_index("loop_id")["T"].loc[wide.index]
    dT = wide.div(T, axis=0)
    return np.array(sorted(wide.index), dtype=int), T, dT, rep


def windows(ids: np.ndarray, n: int) -> list[np.ndarray]:
    """All windows of n CONSECUTIVE loop_ids that are all in the kept set (a contiguous stretch of the recording)."""
    out = []
    for i in range(len(ids) - n + 1):
        w = ids[i : i + n]
        if w[-1] - w[0] == n - 1:
            out.append(w)
    return out


def eval_ids(ids: np.ndarray, w: np.ndarray) -> np.ndarray:
    lo, hi = w[0] - GUARD, w[-1] + GUARD
    return ids[(ids < lo) | (ids > hi)]


# ------------------------------------------------------------------ Q1 pieces
def chosen_quantiles(ref: str) -> dict:
    """Re-derive the s01-chosen quantile for the frame-centroid proxy (same rule/code path as Q1's run_grid)."""
    fr = C.load_frames(ref)
    pts = Q1.proxy_points("frame_cent", ref)
    res = {}
    qsel = {}
    for variant, cols in (("xy", ["x", "y"]), ("x", ["x"])):
        X = pts[cols].to_numpy()
        for q in Q1.QUANTILES:
            r = Q1.discover_modes_full(X, q, Q1.SEED, None)
            a = Q1.assign_frames("frame_cent", ref, r["centers"], variant)
            r.update(Q1.evaluate_assignment(fr, r["centers"], a))
            res[("frame_cent", variant, ref, q)] = r
        qsel[variant] = Q1.select_quantile(res, "frame_cent", variant, ref)
    return qsel


def nn_acc(cent: pd.DataFrame, frames: pd.DataFrame) -> float:
    y, p = Q1.nn_classify(cent, frames, 2, False)
    return float((y == p).mean())


def mode_eval(cal: pd.DataFrame, ev: pd.DataFrame, variant: str, q: float) -> dict:
    cols = ["cx", "cy"] if variant == "xy" else ["cx"]
    r = Q1.discover_modes_full(cal[cols].to_numpy(), q, Q1.SEED, None)
    e = Q1.evaluate_assignment(ev, r["centers"])
    return {"K": e["K"], "K_raw": r["K_raw"], "NMI": e["NMI"], "ARI": e["ARI"], "n_eval": e["n_eval"]}


# ------------------------------------------------------------------ Q2 pieces
def boot_hw(dTw: np.ndarray, rng: np.random.Generator):
    """Bootstrap (loop resampling) CI of the per-segment mean d/T -> half-width, lo, hi. n < MIN_N_FOR_CI -> NaN."""
    n = len(dTw)
    if n < MIN_N_FOR_CI:
        nan = np.full(dTw.shape[1], np.nan)
        return nan, nan, nan
    idx = rng.integers(0, n, (N_BOOT, n))
    m = dTw[idx].mean(axis=1)
    a = (1 - CI_LEVEL) / 2 * 100
    lo, hi = np.percentile(m, [a, 100 - a], axis=0)
    return (hi - lo) / 2, lo, hi


def t_hw(dTw: np.ndarray) -> np.ndarray:
    n = len(dTw)
    if n < MIN_N_FOR_CI:
        return np.full(dTw.shape[1], np.nan)
    return st.t.ppf(0.5 + CI_LEVEL / 2, n - 1) * dTw.std(axis=0, ddof=1) / np.sqrt(n)


# ------------------------------------------------------------------ main computation
def run() -> dict:
    sess = C.sessions()
    by = {C.session_info(s)["scenario"]: s for s in sess}
    ref = [s for s in sess if C.session_info(s)["role"] == "train"][0]
    s_norm, s_fast, s_slow = by["normal"], by["fast"], by["slow"]
    qsel = chosen_quantiles(ref)

    fr = {s: C.load_frames(s) for s in (s_norm, s_fast, s_slow)}
    ids, T, dT, rep = kept_loops(s_norm)
    dT_np = dT.to_numpy()
    full_dT = dT_np.mean(axis=0)
    pos = {int(l): i for i, l in enumerate(ids)}
    full_cent = Q1.box_centroids(fr[s_norm][fr[s_norm]["loop_id"].isin(ids)])
    full_acc_cross = {s: nn_acc(full_cent, fr[s]) for s in (s_fast, s_slow)}
    # Q1 sanity: full-session unsupervised fit (all s02 frames) reproduces the Q1 frame_cent number
    sanity = {v: mode_eval(fr[s_norm], fr[s_norm], v, qsel[v]) for v in ("xy", "x")}
    full_kept = {v: mode_eval(fr[s_norm][fr[s_norm]["loop_id"].isin(ids)], fr[s_norm][fr[s_norm]["loop_id"].isin(ids)], v, qsel[v])
                 for v in ("xy", "x")}

    rows = []
    for n in N_GRID:
        wins = windows(ids, n)
        if not wins:
            raise RuntimeError(f"no windows for n={n}")
        for w in wins:
            ev = eval_ids(ids, w)
            assert len(ev) > 0 and not set(w) & set(ev), "calibration/evaluation overlap"
            assert all(abs(int(e) - int(x)) > GUARD for e in ev for x in (w[0], w[-1])), "guard violated"
            f = fr[s_norm]
            cal = f[f["loop_id"].isin(w)]
            evf = f[f["loop_id"].isin(ev)]
            cent = Q1.box_centroids(cal)
            row = {"n": n, "start": int(w[0]), "n_eval_loops": len(ev), "n_cal_frames": len(cal), "n_cal_boxes": len(cent),
                   "n_eval_box_frames": int((evf["label"] > 0).sum()),
                   "acc_s02": nn_acc(cent, evf), "acc_s05": nn_acc(cent, fr[s_fast]), "acc_s06": nn_acc(cent, fr[s_slow])}
            for v in ("xy", "x"):
                m = mode_eval(cal, evf, v, qsel[v])
                row.update({f"K_{v}": m["K"], f"Kraw_{v}": m["K_raw"], f"NMI_{v}": m["NMI"], f"ARI_{v}": m["ARI"]})
            # Q2
            dTw = dT_np[[pos[int(l)] for l in w]]
            dTe = dT_np[[pos[int(l)] for l in ev]]
            rng = np.random.default_rng([SEED, n, int(w[0])])
            hw, lo, hi = boot_hw(dTw, rng)
            hwt = t_hw(dTw)
            mw = dTw.mean(axis=0)
            for j, sg in enumerate(SEGS):
                row[f"mean_{sg}"] = mw[j]
                row[f"hwb_{sg}"] = hw[j]
                row[f"hwt_{sg}"] = hwt[j]
                row[f"dfull_{sg}"] = abs(mw[j] - full_dT[j])
                row[f"dheld_{sg}"] = abs(mw[j] - dTe[:, j].mean())
                row[f"cover_{sg}"] = float(lo[j] <= full_dT[j] <= hi[j]) if np.isfinite(lo[j]) else np.nan
            row["hwb_max"] = np.nanmax(hw) if np.isfinite(hw).any() else np.nan
            row["hwt_max"] = np.nanmax(hwt) if np.isfinite(hwt).any() else np.nan
            rows.append(row)
    W = pd.DataFrame(rows)

    # tempo: T median from n loops of s05 / s06 (their own kept loops, contiguous windows)
    trows = []
    tfull = {}
    for s in (s_fast, s_slow):
        ids_s, T_s, _, rep_s = kept_loops(s)
        tfull[s] = float(np.median(T_s.to_numpy()))
        for n in N_GRID:
            for w in windows(ids_s, n):
                tm = float(np.median(T_s.loc[w].to_numpy()))
                trows.append({"session": sid(s), "n": n, "start": int(w[0]), "T_med": tm, "T_full": tfull[s],
                              "relerr": abs(tm / tfull[s] - 1.0)})
    TW = pd.DataFrame(trows)
    return {"W": W, "TW": TW, "qsel": qsel, "ref": ref, "ids": ids, "T": T, "full_dT": full_dT, "dT_sd": dT_np.std(axis=0, ddof=1), "rep": rep, "sess": (s_norm, s_fast, s_slow),
            "full_acc_cross": full_acc_cross, "sanity": sanity, "full_kept": full_kept, "tfull": tfull}


# ------------------------------------------------------------------ aggregation / claim
def aggregate(R: dict) -> dict:
    W, TW = R["W"], R["TW"]
    g = W.groupby("n")
    A = {"n_win": g.size(), "n_eval_loops": g["n_eval_loops"].mean()}
    for c in ("acc_s02", "acc_s05", "acc_s06", "K_xy", "NMI_xy", "K_x", "NMI_x", "hwb_max", "hwt_max"):
        A[c + "_m"], A[c + "_s"] = g[c].mean(), g[c].std(ddof=0)
    for v in ("xy", "x"):
        A[f"Kok_{v}"] = W.assign(ok=W[f"K_{v}"].isin(EXPECTED_K)).groupby("n")["ok"].mean()
    # claim rule
    acc_ref = A["acc_s02_m"][REF_N]
    rows = []
    for n in N_GRID:
        acc_gap = abs(A["acc_s02_m"][n] - acc_ref)
        hwb = {sg: float(W[W.n == n][f"hwb_{sg}"].mean()) for sg in SEGS}
        hwt = {sg: float(W[W.n == n][f"hwt_{sg}"].mean()) for sg in SEGS}
        worst_b = float(W[W.n == n]["hwb_max"].max()) if W[W.n == n]["hwb_max"].notna().any() else np.nan
        mx_b = max(hwb.values()) if all(np.isfinite(v) for v in hwb.values()) else np.nan
        mx_t = max(hwt.values()) if all(np.isfinite(v) for v in hwt.values()) else np.nan
        acc_ok = bool(acc_gap <= ACC_TOL + 1e-12)
        ci_ok = bool(np.isfinite(mx_b) and mx_b < CI_HW_MAX)
        ci_ok_t = bool(np.isfinite(mx_t) and mx_t < CI_HW_MAX)
        ci_ok_w = bool(np.isfinite(worst_b) and worst_b < CI_HW_MAX)
        bind = max(hwb, key=lambda k: hwb[k]) if np.isfinite(mx_b) else "-"
        rows.append({"n": n, "acc_gap_pp": acc_gap * 100, "acc_ok": acc_ok, "max_seg_hw_boot": mx_b, "binding_segment": bind,
                     "ci_ok": ci_ok, "CLAIM": acc_ok and ci_ok,
                     "max_seg_hw_t": mx_t, "ci_ok_t": ci_ok_t, "CLAIM_t": acc_ok and ci_ok_t,
                     "worst_window_hw_boot": worst_b, "ci_ok_worst": ci_ok_w, "CLAIM_worst": acc_ok and ci_ok_w})
    claim = pd.DataFrame(rows)
    def first(col):
        x = claim[claim[col]]
        return int(x["n"].iloc[0]) if len(x) else None
    N = {"primary": first("CLAIM"), "t": first("CLAIM_t"), "worst": first("CLAIM_worst")}
    # monotonicity: does every larger n in the grid also satisfy the claim?
    mono = None
    if N["primary"] is not None:
        mono = bool(claim[claim["n"] >= N["primary"]]["CLAIM"].all())
    # tempo aggregate
    tg = TW.groupby(["session", "n"])
    T_agg = pd.DataFrame({"n_win": tg.size(), "relerr_m": tg["relerr"].mean(), "relerr_s": tg["relerr"].std(ddof=0),
                          "within_tol": tg["relerr"].apply(lambda x: float((x <= TEMPO_TOL).mean())),
                          "T_med_m": tg["T_med"].mean(), "T_med_s": tg["T_med"].std(ddof=0), "T_full": tg["T_full"].first()}).reset_index()
    return {"A": A, "claim": claim, "N": N, "mono": mono, "T_agg": T_agg}


# ------------------------------------------------------------------ csv
def make_csv(R: dict, G: dict) -> pd.DataFrame:
    W, TW = R["W"], R["TW"]
    rows = []
    def add(metric, scope, n, vals):
        v = np.asarray(vals, float)
        rows.append({"metric": metric, "scope": scope, "n": n, "n_windows": int(np.isfinite(v).sum()), "mean": np.nanmean(v) if np.isfinite(v).any() else np.nan,
                     "sd": np.nanstd(v) if np.isfinite(v).any() else np.nan, "min": np.nanmin(v) if np.isfinite(v).any() else np.nan,
                     "max": np.nanmax(v) if np.isfinite(v).any() else np.nan})
    for n in N_GRID:
        w = W[W.n == n]
        for c, scope in (("acc_s02", "s02 held-out loops"), ("acc_s05", "s05 all box frames"), ("acc_s06", "s06 all box frames")):
            add("nn_accuracy", scope, n, w[c])
        for v in ("xy", "x"):
            add("K_found", f"meanshift_{v}", n, w[f"K_{v}"])
            add("NMI", f"meanshift_{v}", n, w[f"NMI_{v}"])
        add("n_eval_loops", "s02", n, w["n_eval_loops"])
        for sg in SEGS:
            add("dT_mean", sg, n, w[f"mean_{sg}"])
            add("dT_ci_halfwidth_boot", sg, n, w[f"hwb_{sg}"])
            add("dT_ci_halfwidth_t", sg, n, w[f"hwt_{sg}"])
            add("dT_abs_dist_full_s02", sg, n, w[f"dfull_{sg}"])
            add("dT_abs_dist_heldout", sg, n, w[f"dheld_{sg}"])
            add("dT_ci_covers_full", sg, n, w[f"cover_{sg}"])
        for s in sorted(TW["session"].unique()):
            add("T_median_rel_err", s, n, TW[(TW.session == s) & (TW.n == n)]["relerr"])
    out = pd.DataFrame(rows)
    cl = G["claim"].copy()
    cl.insert(0, "metric", "claim_rule")
    return pd.concat([out, cl.rename(columns={"n": "n"})], ignore_index=True, sort=False)


# ------------------------------------------------------------------ plot
def make_plot(R: dict, G: dict) -> None:
    W, TW, A = R["W"], R["TW"], G["A"]
    ns = list(N_GRID)
    fig, ax = plt.subplots(2, 4, figsize=(21, 9.5))
    def band(a, key, color, label, ls="-", marker="o"):
        m, s = A[key + "_m"].reindex(ns).to_numpy(), A[key + "_s"].reindex(ns).to_numpy()
        a.errorbar(ns, m, yerr=s, color=color, marker=marker, ls=ls, capsize=3, lw=1.6, label=label)
    def setx(a):
        a.set_xscale("log"); a.set_xticks(ns); a.set_xticklabels([str(n) for n in ns]); a.set_xlabel("n calibration loops"); a.grid(alpha=0.25)
    # 1 supervised accuracy, s02 held-out
    a = ax[0, 0]
    band(a, "acc_s02", "#0072B2", "s02 held-out loops")
    ref = A["acc_s02_m"][REF_N]
    a.axhspan(ref - ACC_TOL, ref + ACC_TOL, color="#0072B2", alpha=0.12, label="n=20 ± 2 pp")
    a.set_title("(a) ToF-calibrated nearest-centroid accuracy (s02 held-out)", fontsize=10); a.set_ylabel("accuracy"); setx(a); a.legend(fontsize=8)
    # 2 cross-session
    a = ax[0, 1]
    band(a, "acc_s05", "#D55E00", "s05 fast (all box frames)", marker="s")
    band(a, "acc_s06", "#009E73", "s06 slow (all box frames)", marker="^")
    for s, c in ((R["sess"][1], "#D55E00"), (R["sess"][2], "#009E73")):
        a.axhline(R["full_acc_cross"][s], color=c, ls=":", lw=1.2)
    a.set_title("(a') cross-session: centroids from n s02 loops (dotted = all s02 loops)", fontsize=10); a.set_ylabel("accuracy"); setx(a); a.legend(fontsize=8)
    # 3 K_found
    a = ax[0, 2]
    band(a, "K_xy", "#0072B2", "MeanShift 2D (x,y)")
    band(a, "K_x", "#E69F00", "MeanShift 1D (x)", marker="s")
    a.axhspan(min(EXPECTED_K) - 0.5, max(EXPECTED_K) + 0.5, color="#999999", alpha=0.15, label="expected K {3,4}")
    a.set_title("(b) unsupervised K_found vs n", fontsize=10); a.set_ylabel("K_found"); setx(a); a.legend(fontsize=8)
    # 4 NMI
    a = ax[0, 3]
    band(a, "NMI_xy", "#0072B2", "MeanShift 2D (x,y)")
    band(a, "NMI_x", "#E69F00", "MeanShift 1D (x)", marker="s")
    a.axhline(Q1.NMI_PASS, color="#999999", ls="--", lw=1, label=f"Q1 gate NMI={Q1.NMI_PASS}")
    a.set_title("(b) unsupervised NMI on held-out box frames", fontsize=10); a.set_ylabel("NMI"); setx(a); a.legend(fontsize=8)
    # 5 CI half-width (bootstrap)
    a = ax[1, 0]
    for j, sg in enumerate(SEGS):
        m = [W[W.n == n][f"hwb_{sg}"].mean() for n in ns]
        a.plot(ns, m, color=SEG_COLORS[j], ls=SEG_LS[j], marker="o", lw=1.6, label=sg)
    a.axhline(CI_HW_MAX, color="k", ls="--", lw=1.2, label=f"{CI_HW_MAX} (claim threshold)")
    a.set_title("(c) d/T mean: bootstrap 95% CI half-width (mean over windows; n=1 undefined)", fontsize=10); a.set_ylabel("half-width of d/T"); setx(a); a.legend(fontsize=7, ncol=2)
    # 6 CI half-width t-based
    a = ax[1, 1]
    for j, sg in enumerate(SEGS):
        m = [W[W.n == n][f"hwt_{sg}"].mean() for n in ns]
        a.plot(ns, m, color=SEG_COLORS[j], ls=SEG_LS[j], marker="o", lw=1.6, label=sg)
    a.axhline(CI_HW_MAX, color="k", ls="--", lw=1.2)
    a.set_title("(c') same, Student-t interval (sensitivity; honest for small n)", fontsize=10); a.set_ylabel("half-width of d/T"); setx(a)
    # 7 distance to full
    a = ax[1, 2]
    for j, sg in enumerate(SEGS):
        m = np.array([W[W.n == n][f"dfull_{sg}"].mean() for n in ns])
        s = np.array([W[W.n == n][f"dfull_{sg}"].std(ddof=0) for n in ns])
        a.errorbar(ns, m, yerr=s, color=SEG_COLORS[j], ls=SEG_LS[j], marker="o", capsize=2, lw=1.4, label=sg)
    a.set_title("(c) |mean d/T(n loops) − full-s02 mean| (mean ± sd over windows)", fontsize=10); a.set_ylabel("abs. distance"); setx(a); a.legend(fontsize=7, ncol=2)
    # 8 tempo
    a = ax[1, 3]
    for s, c, mk in (("s05", "#D55E00", "s"), ("s06", "#009E73", "^")):
        t = G["T_agg"][G["T_agg"]["session"] == s].set_index("n").reindex(ns)
        a.errorbar(ns, t["relerr_m"], yerr=t["relerr_s"], color=c, marker=mk, capsize=3, lw=1.6, label=f"{s} ({'fast' if s == 's05' else 'slow'})")
    a.axhline(TEMPO_TOL, color="#999999", ls="--", lw=1, label=f"{TEMPO_TOL:.0%} (descriptive)")
    a.set_title("(c) tempo: |median T(n loops) / median T(full session) − 1|", fontsize=10); a.set_ylabel("relative error"); setx(a); a.legend(fontsize=8)
    N = G["N"]["primary"]
    fig.suptitle(f"Q3: calibration vs n normal loops (s02; mean ± sd over all contiguous windows, guard={GUARD}).  "
                 f"Claim rule -> N = {N if N is not None else 'not reached in grid'}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(C.REPORTS / "q3_curves.png", dpi=110)
    plt.close(fig)


def full_dT_txt(R: dict, sg: str) -> str:
    v = R["full_dT"][SEGS.index(sg)]
    return f"ort. d/T {v:.3f}"


# ------------------------------------------------------------------ report
def report(R: dict, G: dict) -> None:
    W, TW, A, claim = R["W"], R["TW"], G["A"], G["claim"]
    s_norm, s_fast, s_slow = R["sess"]
    ids, T, rep = R["ids"], R["T"], R["rep"]
    Tmed = float(np.median(T.to_numpy()))
    emit("# Q3 — Kaç normal döngü kalibrasyona yeter?")
    emit()
    emit("Kalibrasyon yalnız **s02** (normal, 1-2-5) döngülerinden; n ∈ " + str(list(N_GRID)) + " ardışık tam döngü, **tüm kayan pencereler** (adım 1 loop). "
         f"Değerlendirme: pencere dışında kalan tam s02 döngüleri, pencerenin iki yanından **{GUARD} döngülük guard boşluğu** çıkarılarak "
         "(komşu döngüler otokorelasyonlu: el pozu, kutu kayması, operatör ritmi taşınıyor; bu yüzden hemen bitişik döngü bağımsız sayılmaz). "
         "Tüm bölmeler `loop_id` bazlı; rastgele kare/pencere bölmesi yok.")
    emit()
    emit("**Döngü seçimi** (Q2 kuralları, `02_tempo_scaling.loop_selection`): loop 0 ve oturumun son loop'u çıkarılır, yalnız box_order beklenen diziye TAM eşit loop'lar kalır. "
         f"s02'de kalan loop sayısı = **{len(ids)}**, medyan T = **{Tmed:.2f} s** (loop_id aralığı {ids.min()}–{ids.max()}; "
         f"dışlananlar: " + (", ".join(f"{l}:{o or 'boş'}" for l, o in rep['excluded_nonexact']) or "yok") + "). "
         "Pencere = ardışık loop_id'ler (dışlanan bir loop pencereyi böler), yani zamanda kesintisiz bir kayıt parçası.")
    emit()
    emit("**Sabit kriterler** (sonuçlara bakmadan önce dosya başında sabitlendi): "
         f"CLAIM = (a) ToF-kalibreli en-yakın-merkez doğruluğu (s02 held-out) n=20'ye göre ≤ {ACC_TOL * 100:.0f} puan içinde (pencereler üzerinden ortalama) "
         f"VE (b) her segment için d/T ortalamasının bootstrap %{int(CI_LEVEL * 100)} GA yarı genişliği < {CI_HW_MAX} (pencereler üzerinden ortalama; "
         f"loop yeniden örnekleme, {N_BOOT} tekrar, sabit tohum {SEED}). n=1'de bootstrap dejenere (genişlik 0), bu yüzden GA tanımsız ve (b) **karşılanmış sayılmaz**. "
         "Küçük n'de yüzdelik-bootstrap GA dar kalır (anti-konservatif); bu yüzden Student-t GA ve en kötü pencere duyarlılık olarak ayrıca raporlanır (CLAIM'i değiştirmez, birincil bootstrap).")
    emit()
    q = R["qsel"]
    emit(f"**Q1 bağlantısı**: proxy (iii) kare ağırlık merkezleri, MeanShift (`01_dwell_modes.discover_modes_full`, bin_seeding, MIN_MODE_SHARE={Q1.MIN_MODE_SHARE}); "
         f"quantile s01'de Q1 kuralıyla yeniden türetildi: 2B (x,y) → q={q['xy']}, 1B (x) → q={q['x']} (Q1 raporuyla aynı olmalı: 0.1 / 0.2). "
         f"Referans (tüm s02 kareleri, Q1 sayıları): 2B K={R['sanity']['xy']['K']} NMI={R['sanity']['xy']['NMI']:.3f}; "
         f"1B K={R['sanity']['x']['K']} NMI={R['sanity']['x']['NMI']:.3f}. Yalnız kalan s02 döngüleri (tam-döngü seçimi) ile in-sample: "
         f"2B K={R['full_kept']['xy']['K']} NMI={R['full_kept']['xy']['NMI']:.3f}; 1B K={R['full_kept']['x']['K']} NMI={R['full_kept']['x']['NMI']:.3f}.")
    emit()
    emit("Not: pencereler örtüşüyor (adım 1), bu yüzden \"± sd\" pencere konumuna bağlı değişkenliği gösterir, bağımsız tekrar değildir; n büyüdükçe pencere sayısı ve değerlendirme kümesi küçülür.")
    emit()

    # ---------------- table 0: windows
    emit("## 0. Pencere sayıları")
    emit()
    t0 = pd.DataFrame({"n": [str(n) for n in N_GRID], "pencere": [str(int(A['n_win'][n])) for n in N_GRID],
                       "değerlendirme loop (ort.)": [float(A['n_eval_loops'][n]) for n in N_GRID],
                       "kalibrasyon kare (ort.)": [float(W[W.n == n]['n_cal_frames'].mean()) for n in N_GRID],
                       "eval kutu karesi (ort.)": [float(W[W.n == n]['n_eval_box_frames'].mean()) for n in N_GRID],
                       "kalibrasyonda kutu sayısı (min)": [str(int(W[W.n == n]['n_cal_boxes'].min())) for n in N_GRID]})
    emit(C.md_table(t0, "{:.1f}"))
    emit()

    # ---------------- Q1 supervised
    emit("## 1. Q1 / (a) ToF-kalibreli en-yakın-merkez doğruluğu (ortalama ± sd, pencereler üzerinden)")
    emit()
    t1 = pd.DataFrame({"n": list(N_GRID),
                       "s02 held-out": [pm(A['acc_s02_m'][n], A['acc_s02_s'][n]) for n in N_GRID],
                       "s05 fast (çapraz)": [pm(A['acc_s05_m'][n], A['acc_s05_s'][n]) for n in N_GRID],
                       "s06 slow (çapraz)": [pm(A['acc_s06_m'][n], A['acc_s06_s'][n]) for n in N_GRID],
                       "n=20'ye fark (puan)": [f"{(A['acc_s02_m'][n] - A['acc_s02_m'][REF_N]) * 100:+.2f}" for n in N_GRID],
                       "min (s02)": [float(W[W.n == n]['acc_s02'].min()) for n in N_GRID]})
    emit(C.md_table(t1, "{:.3f}"))
    emit()
    emit("Çapraz oturum referansı (tüm s02 tam döngülerinden merkezler): " +
         f"s05 doğruluk = {R['full_acc_cross'][s_fast]:.3f}, s06 doğruluk = {R['full_acc_cross'][s_slow]:.3f}.")
    emit()

    # ---------------- Q1 unsupervised
    emit("## 2. Q1 / (b) Etiketsiz MeanShift: K_found ve held-out NMI")
    emit()
    t2 = pd.DataFrame({"n": list(N_GRID),
                       "K 2B": [pm(A['K_xy_m'][n], A['K_xy_s'][n], "{:.2f}") for n in N_GRID],
                       "K∈{3,4} oranı 2B": [float(A['Kok_xy'][n]) for n in N_GRID],
                       "NMI 2B": [pm(A['NMI_xy_m'][n], A['NMI_xy_s'][n]) for n in N_GRID],
                       "K 1B": [pm(A['K_x_m'][n], A['K_x_s'][n], "{:.2f}") for n in N_GRID],
                       "K∈{3,4} oranı 1B": [float(A['Kok_x'][n]) for n in N_GRID],
                       "NMI 1B": [pm(A['NMI_x_m'][n], A['NMI_x_s'][n]) for n in N_GRID]})
    emit(C.md_table(t2, "{:.3f}"))
    emit()
    for v, nm in (("xy", "2B"), ("x", "1B")):
        emit(f"K_found dağılımı ({nm}; K: pencere sayısı): " + "; ".join(
            f"n={n}: " + ", ".join(f"{int(k)}:{int(c)}" for k, c in sorted(W[W.n == n][f"K_{v}"].value_counts().items())) for n in N_GRID))
        emit()
    emit("Betimleyici (sonradan, CLAIM'e dahil değil): K∈{3,4} oranı ≥ 0.90 olan en küçük n — 2B: " +
         str(next((n for n in N_GRID if A['Kok_xy'][n] >= 0.90), 'ızgarada yok')) + "; 1B: " +
         str(next((n for n in N_GRID if A['Kok_x'][n] >= 0.90), 'ızgarada yok')) + ".")
    emit()

    # ---------------- Q2
    emit("## 3. Q2 — d/T ortalaması: GA, tam-s02 tahminine uzaklık")
    emit()
    full = R["full_dT"]
    emit("Tam s02 (tüm tam döngüler) d/T ortalaması: " + ", ".join(f"{sg} {full[j]:.4f}" for j, sg in enumerate(SEGS)) + f" (toplam {full.sum():.3f}).")
    emit()
    def wide(prefix, f="{:.4f}", sd=True):
        rows = []
        for n in N_GRID:
            r = {"n": n}
            w = W[W.n == n]
            for sg in SEGS:
                v = w[f"{prefix}_{sg}"]
                r[sg] = pm(v.mean(), v.std(ddof=0), f) if sd else (f.format(v.mean()) if v.notna().any() else "nan")
            rows.append(r)
        return pd.DataFrame(rows)
    emit("### 3a. Bootstrap %95 GA yarı genişliği (ortalama ± sd; eşik < " + str(CI_HW_MAX) + "; n=1 tanımsız)")
    emit()
    emit(C.md_table(wide("hwb")))
    emit()
    emit("### 3b. Student-t %95 GA yarı genişliği (duyarlılık; ortalama ± sd)")
    emit()
    emit(C.md_table(wide("hwt")))
    emit()
    emit("### 3c. |ortalama d/T (n loop) − tam-s02 ortalaması| (ortalama ± sd; pencere tam-s02'nin parçası, spec gereği)")
    emit()
    emit(C.md_table(wide("dfull")))
    emit()
    emit("### 3d. |ortalama d/T (n loop) − held-out loop'ların ortalaması| (örnek dışı; ortalama ± sd)")
    emit()
    emit(C.md_table(wide("dheld")))
    emit()
    emit("### 3e. Bootstrap GA'nın tam-s02 ortalamasını kapsama oranı (nominal 0.95; pencere başına)")
    emit()
    emit(C.md_table(wide("cover", "{:.3f}", sd=False)))
    emit()
    emit("### 3f. Pencerelerde ortalama d/T (n'e göre; yanlılık kontrolü)")
    emit()
    emit(C.md_table(wide("mean", "{:.4f}", sd=False)))
    emit()

    # ---------------- tempo
    emit("## 4. Tempo: n loop ile medyan T tahmini (s05 fast / s06 slow, kendi oturumlarının tam-oturum medyanına göre)")
    emit()
    Ta = G["T_agg"]
    t4 = pd.DataFrame({"session": Ta["session"], "n": Ta["n"], "pencere": Ta["n_win"], "T_full (s)": Ta["T_full"],
                       "T_med(n) ort. ± sd (s)": [pm(a, b, "{:.3f}") for a, b in zip(Ta["T_med_m"], Ta["T_med_s"])],
                       "|bağıl hata| ort. ± sd": [pm(a, b) for a, b in zip(Ta["relerr_m"], Ta["relerr_s"])],
                       f"pencere oranı hata ≤ {TEMPO_TOL:.0%}": Ta["within_tol"]})
    emit(C.md_table(t4, "{:.3f}"))
    emit()
    emit(f"Referans: s02 medyan T = {Tmed:.3f} s; s05 T_full = {R['tfull'][s_fast]:.3f} s; s06 T_full = {R['tfull'][s_slow]:.3f} s.")
    emit()

    # ---------------- claim
    emit("## 5. CLAIM kuralı")
    emit()
    c = claim.copy()
    c5 = pd.DataFrame({"n": c["n"], "acc farkı (puan)": c["acc_gap_pp"], "(a) ≤2 puan": c["acc_ok"].map({True: "OK", False: "x"}),
                       "max segment GA (bootstrap)": c["max_seg_hw_boot"], "bağlayıcı segment": c["binding_segment"],
                       "(b) <0.02": c["ci_ok"].map({True: "OK", False: "x"}), "CLAIM": c["CLAIM"].map({True: "GEÇTİ", False: "x"}),
                       "max segment GA (t)": c["max_seg_hw_t"], "CLAIM (t-GA)": c["CLAIM_t"].map({True: "GEÇTİ", False: "x"}),
                       "en kötü pencere GA (boot)": c["worst_window_hw_boot"], "CLAIM (en kötü pencere)": c["CLAIM_worst"].map({True: "GEÇTİ", False: "x"})})
    emit(C.md_table(c5, "{:.4f}"))
    emit()
    N = G["N"]
    def claim_str(n):
        return f"**{n} döngüde kalibre olur (~{n} × {Tmed:.1f} s ≈ {n * Tmed:.0f} s ≈ {n * Tmed / 60:.1f} dk)**" if n is not None else "**n ≤ 20 ızgarasında kural karşılanmadı**"
    emit(f"### Sonuç (birincil kural, bootstrap GA): {claim_str(N['primary'])}")
    emit()
    emit(f"- Duyarlılık — Student-t GA: N = {N['t']}; en kötü pencere (bootstrap): N = {N['worst']}.")
    if G["mono"] is not None:
        emit(f"- Monotonluk: birincil N'den büyük tüm ızgara noktaları da kuralı geçiyor mu? {'evet' if G['mono'] else 'HAYIR'}.")
    a_ok = [int(n) for n in c[c["acc_ok"]]["n"]]
    b_ok = [int(n) for n in c[c["ci_ok"]]["n"]]
    emit(f"- (a) tek başına geçen n: {a_ok}; (b) tek başına geçen n: {b_ok}.")
    emit()
    sd = R["dT_sd"]
    nreq = [(1.96 * sd[j] / CI_HW_MAX) ** 2 for j in range(len(SEGS))]
    emit("**Betimleyici ekstrapolasyon (sonradan; CLAIM değil, ızgara dışı)**: normal yaklaşım n_gerek ≈ (1.96·sd/0.02)² ile, tam-s02 d/T standart sapmalarından (loop düzeyi) segment başına: " +
         ", ".join(f"{sg}: sd={sd[j]:.4f} → n≈{nreq[j]:.1f}" for j, sg in enumerate(SEGS)) +
         f". En büyük gereksinim n≈{max(nreq):.1f} loop (~{max(nreq) * Tmed:.0f} s ≈ {max(nreq) * Tmed / 60:.1f} dk). "
         f"0.02'lik mutlak eşik d/T'si küçük segmentlerde (dwell_5 {full_dT_txt(R, 'dwell_5')}) göreli olarak gevşek, büyük gap_b5'te ({full_dT_txt(R, 'gap_b5')}) göreli olarak sıkı; tek mutlak eşik bağlayıcı segmenti belirliyor.")
    emit()

    # ---------------- interpretation (data-driven)
    emit("## 6. Yorum")
    emit()
    a1 = A["acc_s02_m"][1]
    emit(f"- **Yapı (konum/kutu) çok az döngüyle oturuyor**: ToF-kalibreli en-yakın-merkez doğruluğu n=1'de {a1:.3f}, n=3'te {A['acc_s02_m'][3]:.3f}, n=5'te {A['acc_s02_m'][5]:.3f}, "
         f"n=20'de {A['acc_s02_m'][20]:.3f} (s02 held-out). n=1 ile n=20 arası fark {(a1 - A['acc_s02_m'][20]) * 100:+.2f} puan; en küçük n için kural (a) "
         f"{'zaten sağlanıyor' if abs(a1 - A['acc_s02_m'][20]) <= ACC_TOL else 'sağlanmıyor'}.")
    emit(f"- **Zamanlama (d/T) belirleyici darboğaz**: bağlayıcı segment n={N['primary'] if N['primary'] is not None else 20}'de "
         f"{c[c['n'] == (N['primary'] if N['primary'] is not None else 20)]['binding_segment'].iloc[0]} "
         f"(max GA yarı genişliği {c[c['n'] == (N['primary'] if N['primary'] is not None else 20)]['max_seg_hw_boot'].iloc[0]:.4f}). "
         f"n=3'te max bootstrap GA = {c[c['n'] == 3]['max_seg_hw_boot'].iloc[0]:.4f}, n=5'te {c[c['n'] == 5]['max_seg_hw_boot'].iloc[0]:.4f}, n=10'da {c[c['n'] == 10]['max_seg_hw_boot'].iloc[0]:.4f}, "
         f"n=20'de {c[c['n'] == 20]['max_seg_hw_boot'].iloc[0]:.4f}; Student-t ile aynı n'lerde {c[c['n'] == 3]['max_seg_hw_t'].iloc[0]:.4f} / {c[c['n'] == 5]['max_seg_hw_t'].iloc[0]:.4f} / "
         f"{c[c['n'] == 10]['max_seg_hw_t'].iloc[0]:.4f} / {c[c['n'] == 20]['max_seg_hw_t'].iloc[0]:.4f}.")
    emit(f"- **Etiketsiz yol (2B, s01'de seçilen q={R['qsel']['xy']})**: K_found ortalaması n=1: {A['K_xy_m'][1]:.2f}, n=3: {A['K_xy_m'][3]:.2f}, n=5: {A['K_xy_m'][5]:.2f}, n=10: {A['K_xy_m'][10]:.2f}, n=20: {A['K_xy_m'][20]:.2f}; "
         f"K∈{{3,4}} oranı {A['Kok_xy'][1]:.2f}/{A['Kok_xy'][3]:.2f}/{A['Kok_xy'][5]:.2f}/{A['Kok_xy'][10]:.2f}/{A['Kok_xy'][20]:.2f}; NMI {A['NMI_xy_m'][1]:.3f}/{A['NMI_xy_m'][3]:.3f}/{A['NMI_xy_m'][5]:.3f}/{A['NMI_xy_m'][10]:.3f}/{A['NMI_xy_m'][20]:.3f}. "
         f"Yani 2B etiketsiz mod keşfi, bu quantile ile n=20'de bile aşırı bölüyor (K≈{A['K_xy_m'][20]:.1f} > 3 kutu); NMI yine de n ile artıyor (mod sayısı yanlış olsa da atamalar kutularla uyumlu hale geliyor).")
    emit(f"- **Etiketsiz yol (1B x, q={R['qsel']['x']})**: K ortalaması {A['K_x_m'][1]:.2f}/{A['K_x_m'][3]:.2f}/{A['K_x_m'][5]:.2f}/{A['K_x_m'][10]:.2f}/{A['K_x_m'][20]:.2f}; "
         f"K∈{{3,4}} oranı {A['Kok_x'][1]:.2f}/{A['Kok_x'][3]:.2f}/{A['Kok_x'][5]:.2f}/{A['Kok_x'][10]:.2f}/{A['Kok_x'][20]:.2f}; NMI {A['NMI_x_m'][1]:.3f}/{A['NMI_x_m'][3]:.3f}/{A['NMI_x_m'][5]:.3f}/{A['NMI_x_m'][10]:.3f}/{A['NMI_x_m'][20]:.3f}. "
         f"Etiketli (ToF) yol n=1'de n=20'nin yalnız {(A['acc_s02_m'][20] - A['acc_s02_m'][1]) * 100:.2f} puan gerisinde; etiketsiz yol ise n≈10'a kadar kararsız.")
    emit(f"- **Kapsama**: bootstrap GA'nın tam-s02 ortalamasını kapsama oranı (segment ortalaması) n=3: {np.nanmean([W[W.n == 3][f'cover_{sg}'].mean() for sg in SEGS]):.2f}, "
         f"n=5: {np.nanmean([W[W.n == 5][f'cover_{sg}'].mean() for sg in SEGS]):.2f}, n=10: {np.nanmean([W[W.n == 10][f'cover_{sg}'].mean() for sg in SEGS]):.2f}, "
         f"n=20: {np.nanmean([W[W.n == 20][f'cover_{sg}'].mean() for sg in SEGS]):.2f} (nominal 0.95; pencere tam-s02'nin parçası olduğu için n=20'de yapay olarak yüksek, küçük n'de bootstrap anti-konservatif).")
    emit()
    emit("Dosyalar: `q3_curves.png`, `q3_table.csv`, `q3.md`.")


def main() -> None:
    np.random.seed(SEED)
    R = run()
    G = aggregate(R)
    report(R, G)
    make_plot(R, G)
    make_csv(R, G).to_csv(C.REPORTS / "q3_table.csv", index=False)
    (C.REPORTS / "q3.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
    print("[q3] done", flush=True)


if __name__ == "__main__":
    main()

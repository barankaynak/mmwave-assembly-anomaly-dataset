"""02 — Q2 / H2: do dwell and gap durations scale with a per-session tempo, d_{c,i} = tau_c * delta_i ?

Writes reports/probe/q2.md, q2_table.csv, q2_boxplots.png.
Reusable (Q3 imports via importlib.import_module("02_tempo_scaling")):
    loop_segments(session, source="tof") -> DataFrame [session, loop_id, segment, d, T]
    loop_selection(session, source="tof") -> (segments DataFrame, selection-report dict)
    scaling_stats(df, sessions=None, n_boot=...) -> dict
Segment types per loop k with expected sequence 1-b-5:
    dwell_1, gap_1b, dwell_b, gap_b5 (contains the in-air join; ToF cannot separate it), dwell_5,
    gap_51 (end of Box5 -> NEXT loop's Box1 entry; the loop boundary is the Box1 entry, so it lies inside loop k).
T = Box1 entry -> next Box1 entry = loop span = sum of the six segments.
"""
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REPORTS, load_frames, load_tof_raw, loops, md_table, session_info, sessions, tof_segments

# ------------------------------------------------------------------ pre-specified criteria (fixed BEFORE looking at results)
ETA2_INTERACTION_MAX = 0.06   # H2 pass needs interaction partial eta^2 < this ...
EPS2_SEGMENT_MAX = 0.06       # ... AND every segment's KW epsilon^2 on d/T < this
INDEX_CI = 0.95               # disproportionality flag: bootstrap CI of (d ratio / T ratio) excludes 1
N_BOOT = 2000
SEED = 20261104
SEGMENTS = ["dwell_1", "gap_1b", "dwell_b", "gap_b5", "dwell_5", "gap_51"]
PRIMARY = ["normal", "fast", "slow"]          # same sequence 1-2-5
SECONDARY = ["seq135", "seq145"]
SCEN_ORDER = ["fast", "normal", "slow", "seq135", "seq145"]
COLORS = {"fast": "#d95f02", "normal": "#1b9e77", "slow": "#7570b3", "seq135": "#999999", "seq145": "#bbbbbb"}
IDENTITY_TOL = 1e-9

out = []
def emit(s=""):
    print(s)
    out.append(str(s))


# ------------------------------------------------------------------ segment construction
def loop_selection(session, source="tof"):
    """Build per-loop segment durations for the exactly-expected-sequence loops. Returns (df, report)."""
    info = session_info(session)
    exp = info["expected_seq"]
    assert exp is not None and len(exp) == 3, f"{session}: need a 3-box expected sequence"
    seg = tof_segments(session, source=source).sort_values("start").reset_index(drop=True)
    fr = load_frames(session)
    last_lid = int(fr["loop_id"].max())
    all_lids = sorted(int(x) for x in fr["loop_id"].unique())
    rep = {"session": session, "n_loops": len(all_lids), "loop0": 0 in all_lids, "last_loop": last_lid,
           "excluded_nonexact": [], "excluded_nofollow": [], "n_kept": 0}
    rows = []
    # first Box1 segment of each loop (start = Box1 entry)
    first1 = {}
    for lid, g in seg[seg["label"] == 1].groupby("loop_id"):
        first1[int(lid)] = g.iloc[0]
    for lid in all_lids:
        if lid == 0:
            continue                                    # pre-start fragment
        if lid == last_lid:
            continue                                    # truncated: no successor Box1 entry
        g = seg[(seg["loop_id"] == lid) & (seg["label"] > 0)]
        order = tuple(int(v) for v in g["label"])
        if order != tuple(exp):
            rep["excluded_nonexact"].append((lid, "-".join(map(str, order))))
            continue
        nxt = first1.get(lid + 1)
        if nxt is None:
            rep["excluded_nofollow"].append(lid)
            continue
        b1, bb, b5 = (g.iloc[0], g.iloc[1], g.iloc[2])
        t_next = float(nxt["start"])
        T = t_next - float(b1["start"])
        d = {
            "dwell_1": b1["end"] - b1["start"],
            "gap_1b": bb["start"] - b1["end"],
            "dwell_b": bb["end"] - bb["start"],
            "gap_b5": b5["start"] - bb["end"],
            "dwell_5": b5["end"] - b5["start"],
            "gap_51": t_next - b5["end"],
        }
        s_sum = float(sum(d.values()))
        if abs(s_sum - T) > IDENTITY_TOL:
            rep.setdefault("identity_viol", []).append((lid, s_sum, T))
        for k, v in d.items():
            rows.append({"session": session, "loop_id": lid, "segment": k, "d": float(v), "T": float(T)})
    rep["n_kept"] = len({r["loop_id"] for r in rows})
    return pd.DataFrame(rows), rep


def loop_segments(session, source="tof"):
    """DataFrame [session, loop_id, segment, d, T] (seconds); see module docstring."""
    return loop_selection(session, source=source)[0]


# ------------------------------------------------------------------ statistics
def _eta2_table(df):
    """Two-way ANOVA (Type II) on log d ~ session * segment; partial eta^2 per effect."""
    x = df[df["d"] > 0].copy()
    x["logd"] = np.log(x["d"])
    fit = smf.ols("logd ~ C(session) * C(segment)", data=x).fit()
    a = sm.stats.anova_lm(fit, typ=2)
    ssr = a.loc["Residual", "sum_sq"]
    res = {}
    for nm, key in [("C(session)", "session"), ("C(segment)", "segment"), ("C(session):C(segment)", "interaction")]:
        ss = a.loc[nm, "sum_sq"]
        res[key] = {"SS": float(ss), "df": float(a.loc[nm, "df"]), "F": float(a.loc[nm, "F"]), "p": float(a.loc[nm, "PR(>F)"]),
                    "eta2p": float(ss / (ss + ssr))}
    res["resid"] = {"SS": float(ssr), "df": float(a.loc["Residual", "df"])}
    res["n_obs"] = int(len(x))
    res["n_zero_dropped"] = int((df["d"] <= 0).sum())
    return res


def _kw(groups):
    groups = [np.asarray(g, float) for g in groups if len(g) > 0]
    k, n = len(groups), sum(len(g) for g in groups)
    if k < 2:
        return {"H": np.nan, "p": np.nan, "eps2": np.nan, "k": k, "n": n}
    H, p = st.kruskal(*groups)
    return {"H": float(H), "p": float(p), "eps2": float((H - k + 1) / (n - k)), "k": k, "n": int(n)}


def scaling_stats(df, sessions=None, n_boot=N_BOOT, seed=SEED):
    """df: loop_segments() output (possibly several sessions concatenated; needs a 'scenario' column or
    session names that session_info can parse). `sessions`: restrict to these session names (default: all in df).
    Returns dict with: table (per segment x session), ratios (per segment, Fast & Slow vs Normal), anova (partial eta^2),
    kw (per segment, d/T and raw d), loo_interaction (leave-one-segment-out), elasticity, within."""
    df = df.copy()
    if sessions is not None:
        df = df[df["session"].isin(sessions)]
    df["scenario"] = df["session"].map(lambda s: session_info(s)["scenario"])
    df["dT"] = df["d"] / df["T"]
    scen_sess = df.groupby("scenario")["session"].first().to_dict()
    scen = [s for s in SCEN_ORDER if s in scen_sess]
    R = {"scenarios": scen}

    # per segment x scenario table
    rows = []
    for seg in SEGMENTS:
        for sc in scen:
            x = df[(df["segment"] == seg) & (df["scenario"] == sc)]
            rows.append({"segment": seg, "scenario": sc, "n": len(x), "med_d": x["d"].median(), "iqr_d": x["d"].quantile(.75) - x["d"].quantile(.25),
                         "med_dT": x["dT"].median(), "iqr_dT": x["dT"].quantile(.75) - x["dT"].quantile(.25), "med_T": x["T"].median()})
    R["table"] = pd.DataFrame(rows)

    # ANOVA on the scenarios that share the segment semantic (primary three; caller decides)
    R["anova"] = _eta2_table(df)
    R["loo_interaction"] = {}
    for seg in SEGMENTS:
        R["loo_interaction"][seg] = _eta2_table(df[df["segment"] != seg])["interaction"]["eta2p"]

    # Kruskal-Wallis per segment across scenarios
    R["kw_dT"], R["kw_d"] = {}, {}
    for seg in SEGMENTS:
        R["kw_dT"][seg] = _kw([df[(df["segment"] == seg) & (df["scenario"] == sc)]["dT"].to_numpy() for sc in scen])
        R["kw_d"][seg] = _kw([df[(df["segment"] == seg) & (df["scenario"] == sc)]["d"].to_numpy() for sc in scen])

    # ratios to Normal: d ratio vs T ratio, index = d ratio / T ratio, elasticity, bootstrap CI of index
    if "normal" in scen:
        rng = np.random.default_rng(seed)
        wide = {}
        for sc in scen:
            w = df[df["scenario"] == sc].pivot(index="loop_id", columns="segment", values="d")[SEGMENTS]
            T = df[df["scenario"] == sc].drop_duplicates("loop_id").set_index("loop_id")["T"].loc[w.index]
            wide[sc] = (w.to_numpy(), T.to_numpy())
        def meds(sc, idx=None):
            a, T = wide[sc]
            if idx is not None:
                a, T = a[idx], T[idx]
            return np.median(a, axis=0), np.median(T)
        base_d, base_T = meds("normal")
        ratios = []
        boots = {sc: [] for sc in scen if sc != "normal"}
        lo_q, hi_q = (1 - INDEX_CI) / 2 * 100, (1 + INDEX_CI) / 2 * 100
        for sc in scen:
            if sc == "normal":
                continue
            for _ in range(n_boot):
                i_n = rng.integers(0, len(wide["normal"][1]), len(wide["normal"][1]))
                i_s = rng.integers(0, len(wide[sc][1]), len(wide[sc][1]))
                dn, Tn = meds("normal", i_n)
                ds, Ts = meds(sc, i_s)
                boots[sc].append((ds / dn) / (Ts / Tn))
            boots[sc] = np.array(boots[sc])
            d_s, T_s = meds(sc)
            for j, seg in enumerate(SEGMENTS):
                lo, hi = np.percentile(boots[sc][:, j], [lo_q, hi_q])
                ratios.append({"segment": seg, "scenario": sc, "d_ratio": d_s[j] / base_d[j], "T_ratio": T_s / base_T,
                               "index": (d_s[j] / base_d[j]) / (T_s / base_T), "index_lo": lo, "index_hi": hi,
                               "disproportionate": bool(lo > 1 or hi < 1)})
        R["ratios"] = pd.DataFrame(ratios)
        # elasticity: log-log slope of session-median d on session-median T across the scenarios present (1 = proportional)
        el = {}
        for j, seg in enumerate(SEGMENTS):
            xs = np.log([meds(sc)[1] for sc in scen if sc in ("fast", "normal", "slow")])
            ys = np.log([meds(sc)[0][j] for sc in scen if sc in ("fast", "normal", "slow")])
            el[seg] = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 3 else np.nan
        R["elasticity"] = el

    # within-session elasticity: log d ~ C(session) + log T  (T contains d -> part-whole; interpret as descriptive)
    wi = {}
    for seg in SEGMENTS:
        x = df[(df["segment"] == seg) & (df["d"] > 0)].copy()
        x["logd"], x["logT"] = np.log(x["d"]), np.log(x["T"])
        fit = smf.ols("logd ~ C(session) + logT", data=x).fit()
        wi[seg] = (float(fit.params["logT"]), float(fit.bse["logT"]))
    R["within"] = wi
    return R


# ------------------------------------------------------------------ plotting
def make_boxplots(df, path):
    df = df.copy()
    df["scenario"] = df["session"].map(lambda s: session_info(s)["scenario"])
    df["dT"] = df["d"] / df["T"]
    scen = [s for s in SCEN_ORDER if s in set(df["scenario"])]
    fig, axes = plt.subplots(2, len(SEGMENTS), figsize=(3.0 * len(SEGMENTS), 7.0), sharey="row")
    for r, (col, lab) in enumerate([("d", "d  (s)"), ("dT", "d / T")]):
        for c, seg in enumerate(SEGMENTS):
            ax = axes[r, c]
            data = [df[(df["segment"] == seg) & (df["scenario"] == sc)][col].to_numpy() for sc in scen]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True, flierprops={"markersize": 2})
            for patch, sc in zip(bp["boxes"], scen):
                patch.set_facecolor(COLORS[sc]); patch.set_alpha(0.75)
            for m in bp["medians"]:
                m.set_color("black")
            ax.set_xticks(range(1, len(scen) + 1)); ax.set_xticklabels(scen, rotation=45, ha="right", fontsize=7)
            if r == 0:
                ax.set_title(seg, fontsize=10)
            if c == 0:
                ax.set_ylabel(lab)
            ax.grid(alpha=0.25)
    fig.suptitle("Q2: segment durations per session (raw d, top) and share of loop time d/T (bottom). "
                 "seq135/seq145 'b' = box 3/4; grey = secondary", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    S = sessions()
    by_scen = {session_info(s)["scenario"]: s for s in S}
    use = [by_scen[sc] for sc in SCEN_ORDER if sc in by_scen]
    prim = [by_scen[sc] for sc in PRIMARY]
    sec = [by_scen[sc] for sc in SECONDARY if sc in by_scen]

    emit("# Q2 — Kalış/boşluk süreleri tempo ile ölçekleniyor mu? (H2: d_{c,i} = tau_c * delta_i)")
    emit()
    emit("Kriterler (sonuçlara bakmadan önce sabitlendi): "
         f"interaction partial eta^2 < {ETA2_INTERACTION_MAX} VE her segmentte KW epsilon^2 (d/T) < {EPS2_SEGMENT_MAX} -> H2 geçer. "
         f"Orantısızlık bayrağı: (d oranı / T oranı) indeksinin %{int(INDEX_CI*100)} bootstrap GA'sı (loop düzeyinde, {N_BOOT} yeniden örnekleme, seed {SEED}) 1'i dışlıyorsa.")
    emit()

    # ---- build segments (ToF 30 Hz primary; frames 10 Hz cross-check)
    seg_tof, rep_tof, seg_fr = {}, {}, {}
    for s in use:
        seg_tof[s], rep_tof[s] = loop_selection(s, "tof")
        seg_fr[s] = loop_selection(s, "frames")[0]
    all_tof = pd.concat(seg_tof.values(), ignore_index=True)
    all_fr = pd.concat(seg_fr.values(), ignore_index=True)

    # ---- selection report
    sel_rows = []
    for s in use:
        r = rep_tof[s]
        sel_rows.append({"session": s, "n_loop_id": r["n_loops"], "loop0_drop": 1 if r["loop0"] else 0,
                         "last_loop_drop": f"{r['last_loop']}", "nonexact_drop": len(r["excluded_nonexact"]),
                         "nonexact_orders": "; ".join(f"{l}:{o or 'empty'}" for l, o in r["excluded_nonexact"]) or "-",
                         "no_next_box1": len(r["excluded_nofollow"]), "kept": r["n_kept"]})
    sel = pd.DataFrame(sel_rows)

    # ---- identity check: T == sum of six segments (assert inside loop_selection stores violations)
    piv = all_tof.pivot_table(index=["session", "loop_id"], columns="segment", values="d")[SEGMENTS]
    Tl = all_tof.drop_duplicates(["session", "loop_id"]).set_index(["session", "loop_id"])["T"].loc[piv.index]
    max_dev = float((piv.sum(axis=1) - Tl).abs().max())
    n_viol = sum(len(rep_tof[s].get("identity_viol", [])) for s in use)
    # cross-check vs frame-level loop duration (loops(): first->last frame of loop_id, ~ same span)
    xchk = []
    for s in use:
        lp = loops(s).set_index("loop_id")["duration_s"]
        t = seg_tof[s].drop_duplicates("loop_id").set_index("loop_id")["T"]
        j = pd.concat([t, lp.reindex(t.index)], axis=1, keys=["T", "frame_dur"]).dropna()
        xchk.append({"session": s, "med_T_tof": j["T"].median(), "med_loop_duration_frames": j["frame_dur"].median(),
                     "max_abs_diff_s": float((j["T"] - j["frame_dur"]).abs().max())})
    xchk = pd.DataFrame(xchk)
    n_zero = int((all_tof["d"] <= 0).sum())

    # ---- stats: primary (3 sessions, same sequence)
    SP = scaling_stats(all_tof, sessions=prim)
    SP_fr = scaling_stats(all_fr, sessions=prim, n_boot=200)
    # secondary: all five sessions, d/T KW per positional segment; plus pairwise vs normal
    SA = scaling_stats(all_tof, sessions=use, n_boot=200)

    an = SP["anova"]
    eta_int, eta_ses, eta_seg = an["interaction"]["eta2p"], an["session"]["eta2p"], an["segment"]["eta2p"]
    eps_dT = {k: v["eps2"] for k, v in SP["kw_dT"].items()}
    max_eps = max(eps_dT.values())
    fail_segs = [k for k, v in eps_dT.items() if v >= EPS2_SEGMENT_MAX]
    h2_pass = (eta_int < ETA2_INTERACTION_MAX) and (max_eps < EPS2_SEGMENT_MAX)

    # ================= markdown =================
    tb = SP["table"]
    emit("## 1. Segment başına tablo (birincil: Normal / Fast / Slow, ToF 30 Hz)")
    emit()
    emit("Medyan d (s) ve medyan d/T, oturum başına. n = kullanılan loop sayısı.")
    t1 = []
    for seg in SEGMENTS:
        r = {"segment": seg}
        for sc in ["normal", "fast", "slow"]:
            x = tb[(tb.segment == seg) & (tb.scenario == sc)].iloc[0]
            r[f"d {sc}"] = x["med_d"]
        for sc in ["normal", "fast", "slow"]:
            x = tb[(tb.segment == seg) & (tb.scenario == sc)].iloc[0]
            r[f"d/T {sc}"] = x["med_dT"]
        t1.append(r)
    t1 = pd.DataFrame(t1)
    emit(md_table(t1))
    emit()
    ns = {sc: int(tb[(tb.segment == "dwell_1") & (tb.scenario == sc)].iloc[0]["n"]) for sc in ["normal", "fast", "slow"]}
    medT = {sc: float(tb[(tb.segment == "dwell_1") & (tb.scenario == sc)].iloc[0]["med_T"]) for sc in ["normal", "fast", "slow"]}
    emit(f"n loop: Normal {ns['normal']}, Fast {ns['fast']}, Slow {ns['slow']}; medyan T: Normal {medT['normal']:.3f} s, Fast {medT['fast']:.3f} s, Slow {medT['slow']:.3f} s.")
    emit()
    emit("### Oransal ölçekleme: d oranı vs T oranı (indeks = d oranı / T oranı; 1 = tam orantılı; elasticity = log-log eğim, 3 oturum)")
    rt = SP["ratios"]
    t2 = []
    for seg in SEGMENTS:
        r = {"segment": seg}
        for sc, tag in [("fast", "Fast/Normal"), ("slow", "Slow/Normal")]:
            x = rt[(rt.segment == seg) & (rt.scenario == sc)].iloc[0]
            r[f"d oranı {tag}"] = x["d_ratio"]
            r[f"T oranı {tag}"] = x["T_ratio"]
            r[f"indeks {tag}"] = x["index"]
            r[f"GA {tag}"] = f"[{x['index_lo']:.2f}, {x['index_hi']:.2f}]"
            r[f"orantısız {tag}"] = "EVET" if x["disproportionate"] else "hayır"
        r["elasticity"] = SP["elasticity"][seg]
        t2.append(r)
    t2 = pd.DataFrame(t2)
    emit(md_table(t2, "{:.3f}"))
    emit()
    disp = rt[rt.disproportionate]
    disp_list = sorted({(r.segment, r.scenario) for r in disp.itertuples()})
    emit("Orantısız ölçeklenen (segment, oturum): " + (", ".join(f"{a}/{b}" for a, b in disp_list) if disp_list else "yok"))
    emit()

    emit("## 2. Karar sayıları")
    emit()
    emit("### Two-way ANOVA (Type II) — log d ~ session * segment (Normal/Fast/Slow)")
    a_rows = [{"effect": k, "df": an[k]["df"], "SS": an[k]["SS"], "F": an[k]["F"], "p": an[k]["p"], "partial_eta2": an[k]["eta2p"]}
              for k in ["session", "segment", "interaction"]]
    emit(md_table(pd.DataFrame(a_rows), "{:.4g}"))
    emit()
    emit(f"n gözlem = {an['n_obs']} (d <= 0 olduğu için log'dan çıkarılan: {an['n_zero_dropped']}); residual df = {an['resid']['df']:.0f}. "
         "Loop'lar oturum içinde otokorelasyonlu; p-değerleri iyimser, karar sayısı partial eta^2.")
    emit()
    emit(f"- **interaction partial eta^2 = {eta_int:.4f}** (eşik {ETA2_INTERACTION_MAX}); session = {eta_ses:.4f}; segment = {eta_seg:.4f}")
    emit()
    emit("Leave-one-segment-out: o segment çıkarıldığında interaction partial eta^2 (düşüş = o segment orantısızlığı sürüklüyor).")
    loo = pd.DataFrame([{"çıkarılan segment": k, "interaction_eta2p": v, "değişim": v - eta_int} for k, v in SP["loo_interaction"].items()])
    emit(md_table(loo, "{:.4f}"))
    emit()
    emit("### Kruskal-Wallis (Normal/Fast/Slow), segment başına")
    kw_rows = []
    for seg in SEGMENTS:
        a, b = SP["kw_dT"][seg], SP["kw_d"][seg]
        kw_rows.append({"segment": seg, "H(d/T)": a["H"], "p(d/T)": a["p"], "eps2(d/T)": a["eps2"],
                        "H(d)": b["H"], "p(d)": b["p"], "eps2(d)": b["eps2"],
                        "d/T eşik": "AŞTI" if a["eps2"] >= EPS2_SEGMENT_MAX else "ok"})
    emit(md_table(pd.DataFrame(kw_rows), "{:.4g}"))
    emit()
    emit("Not: d/T bir kompozisyondur (altı segmentin payı = 1); bir segment orantısızsa diğerleri telafi eder, "
         "yani eps^2 (d/T) bağımsız altı test değil. Raw d'deki büyük eps^2 beklenen tempo etkisidir (oturumlar arası T farkı).")
    emit()
    emit("### Oturum içi (within-session) elasticity: log d ~ C(session) + log T")
    emit("Eğim 1 ise o segment loop-to-loop tempo değişimini orantılı taşır. T, d'yi içerir (part-whole) → betimleyici; "
         "log-eğimlerin toplamı 1 olmak zorunda değil (kimlik d düzeyinde: sum d = T).")
    emit(md_table(pd.DataFrame([{"segment": k, "slope": v[0], "se": v[1]} for k, v in SP["within"].items()]), "{:.3f}"))
    emit()

    emit("## 3. Karar")
    emit()
    verdict = "**H2 GEÇTİ**" if h2_pass else "**H2 KALDI**"
    emit(f"{verdict}: interaction partial eta^2 = {eta_int:.4f} (eşik < {ETA2_INTERACTION_MAX}: {'ok' if eta_int < ETA2_INTERACTION_MAX else 'AŞTI'}); "
         f"en büyük segment eps^2(d/T) = {max_eps:.4f} (eşik < {EPS2_SEGMENT_MAX}: {'ok' if max_eps < EPS2_SEGMENT_MAX else 'AŞTI'}).")
    if fail_segs:
        emit("Eşiği aşan segmentler (d/T): " + ", ".join(f"{k} (eps^2={eps_dT[k]:.3f})" for k in fail_segs))
    emit()

    # interpretation (data-driven wording)
    emit("## 4. Yorum")
    emit()
    t_f, t_s = rt.iloc[0]["T_ratio"], rt[rt.scenario == "slow"].iloc[0]["T_ratio"]
    emit(f"- Tempo: T oranı Fast/Normal = {t_f:.3f}, Slow/Normal = {t_s:.3f}. Tam orantılı (çarpansal) ölçekleme altında her segmentin d oranı bu değere eşit olurdu (indeks 1).")
    ranked = rt.assign(dev=np.abs(np.log(rt["index"]))).sort_values("dev", ascending=False)
    top = ranked.head(4)
    emit("- Orantılıktan en uzak (|log indeks| en büyük) dört (segment, oturum): " +
         "; ".join(f"{r.segment}/{r.scenario} indeks {r.index:.2f} [{r.index_lo:.2f}, {r.index_hi:.2f}]" for r in top.itertuples()) + ".")
    over = [(r.segment, r.scenario, r.index) for r in rt.itertuples() if r.index > 1 and r.disproportionate]
    under = [(r.segment, r.scenario, r.index) for r in rt.itertuples() if r.index < 1 and r.disproportionate]
    if over:
        emit("- d/T payı büyüyen (indeks > 1; Fast'te orantılıdan az kısalan, Slow'da orantılıdan çok uzayan): " + ", ".join(f"{a}/{b} ({c:.2f})" for a, b, c in over) + ".")
    if under:
        emit("- d/T payı küçülen (indeks < 1; Fast'te orantılıdan çok kısalan, Slow'da orantılıdan az uzayan): " + ", ".join(f"{a}/{b} ({c:.2f})" for a, b, c in under) + ".")
    emit("- Elasticity (log-log, 3 oturum): " + ", ".join(f"{k} {v:.2f}" for k, v in SP["elasticity"].items()) +
         ". 1'in altı = segment tempo değişiminden daha az etkileniyor (sabit/zorunlu bileşen), 1'in üstü = tempo değişimini büyütüyor.")
    emit("- gap_b5 'havada birleştirme/taşıma' süresini içerir; ToF bunu Box_b çıkışı ile Box5 girişi arasından ayıramaz. gap_51 ise Box5'ten sonra elin dönüş + bekleme süresidir (loop sınırı Box1 girişidir).")
    emit("- H3 skoru için çıkarım (H2 olmasa bile): oransal zamanlama skoru d/T yerine segment-bazlı normalizasyon (örn. log d_{c,i} - elasticity_i * log T) gerektirir; bu, 'adım-bazlı normalize edilir' dalına (CLAUDE.md §5) karşılık gelir.")
    emit()

    # ---- cross-check frames
    emit("## 5. Çapraz kontrol: 10 Hz kare etiketlerinden aynı medyanlar")
    emit()
    ct = scaling_stats(all_fr, sessions=prim, n_boot=10)["table"]
    xr = []
    for seg in SEGMENTS:
        r = {"segment": seg}
        for sc in ["normal", "fast", "slow"]:
            a = tb[(tb.segment == seg) & (tb.scenario == sc)].iloc[0]["med_d"]
            b = ct[(ct.segment == seg) & (ct.scenario == sc)].iloc[0]["med_d"]
            r[f"ToF {sc}"] = a
            r[f"frames {sc}"] = b
        xr.append(r)
    emit(md_table(pd.DataFrame(xr)))
    emit()
    emit(f"10 Hz kareleriyle: interaction partial eta^2 = {SP_fr['anova']['interaction']['eta2p']:.4f}; "
         f"max segment eps^2(d/T) = {max(v['eps2'] for v in SP_fr['kw_dT'].values()):.4f} "
         f"(ToF: {eta_int:.4f} / {max_eps:.4f}). Kare etiketleri 100 ms'ye kuantize; kısa gap'lerde medyan sapması beklenir.")
    emit()

    # ---- secondary
    emit("## 6. İkincil: Seq135 / Seq145 (konumsal segmentler, d/T)")
    emit()
    emit("Ortak segment türleri dwell_1, dwell_5, gap_51; ortadakiler (gap_1b, dwell_b, gap_b5) kutu b = 2/3/4 olduğundan konumsal karşılaştırılır. "
         "Karşılaştırma yalnız d/T üzerinden (T farklı, kutu geometrisi farklı).")
    ta = SA["table"]
    t6 = []
    for seg in SEGMENTS:
        r = {"segment": seg}
        for sc in ["normal", "seq135", "seq145"]:
            r[f"d/T {sc}"] = ta[(ta.segment == seg) & (ta.scenario == sc)].iloc[0]["med_dT"]
        # pairwise KW normal vs each secondary
        dn = all_tof[(all_tof.session == by_scen["normal"]) & (all_tof.segment == seg)]
        for sc in ["seq135", "seq145"]:
            ds = all_tof[(all_tof.session == by_scen[sc]) & (all_tof.segment == seg)]
            kw = _kw([(dn["d"] / dn["T"]).to_numpy(), (ds["d"] / ds["T"]).to_numpy()])
            r[f"eps2 N vs {sc}"] = kw["eps2"]
        r["eps2 (5 oturum)"] = SA["kw_dT"][seg]["eps2"]
        t6.append(r)
    emit(md_table(pd.DataFrame(t6), "{:.3f}"))
    emit()
    emit(f"Seq135/145 oturumlarında loop sayıları: " + ", ".join(f"{sc}: {rep_tof[by_scen[sc]]['n_kept']}" for sc in SECONDARY) +
         f". T medyanları: " + ", ".join(f"{sc}: {ta[(ta.segment=='dwell_1')&(ta.scenario==sc)].iloc[0]['med_T']:.3f} s" for sc in ["normal", "seq135", "seq145"]) + ".")
    emit()

    # ---- selection + identity
    emit("## 7. Loop seçimi ve doğrulamalar")
    emit()
    emit("Yalnızca box_order beklenen diziye TAM eşit loop'lar; loop 0 (pre-start) ve oturumun SON loop'u (T ve gap_51 kesik) dışarıda.")
    emit(md_table(sel))
    emit()
    emit(f"Kimlik T = sum(6 segment): max |sum - T| = {max_dev:.3e} s; ihlal sayısı = {n_viol} (tolerans {IDENTITY_TOL}). "
         f"d <= 0 segment sayısı (tüm oturumlar) = {n_zero}.")
    emit()
    emit("T (ToF, Box1 girişi -> sonraki Box1 girişi) ile `loops()` kare-düzeyi loop süresi karşılaştırması:")
    emit(md_table(xchk))
    emit()
    emit("## 8. Şekil / dosyalar")
    emit("- `reports/probe/q2_boxplots.png`: üst satır ham d, alt satır d/T; segment x oturum (gri = ikincil).")
    emit("- `reports/probe/q2_table.csv`: segment x oturum özet (n, medyan, IQR, d/T).")
    emit("- Q3 için: `loop_segments(session)`, `scaling_stats(df, sessions=...)`.")

    # ---- files
    csv = SA["table"].copy()
    rr = SP["ratios"].rename(columns={"scenario": "scenario"})
    csv = csv.merge(rr[["segment", "scenario", "d_ratio", "T_ratio", "index", "index_lo", "index_hi", "disproportionate"]],
                    on=["segment", "scenario"], how="left")
    csv["kw_eps2_dT_primary"] = csv["segment"].map(eps_dT)
    csv.to_csv(REPORTS / "q2_table.csv", index=False)
    make_boxplots(all_tof[all_tof["session"].isin(use)], REPORTS / "q2_boxplots.png")
    (REPORTS / "q2.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

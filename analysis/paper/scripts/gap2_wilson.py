"""Gap 2 (OUTLINE s9.3): binomial (Wilson) 95% CIs for every false-alarm rate and detection rate the outline cites.

Blocks
  A  Faz A primary [P]: D3, B2 x pred, standard conformal, alpha = 0.10 (and 0.05), from reports/faz_a/04_cycle_scores.csv.gz (role == test, p3 <= alpha)
  B  A2.1 secondary [S]: D3, B2 x pred, cross + smoothed p, alpha = 0.10, from reports/faz_a2/01_conformal_table.csv
        (rates are means over 200 uniform draws -> expected counts rate * n, NOT integer alarm counts)
  C  A3.3 secondary [S]: within-cycle alarm counts from reports/faz_a3/03_online_summary.csv (+ check file, + alarm table for timing)

Counting units
  normal  : one test cycle appears in exactly one test fold -> pairs == cycles (n = 51); Wilson is exact-in-unit.
  anomaly : each anomaly cycle is scored in EVERY fold f (different calibration set each time) -> (cycle, f) pairs, n_pairs = 5 x n_cycles
            (checked: every anomaly cycle appears in all 5 folds). Pair-based Wilson treats the 5 repeats as independent. The repeats share the
            same cycle, so that CI is too NARROW (anti-conservative) w.r.t. sampling of cycles. We therefore also report
            (i) a cycle-level Wilson CI with n = distinct cycles and the same rate (effective count = rate * n_cycles; conservative: design effect = 5) and
            (ii) a cluster bootstrap over cycles (percentile, 10,000 resamples, stratified by anomaly type).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"
OUT = REP / "paper_prep"
Z = 1.959963984540054
RNG = np.random.RandomState(20261104)
B = 10000


def wilson(k, n, z=Z):
    """Wilson score interval; k may be fractional (expected count)."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def cp(k, n, a=0.05):
    if abs(k - round(k)) > 1e-9:
        return (np.nan, np.nan)
    k = int(round(k))
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


rows = []


def add(block, cell, group, alpha, k, n_pairs, n_cycles, unit, source, tag, boot=(np.nan, np.nan), note=""):
    p = k / n_pairs
    lo, hi = wilson(k, n_pairs)
    clo, chi = cp(k, n_pairs)
    if n_cycles is None or (isinstance(n_cycles, float) and np.isnan(n_cycles)):
        lo_c, hi_c = np.nan, np.nan                      # no cycle unit defined for this row
    elif n_cycles != n_pairs:
        lo_c, hi_c = wilson(p * n_cycles, n_cycles)
    else:
        lo_c, hi_c = lo, hi
    rows.append(dict(block=block, cell=cell, group=group, alpha=alpha, k=k, n_pairs=n_pairs, n_cycles=n_cycles, rate=p,
                     wilson_lo_pairs=lo, wilson_hi_pairs=hi, cp_lo_pairs=clo, cp_hi_pairs=chi,
                     wilson_lo_cycles=lo_c, wilson_hi_cycles=hi_c, boot_lo_cycles=boot[0], boot_hi_cycles=boot[1],
                     unit=unit, source=source, tag=tag, note=note))


# ====================================================================== A. Faz A [P]
cs = pd.read_csv(REP / "faz_a" / "04_cycle_scores.csv.gz")
t = cs[(cs.state_source == "b2") & (cs.segmentation == "pred") & (cs.role == "test")].copy()
t["key"] = t["session"] + ":" + t["cycle_idx"].astype(str)
GROUPS = {"seq135": ["seq135"], "seq145": ["seq145"], "fast": ["fast"], "slow": ["slow"],
          "seq(135+145)": ["seq135", "seq145"], "tempo(fast+slow)": ["fast", "slow"], "all anomalies": ["seq135", "seq145", "fast", "slow"]}
SRC_A = "reports/faz_a/04_cycle_scores.csv.gz (b2,pred,test; p3<=alpha); table reports/faz_a/04_anomaly_table.csv"
for alpha in (0.10, 0.05):
    n = t[t.atype == "normal"]
    add("A", "B2 x pred, D3 std", "normal (FAR)", alpha, int((n.p3 <= alpha).sum()), len(n), n["key"].nunique(),
        "test cycles (one fold each)", SRC_A, "[P]")
    for gname, types in GROUPS.items():
        d = t[t.atype.isin(types)]
        flag = (d.p3 <= alpha).astype(int)
        npair, ncyc = len(d), d["key"].nunique()
        # cluster bootstrap over cycles, stratified by atype
        per = d.assign(fl=flag).groupby(["atype", "key"])["fl"].agg(["sum", "count"]).reset_index()
        strata = [g[["sum", "count"]].to_numpy() for _, g in per.groupby("atype")]
        bs = np.empty(B)
        for b in range(B):
            s_ = c_ = 0
            for arr in strata:
                idx = RNG.randint(0, len(arr), len(arr))
                s_ += arr[idx, 0].sum(); c_ += arr[idx, 1].sum()
            bs[b] = s_ / c_
        add("A", "B2 x pred, D3 std", gname + " (detection)", alpha, int(flag.sum()), npair, ncyc,
            "(cycle, fold) pairs; each anomaly cycle is scored in all 5 folds", SRC_A, "[P]",
            boot=(float(np.quantile(bs, .025)), float(np.quantile(bs, .975))))

# ====================================================================== B. A2.1 cross+smoothed [S]
ct = pd.read_csv(REP / "faz_a2" / "01_conformal_table.csv")
cell = "B2 x pred (ana)"
ncyc_of = {g: t[t.atype.isin(ty)]["key"].nunique() for g, ty in {**GROUPS, "tum anomali": GROUPS["all anomalies"]}.items()}
SRC_B = "reports/faz_a2/01_conformal_table.csv (cell 'B2 x pred (ana)', variant D3, alpha 0.1)"
for method in ("cross_smooth", "cross", "smooth"):
    x = ct[(ct.cell == cell) & (ct.method == method) & (ct.variant == "D3") & (ct.alpha == 0.1) & (ct.metric.isin(["far", "det"]))]
    for r in x.itertuples():
        if r.metric == "far":
            add("B", f"B2 x pred, D3 {method}", "normal (FAR)", 0.10, r.value * r.n, r.n, 51, "test cycles; mean over draws (expected count)", SRC_B, "[S]",
                note="expected count = rate*n over 200 U-draws; not an integer" if "smooth" in method else "")
        else:
            g = "all anomalies" if r.atype == "tum anomali" else r.atype
            add("B", f"B2 x pred, D3 {method}", g + " (detection)", 0.10, r.value * r.n, r.n, ncyc_of[g],
                "(cycle, fold) pairs; mean over draws" if "smooth" in method else "(cycle, fold) pairs", SRC_B, "[S]",
                note="expected count = rate*n over 200 U-draws; not an integer" if "smooth" in method else "")

# ====================================================================== C. A3.3 [S]
on = pd.read_csv(REP / "faz_a3" / "03_online_summary.csv")
SRC_C = "reports/faz_a3/03_online_summary.csv"
for r in on.itertuples():
    cname = {"b2xpred": "B2 x pred (main)", "tofxtof": "ToF x tof (reference)"}.get(r.cell, r.cell)
    lab = "normal (FAR)" if r.atype == "normal" else r.atype + " (alarm rate)"
    add("C", cname, lab, 0.10, int(r.n_alarm), int(r.n_pairs), int(r.n_cycles), "(cycle, fold) pairs", SRC_C, "[S]")
    if r.atype != "normal" and r.n_alarm > 0:
        k1 = int(round(r.share_before_last_visit * r.n_alarm))
        add("C", cname, lab.replace("(alarm rate)", "share of alarms raised before the last visit began"), 0.10, k1, int(r.n_alarm), np.nan,
            "alarm pairs", SRC_C, "[S]", note="share_before_last_visit * n_alarm")
chk = pd.read_csv(REP / "faz_a3" / "03_online_check.csv")
for c, d in chk.groupby("cell"):
    n = int(d.n_test.sum()); k = int(d.n_diff.sum())
    add("C", {"b2xpred": "B2 x pred (main)", "tofxtof": "ToF x tof (reference)"}[c], "online vs offline alarm-set discordant pairs", 0.10, k, n, np.nan,
        "(cycle, fold) test pairs, 5 folds", "reports/faz_a3/03_online_check.csv", "[S]",
        note="0 discordant: Wilson upper bound = discordance rate that cannot be excluded; fold-level equality 5/5 per cell (10/10 total)")

# timing medians: cluster bootstrap over cycles
al = pd.read_csv(REP / "faz_a3" / "03_online_alarms.csv")
al = al[(al.state_source == "b2") & (al.segmentation == "pred") & (al.alarm_lit == True)].copy()  # noqa: E712
al["key"] = al["session"] + ":" + al["cycle_idx"].astype(str)
tim = []
for typ, d in al.groupby("atype"):
    keys = d["key"].unique()
    grp = {k: g for k, g in d.groupby("key")}
    def med(col, ks):
        v = np.concatenate([grp[k][col].dropna().to_numpy() for k in ks]) if len(ks) else np.array([])
        return float(np.median(v)) if len(v) else np.nan
    for col, nm in (("frac_lit", "alarm time / cycle duration"), ("sec_from_onset_lit", "seconds from onset (sequence anomalies)")):
        if d[col].notna().sum() == 0:
            continue
        m0 = med(col, keys)
        bs = [med(col, keys[RNG.randint(0, len(keys), len(keys))]) for _ in range(2000)]
        tim.append(dict(cell="B2 x pred (main)", atype=typ, quantity=nm, n_alarm_pairs=int(d[col].notna().sum()), n_cycles_with_alarm=len(keys),
                        median=m0, boot_lo=float(np.nanquantile(bs, .025)), boot_hi=float(np.nanquantile(bs, .975)),
                        source="reports/faz_a3/03_online_alarms.csv (alarm_lit, literal definition)", tag="[S]"))
TIM = pd.DataFrame(tim)
TIM.to_csv(OUT / "05_gaps_online_timing_medians.csv", index=False)

R = pd.DataFrame(rows)
R.to_csv(OUT / "05_gaps_wilson.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
show = R[["block", "cell", "group", "alpha", "k", "n_pairs", "n_cycles", "rate", "wilson_lo_pairs", "wilson_hi_pairs", "wilson_lo_cycles", "wilson_hi_cycles", "boot_lo_cycles", "boot_hi_cycles"]]
print(show.round(3).to_string())
print(TIM.round(3).to_string())

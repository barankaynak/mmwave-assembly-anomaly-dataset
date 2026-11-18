"""Phase E - 03: descriptive intervals for the one-shot E1 rates (no new score; reads reports/faz_e/02_oneshot_cycles.csv.gz only).

Same interval conventions as the Phase A numbers in the paper (paper/scripts/gap2_wilson.py):
  normal  : Wilson 95% over the 51 test cycles (one fold each);
  anomaly : rate over (cycle, fold) pairs; cluster bootstrap over cycles (percentile, 10,000 resamples, stratified by anomaly type,
            RandomState(20261104)); plus Clopper-Pearson over cycles when every pair of every cycle alarmed (rate 1, the bootstrap
            interval is then degenerate).
Writes reports/faz_e/03_intervals.csv and .md; prints every number. [S] (Phase E secondary, after the one-shot).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "faz_e"
Z = 1.959963984540054
B = 10000
GROUPS = {"seq135": ["seq135"], "seq145": ["seq145"], "fast": ["fast"], "slow": ["slow"],
          "seq(135+145)": ["seq135", "seq145"], "tempo(fast+slow)": ["fast", "slow"], "all anomalies": ["seq135", "seq145", "fast", "slow"]}


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


T = pd.read_csv(OUT / "02_oneshot_cycles.csv.gz")
T["key"] = T["session"] + ":" + T["cycle_idx"].astype(str)
rows = []
for rule, col in (("Phase A D3", "a_D3"), ("E1 D3'", "a_D3p")):
    RNG = np.random.RandomState(20261104)
    n = T[T["atype"] == "normal"]
    k = int(n[col].sum())
    lo, hi = wilson(k, len(n))
    rows.append(dict(rule=rule, group="normal (FAR)", k=k, n_pairs=len(n), n_cycles=n["key"].nunique(), rate=k / len(n),
                     lo=lo, hi=hi, interval="Wilson over test cycles"))
    for g, types in GROUPS.items():
        d = T[T["atype"].isin(types)]
        fl = d[col].astype(int)
        per = d.assign(fl=fl).groupby(["atype", "key"])["fl"].agg(["sum", "count"]).reset_index()
        strata = [gg[["sum", "count"]].to_numpy() for _, gg in per.groupby("atype")]
        bs = np.empty(B)
        for b in range(B):
            s_ = c_ = 0
            for arr in strata:
                idx = RNG.randint(0, len(arr), len(arr))
                s_ += arr[idx, 0].sum()
                c_ += arr[idx, 1].sum()
            bs[b] = s_ / c_
        rate = fl.sum() / len(d)
        r = dict(rule=rule, group=g, k=int(fl.sum()), n_pairs=len(d), n_cycles=d["key"].nunique(), rate=rate,
                 lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), interval="cluster bootstrap over cycles")
        full = int((per["sum"] == per["count"]).sum())
        r["cycles_alarmed_in_all_folds"] = full
        if full == len(per):                         # rate 1: report the exact Clopper-Pearson bound over cycles
            r["cp_cycles_lo"], r["cp_cycles_hi"] = cp(len(per), len(per))
        rows.append(r)
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
R.to_csv(OUT / "03_intervals.csv", index=False)
md = ["# Phase E - 03: intervals for the one-shot E1 rates (descriptive) [S]", "",
      "Source `reports/faz_e/02_oneshot_cycles.csv.gz`; conventions as `paper/scripts/gap2_wilson.py` (Wilson for the normal cycles; "
      "cluster bootstrap over cycles, 10,000 resamples, stratified by type, RandomState(20261104), for anomaly rates over (cycle, fold) "
      "pairs; Clopper-Pearson over cycles where every cycle alarmed in every fold).", "",
      "| rule | group | k | pairs | cycles | rate | 95% lo | 95% hi | interval | cycles alarmed in all folds | CP lo (cycles) |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in R.itertuples(index=False):
    md.append(f"| {r.rule} | {r.group} | {r.k} | {r.n_pairs} | {r.n_cycles} | {r.rate:.4f} | {r.lo:.4f} | {r.hi:.4f} | {r.interval} | "
              f"{'' if pd.isna(getattr(r, 'cycles_alarmed_in_all_folds', np.nan)) else int(r.cycles_alarmed_in_all_folds)} | "
              f"{'' if pd.isna(getattr(r, 'cp_cycles_lo', np.nan)) else f'{r.cp_cycles_lo:.4f}'} |")
(OUT / "03_intervals.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("wrote reports/faz_e/03_intervals.csv, 03_intervals.md")

# ---------------------------------------------------------------- table metrics (same definitions as the Phase A anomaly table)
# rows: Phase A D3 (re-derived from the one-shot file, must match tables/tab3_anomaly.tex), E1 S_E alone (p_E), E1 D3'.
# FAR over the 51 normal test cycles; detection over (cycle, fold) pairs; AUROC per fold (normal test cycles of the fold vs the
# group's pairs in that fold), mean over folds, s.d. (ddof 1) for all anomalies; score = 1 - p for p-based rules, S_E for E1 alone.
from sklearn.metrics import roc_auc_score  # noqa: E402

GRP = {"Sequence": ["seq135", "seq145"], "Tempo": ["fast", "slow"], "All": ["seq135", "seq145", "fast", "slow"]}
RULES = {"Phase A D3": ("D3_phaseA", "1-p"), "E1 S_E alone": ("p_E", "S_E"), "E1 D3'": ("D3prime", "1-p")}
trows = []
for rule, (pcol, scol) in RULES.items():
    sc = T["S_E"] if scol == "S_E" else 1 - T[pcol]
    r = dict(rule=rule, FAR_010=100 * (T.loc[T.atype == "normal", pcol] <= 0.10 + 1e-9).mean())
    for g, ty in GRP.items():
        d = T[T.atype.isin(ty)]
        r[f"det_{g}_010"] = 100 * (d[pcol] <= 0.10 + 1e-9).mean()
    r["det_All_005"] = 100 * (T.loc[T.atype.isin(GRP["All"]), pcol] <= 0.05 + 1e-9).mean()
    for g, ty in GRP.items():
        aucs = []
        for f in sorted(T.f.unique()):
            neg = sc[(T.atype == "normal") & (T.f == f)].to_numpy(float)
            pos = sc[(T.atype.isin(ty)) & (T.f == f)].to_numpy(float)
            y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
            aucs.append(roc_auc_score(y, np.r_[neg, pos]))
        r[f"auc_{g}"] = float(np.mean(aucs))
        if g == "All":
            r["auc_All_sd"] = float(np.std(aucs, ddof=1))
    trows.append(r)
TT = pd.DataFrame(trows)
print("\ntable metrics (percent / AUROC):")
print(TT.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
pa = TT.set_index("rule").loc["Phase A D3"]
assert abs(pa.FAR_010 - 7.843) < 0.01 and abs(pa.det_Sequence_010 - 35.577) < 0.01 and abs(pa.det_Tempo_010 - 92.525) < 0.01
assert abs(pa.det_All_010 - 63.350) < 0.01 and abs(pa.auc_Sequence - 0.897) < 0.0006 and abs(pa.auc_Tempo - 0.952) < 0.0006
assert abs(pa.auc_All - 0.924) < 0.0006 and abs(pa.auc_All_sd - 0.035) < 0.0006, "Phase A table row not reproduced"
print("Phase A D3 table row reproduced (tables/tab3_anomaly.tex)")
TT.to_csv(OUT / "03_table_metrics.csv", index=False)
with open(OUT / "03_intervals.md", "a", encoding="utf-8") as fh:
    fh.write("\n## Table metrics (definitions as the Phase A anomaly table; Phase A D3 row re-derived and asserted)\n\n")
    fh.write("| rule | FAR % (0.10) | seq % | tempo % | all % | all % (0.05) | AUROC seq | AUROC tempo | AUROC all (sd) |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in TT.itertuples(index=False):
        fh.write(f"| {r.rule} | {r.FAR_010:.1f} | {r.det_Sequence_010:.1f} | {r.det_Tempo_010:.1f} | {r.det_All_010:.1f} | {r.det_All_005:.1f} | "
                 f"{r.auc_Sequence:.3f} | {r.auc_Tempo:.3f} | {r.auc_All:.3f} ({r.auc_All_sd:.3f}) |\n")
print("wrote reports/faz_e/03_table_metrics.csv")

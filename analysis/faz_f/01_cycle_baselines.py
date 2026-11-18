"""Faz F / 01 - cycle-level baselines (plan: faz_f/DESIGN_F.md, sha256 recorded in the report; descriptive).

R0 exact-route rule; LEV (alone and + D2, Bonferroni as D3); isolation forest (alone, seeds 0-4).
Same cycles, roles and conformal rule as the primary analysis (B2 x pred). Reference rows D0/D1/D2/D3 are rebuilt
with faz_a2/a2_cycles and checked against reports/faz_a/04_cycle_scores.csv.gz before any baseline is reported.
s02 false alarms are reported pooled only.
Run: tools/local_run.sh faz_f/01_cycle_baselines.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
sys.path.insert(0, str(ROOT / "faz_a2"))
sys.path.insert(0, str(ROOT / "probe"))
import a2_cycles as ac  # noqa: E402
import fa_common as fc  # noqa: E402

OUT = ROOT / "reports" / "faz_f"
OUT.mkdir(parents=True, exist_ok=True)
PLAN = ROOT / "faz_f" / "DESIGN_F.md"
ROUTE = (1, 2, 5)
ALPHAS = (0.10, 0.05)
TYPES = ["seq135", "seq145", "fast", "slow"]


def lev(a, b):
    a, b = list(a), list(b)
    d = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(b) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return int(d[-1])


def feats(c):
    seq = list(c["seq"])
    return [seq.count(k) for k in range(1, 6)] + [len(seq), c["dur_s"], float(np.sum(c["dw"])), float(np.sum(c["gp"]))]


LAB = ac.source_labels(sources=("b2",), verbose=False)
C, _ = ac.build_cycles("b2", "pred", LAB["b2"])
cl = C.to_dict("records")
rows = []
for f in range(fc.N_FOLDS):
    ro = fc.anom_roles(f)
    tr, ca, te = ac._split_roles(cl, ro)
    par = ac.fit_params(tr)
    sc = {}
    for role, lst in (("cal", ca), ("test", te)):
        d = ac.score_cycles(lst, par)
        d["role"] = role
        d["lev"] = [lev(c["seq"], ROUTE) for c in lst]
        d["r0"] = [tuple(c["seq"]) != ROUTE for c in lst]
        d["natural_dev"] = [c["natural_dev"] for c in lst]
        sc[role] = d
    X_tr = np.array([feats(c) for c in tr])
    for seed in range(5):
        m = IsolationForest(random_state=seed).fit(X_tr)
        for role, lst in (("cal", ca), ("test", te)):
            sc[role][f"if{seed}"] = -m.score_samples(np.array([feats(c) for c in lst]))
    cal, t = sc["cal"], sc["test"].copy()
    for v in ["D0", "D1", "D2", "lev"] + [f"if{s}" for s in range(5)]:
        t["p_" + v] = ac.conf_p(cal[v].to_numpy(), t[v].to_numpy())
    t["p_D3"] = np.minimum(1, 2 * np.minimum(t["p_D1"], t["p_D2"]))
    t["p_levD2"] = np.minimum(1, 2 * np.minimum(t["p_lev"], t["p_D2"]))
    t["f"] = f
    t["n_cal"] = len(ca)
    t["cal_lev1"] = int((cal["lev"] >= 1).sum())
    rows.append(t)
T = pd.concat(rows, ignore_index=True)

# reproduction check against the primary scores
P = pd.read_csv(ROOT / "reports" / "faz_a" / "04_cycle_scores.csv.gz")
P = P[(P.state_source == "b2") & (P.segmentation == "pred") & (P.role == "test")]
k = ["session", "cycle_idx", "f"]
J = T.merge(P[k + ["D0", "D1", "D2", "p3"]], on=k, suffixes=("", "_ref"))
assert len(J) == len(T) == len(P), (len(J), len(T), len(P))
for v in ["D0", "D1", "D2"]:
    assert np.allclose(J[v], J[v + "_ref"], atol=1e-9), v
assert np.allclose(J["p_D3"], J["p3"], atol=1e-9)

VARS = {"D0 alone": ("p_D0", "D0"), "D1 alone": ("p_D1", "D1"), "D2 alone": ("p_D2", "D2"),
        "D3 (primary)": ("p_D3", None), "LEV alone": ("p_lev", "lev"), "LEV + D2": ("p_levD2", None)}
VARS.update({f"IF seed {s}": (f"p_if{s}", f"if{s}") for s in range(5)})
out = []
for name, (pc, sc_) in VARS.items():
    r = dict(detector=name)
    nrm = T[T.atype == "normal"]
    for a in ALPHAS:
        r[f"FAR_{a:.2f}"] = float((nrm[pc] <= a + 1e-9).mean())
        for ty in TYPES + ["seq", "tempo", "all"]:
            mem = {"seq": TYPES[:2], "tempo": TYPES[2:], "all": TYPES}.get(ty, [ty])
            g = T[T.atype.isin(mem)]
            r[f"det_{ty}_{a:.2f}"] = float((g[pc] <= a + 1e-9).mean())
    for grp, mem in (("seq", TYPES[:2]), ("tempo", TYPES[2:]), ("all", TYPES)):
        au = []
        for f, gf in T.groupby("f"):
            neg = gf[gf.atype == "normal"]
            pos = gf[gf.atype.isin(mem)]
            s_n = neg[sc_] if sc_ else -neg[pc]
            s_p = pos[sc_] if sc_ else -pos[pc]
            au.append(ac.auc_ap(s_n, s_p)[0])
        r[f"auroc_{grp}"] = float(np.nanmean(au))
    out.append(r)
nrm = T[T.atype == "normal"]
r = {"detector": "R0 exact route (no calibration)", "FAR_0.10": float(nrm["r0"].mean())}
for ty in TYPES + ["seq", "tempo", "all"]:
    mem = {"seq": TYPES[:2], "tempo": TYPES[2:], "all": TYPES}.get(ty, [ty])
    r[f"det_{ty}_0.10"] = float(T[T.atype.isin(mem)]["r0"].mean())
out.append(r)
R = pd.DataFrame(out)
R.to_csv(OUT / "01_cycle_baselines.csv", index=False)

pf = T.groupby("f").agg(n_cal=("n_cal", "first"), cal_lev_ge1=("cal_lev1", "first"))
pf["det_seq_LEValone"] = T[T.atype.isin(TYPES[:2])].groupby("f").apply(lambda g: (g.p_lev <= 0.10 + 1e-9).mean())
pf["det_seq_LEVD2"] = T[T.atype.isin(TYPES[:2])].groupby("f").apply(lambda g: (g.p_levD2 <= 0.10 + 1e-9).mean())
lev_dist = T[T.atype != "normal"].groupby("atype")["lev"].value_counts().unstack(fill_value=0)

ifs = R[R.detector.str.startswith("IF")]
cols = ["detector", "FAR_0.10", "det_seq135_0.10", "det_seq145_0.10", "det_fast_0.10", "det_slow_0.10",
        "det_seq_0.10", "det_tempo_0.10", "det_all_0.10", "auroc_seq", "auroc_tempo", "auroc_all"]
L = ["# Faz F / 01 Cycle-level baselines (descriptive; 2026-11-10)", "",
     f"Plan sha256: {hashlib.sha256(PLAN.read_bytes()).hexdigest()}", "",
     "Reproduction: D0, D1, D2 and p3 equal reports/faz_a/04_cycle_scores.csv.gz (B2 x pred, all test rows).", "",
     "## Results at alpha = 0.10 (FAR pooled over the 51 normal test cycles; detection over (cycle, fold) pairs)", "",
     "```", R[[c for c in cols if c in R]].to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```", "",
     "## Isolation forest over seeds 0-4 (mean, min-max)", ""]
for c in cols[1:]:
    L.append(f"- {c}: mean {ifs[c].mean():.3f}, range {ifs[c].min():.3f}-{ifs[c].max():.3f}")
L += ["", "## Per fold (LEV)", "", "```", pf.to_string(float_format=lambda v: f"{v:.3f}"), "```", "",
      "## LEV distance of anomaly test pairs", "", "```", lev_dist.to_string(), "```", "",
      "## alpha = 0.05", "", "```",
      R[["detector"] + [c for c in R if c.endswith("0.05")]].to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```"]
(OUT / "01_cycle_baselines.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

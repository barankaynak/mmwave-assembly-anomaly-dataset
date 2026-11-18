"""Faz A4 / 05 - descriptive re-reading for the session-confound section (2026-11-10).

No model is trained and no score is recomputed. Reads reports/faz_a/04_cycle_scores.csv.gz (primary scores,
B2 x pred) and reports, for the sequence-deviation sessions s03/s04:
  - how many duration steps D2 could score (d2_nref) against normal cycles;
  - the alarm rate of the whole-cycle duration score D0 alone at alpha = 0.10 (p0), with a cycle-clustered
    bootstrap interval, against the D0 rate on the 51 normal test cycles (Wilson interval);
  - the share of D0-alarmed sequence pairs that were shorter than the fold's mean training cycle (log scale).
Label in the paper: descriptive (planned after all other results were known).
Output: reports/faz_a4/05_d0_sequence.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "faz_a4" / "05_d0_sequence.md"
d = pd.read_csv(ROOT / "reports" / "faz_a" / "04_cycle_scores.csv.gz")
d = d[(d.state_source == "b2") & (d.segmentation == "pred")]
t = d[d.role == "test"].copy()
t["a0"] = t.p0 <= 0.10 + 1e-9
L = ["# 05 D0 and D2 coverage on the sequence sessions (descriptive; 2026-11-10)", "",
     "Source: reports/faz_a/04_cycle_scores.csv.gz, B2 x pred, test rows, alpha = 0.10.", ""]

L.append("## Steps scored by D2 (d2_nref), per (cycle, fold) pair")
for a in ["normal", "seq135", "seq145", "fast", "slow"]:
    x = t[t.atype == a]
    L.append(f"- {a}: {len(x)} pairs; d2_nref counts {x.d2_nref.value_counts().sort_index().to_dict()}")
L.append("")

nrm = t[t.atype == "normal"]
k, n = int(nrm.a0.sum()), len(nrm)
lo, hi = proportion_confint(k, n, method="wilson")
L.append(f"## D0 alone at alpha = 0.10\n- normal: {k}/{n} = {k/n:.3f} (Wilson {lo:.3f}-{hi:.3f})")

s = t[t.atype.isin(["seq135", "seq145"])].copy()
for a in ["seq135", "seq145"]:
    x = s[s.atype == a]
    L.append(f"- {a}: {int(x.a0.sum())}/{len(x)} = {x.a0.mean():.3f}")
g = s.groupby(["session", "cycle_idx"]).a0.agg(["sum", "count"]).to_numpy()
rng = np.random.default_rng(20261110)
r = []
for _ in range(4000):
    b = g[rng.integers(0, len(g), len(g))].sum(0)
    r.append(b[0] / b[1])
q = np.percentile(r, [2.5, 97.5])
L.append(f"- sequence pooled: {int(s.a0.sum())}/{len(s)} = {s.a0.mean():.3f} "
         f"(cycle-clustered bootstrap over {len(g)} cycles, 4000 draws: {q[0]:.3f}-{q[1]:.3f})")

mu = d[(d.role == "train") & (d.atype == "normal")].groupby("f").dur_s.apply(lambda x: np.log(x).mean())
s["short"] = np.log(s.dur_s) < s.f.map(mu)
al = s[s.a0]
L.append(f"- D0-alarmed sequence pairs shorter than the fold's mean training cycle: "
         f"{int(al.short.sum())}/{len(al)} = {al.short.mean():.3f}")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

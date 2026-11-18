"""Faz A2 / 06 - descriptive re-reading of the 50 commissioning resamples (2026-11-10).

No model is trained and no score is recomputed. Reads reports/faz_a2/04_comm_anomaly_far.csv and
04_comm_anomaly_groups.csv (B2 x pred, n = 20 training cycles, 20 calibration ToF cycles, 11 test ToF cycles,
50 random assignments of the 51 s02 normal cycles) and reports:
  - the distribution of D3 detection per anomaly type at alpha = 0.10 (mean, s.d., 10/50/90 % quantiles, draws with 0);
  - the number of predicted calibration cycles (n_cal) per draw, and detection split by n_cal >= 19 vs <= 18;
  - the false-alarm rate per draw (mean, 90 % quantile), all draws and n_cal >= 19.
Reason: D3 = min(1, 2 min(p1, p2)) has smallest value 2/(n_cal + 1), so it can reach alpha = 0.10 only if n_cal >= 19.
Label in the paper: descriptive (planned after the primary results and the external critique were known).
Output: reports/faz_a2/06_resample_summary.md (+ .csv).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports" / "faz_a2"
f = pd.read_csv(R / "04_comm_anomaly_far.csv")
g = pd.read_csv(R / "04_comm_anomaly_groups.csv")
f = f[(f.src == "b2") & (f.seg == "pred") & (f.n == 20)].set_index("rep")
g = g[(g.src == "b2") & (g.seg == "pred") & (g.n == 20)]
assert len(f) == 50
ok = f.n_cal_cycles >= 19

# arithmetic check of the condition
for n in range(15, 25):
    assert (2 / (n + 1) <= 0.10 + 1e-9) == (n >= 19)

rows = []
for grp, d in g.groupby("group"):
    x = d.set_index("rep").det_a10.reindex(f.index)
    for sub, m in (("all", slice(None)), ("ncal>=19", ok), ("ncal<=18", ~ok)):
        y = x[m]
        rows.append(dict(group=grp, subset=sub, draws=len(y), mean=y.mean(), sd=y.std(),
                         q10=y.quantile(.1), q50=y.quantile(.5), q90=y.quantile(.9),
                         zero=int((y == 0).sum()), min=y.min(), max=y.max()))
for sub, m in (("all", slice(None)), ("ncal>=19", ok), ("ncal<=18", ~ok)):
    y = f.far_a10[m]
    rows.append(dict(group="FAR (normal test cycles)", subset=sub, draws=len(y), mean=y.mean(), sd=y.std(),
                     q10=y.quantile(.1), q50=y.quantile(.5), q90=y.quantile(.9),
                     zero=int((y == 0).sum()), min=y.min(), max=y.max()))
T = pd.DataFrame(rows)
T.to_csv(R / "06_resample_summary.csv", index=False)

lines = ["# 06 Resample summary (descriptive; 2026-11-10)", "",
         "Source: 04_comm_anomaly_far.csv / 04_comm_anomaly_groups.csv, B2 x pred, n = 20 training cycles, 50 draws.", "",
         f"Predicted calibration cycles per draw: min {f.n_cal_cycles.min()}, median {f.n_cal_cycles.median():.0f}, "
         f"max {f.n_cal_cycles.max()}; draws with n_cal <= 18: {int((~ok).sum())} of 50.", "",
         "D3 can alarm at alpha = 0.10 only if n_cal >= 19 (smallest D3 = 2/(n_cal+1)).", "",
         "```", T.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```"]
(R / "06_resample_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

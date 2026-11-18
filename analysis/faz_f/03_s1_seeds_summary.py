"""Faz F / 03 - summary of the S1 seed runs (plan: faz_f/DESIGN_F.md part F.2 and its amendment; secondary).

Inputs: reports/faz_a/cnn_runs.csv (primary CNN-GRU S1 folds, replica "p"), reports/faz_f/02_cnn_s1_runs.csv
(replica 0 fold 0 = same-seed rerun; replicas 1-4 all folds), reports/faz_a/02_baseline_folds.csv (B2 S1 per fold),
reports/faz_d/02_leakage_per_run.csv (CNN-GRU S0 five seeds, accuracy).
Output: reports/faz_f/03_s1_seeds.md (+ .csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
SEED_S1 = 20261104

prim = pd.read_csv(R / "faz_a" / "cnn_runs.csv")
prim = prim[(prim.model == "cnngru") & (prim.split == "S1")][["fold", "test_acc", "test_macro_f1"]].assign(rep="p")
new = pd.read_csv(R / "faz_f" / "02_cnn_s1_runs.csv")
new["rep"] = ((new["seed"] - (SEED_S1 + 1000 + new["fold"])) // 100).astype(int).astype(str)
rerun = new[new.rep == "0"]
new = new[new.rep != "0"][["fold", "test_acc", "test_macro_f1", "rep"]]
A = pd.concat([prim, new], ignore_index=True)
b2 = pd.read_csv(R / "faz_a" / "02_baseline_folds.csv")
b2 = b2[b2.split == "S1"].set_index("fold")
A["b2_mf1"] = A.fold.map(b2.b2_mf1)
A["b2_acc"] = A.fold.map(b2.b2_acc)
A["gap_mf1"] = A.b2_mf1 - A.test_macro_f1
A.to_csv(R / "faz_f" / "03_s1_seeds.csv", index=False)

cells = len(A)
lead = int((A.gap_mf1 > 0).sum())
per_fold = A.groupby("fold").agg(n=("rep", "size"), cnn_mean=("test_macro_f1", "mean"), cnn_min=("test_macro_f1", "min"),
                                 cnn_max=("test_macro_f1", "max"), b2=("b2_mf1", "first"))
per_fold["gap_mean"] = per_fold.b2 - per_fold.cnn_mean
fold_means = A.groupby("fold").test_macro_f1.mean()
gap_by_fold = b2.b2_mf1 - fold_means
t = stats.t.ppf(0.975, len(gap_by_fold) - 1)
ci = (gap_by_fold.mean() - t * gap_by_fold.std(ddof=1) / np.sqrt(len(gap_by_fold)),
      gap_by_fold.mean() + t * gap_by_fold.std(ddof=1) / np.sqrt(len(gap_by_fold)))
# seed-to-seed s.d. within fold
sd_within = A.groupby("fold").test_macro_f1.std(ddof=1)

# leakage: S0 five seeds (accuracy, %) vs S1 five-seed fold means
lk = pd.read_csv(R / "faz_d" / "02_leakage_per_run.csv")
s0 = lk[(lk.model == "cnngru") & (lk.split == "S0")]["acc"].to_numpy() * 100
s1 = A.groupby("fold").test_acc.mean().to_numpy() * 100
w = stats.ttest_ind(s0, s1, equal_var=False)
d = s0.mean() - s1.mean()
se = np.sqrt(s0.var(ddof=1) / len(s0) + s1.var(ddof=1) / len(s1))
dfw = se**4 / ((s0.var(ddof=1) / len(s0))**2 / (len(s0) - 1) + (s1.var(ddof=1) / len(s1))**2 / (len(s1) - 1))
tw = stats.t.ppf(0.975, dfw)

L = ["# Faz F / 03 CNN-GRU under S1, five seeds per fold (secondary; 2026-11-10)", "",
     f"Cells (fold x seed): {cells}; replicas per fold: {A.groupby('fold').size().to_dict()}", "",
     f"Same-seed rerun of fold 0 (replica 0): macro-F1 {rerun.test_macro_f1.iloc[0]:.4f} vs primary "
     f"{prim[prim.fold == 0].test_macro_f1.iloc[0]:.4f}" if len(rerun) else "no rerun", "",
     f"CNN-GRU S1 macro-F1, mean over all cells {A.test_macro_f1.mean():.4f} (min {A.test_macro_f1.min():.4f}, max {A.test_macro_f1.max():.4f}); "
     f"primary fold mean {prim.test_macro_f1.mean():.4f}; B2 fold mean {b2.b2_mf1.mean():.4f}", "",
     f"B2 leads in {lead} of {cells} fold x seed cells", "",
     f"B2 minus CNN-GRU (five-seed fold means), paired by fold: mean {gap_by_fold.mean():+.4f}, 95% t interval "
     f"[{ci[0]:+.4f}, {ci[1]:+.4f}] (5 folds)", "",
     f"Within-fold seed s.d. of CNN-GRU macro-F1: mean {sd_within.mean():.4f} (per fold {sd_within.round(4).to_dict()})", "",
     "```", per_fold.to_string(float_format=lambda v: f"{v:.4f}"), "```", "",
     f"Leakage (secondary): S0 five seeds {s0.mean():.2f} % vs S1 five-seed fold means {s1.mean():.2f} % accuracy; "
     f"difference {d:+.2f} points, Welch 95% interval [{d - tw * se:+.2f}, {d + tw * se:+.2f}] (df {dfw:.1f}, p {w.pvalue:.2g})", ""]
(R / "faz_f" / "03_s1_seeds.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

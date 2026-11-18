"""Descriptive: per-fold calibration counts (n_cal) under predicted segmentation for the
state sources of the primary table and of the unseen-session table. Reads existing OOF files
only; scores nothing new, opens no alarm of any cycle. Writes reports/faz_a4/07_ncal_counts.csv/.md.
Usage: tools/local_run.sh faz_a4/07_ncal_counts.py"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in ("faz_a", "faz_a2", "probe"):
    sys.path.insert(0, str(ROOT / p))
import pandas as pd
import a2_cycles as ac
import fa_common as fc
from common import md_table

OUT = ROOT / "reports" / "faz_a4"
oof_i = fc.load_oof("b2", "S2")
oof_ii = pd.read_csv(OUT / "oof" / "b2_s0102__all.csv.gz")
LABELS = {"i_S2": ac.labels_from_oof(oof_i, strict=False),
          "ii_s01s02": ac.labels_from_oof(oof_ii, strict=False)}
_, _, NCAL = ac.score_all(sources=list(LABELS), segs=("pred",), labels=LABELS, verbose=False)
NCAL.to_csv(OUT / "07_ncal_counts.csv", index=False)
print(NCAL.to_string(index=False))
(OUT / "07_ncal_counts.md").write_text(
    "# Per-fold calibration counts, predicted segmentation (descriptive)\n\n"
    "Source: existing OOF predictions (b2__S2, b2_s0102__all); a2_cycles.score_all NCAL output.\n"
    "D3 can alarm at alpha = 0.10 only if n_cal >= 19.\n\n" + md_table(NCAL) + "\n",
    encoding="utf-8")

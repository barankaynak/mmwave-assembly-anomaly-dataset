"""Insert (or refresh) the Phase E block of the anomaly table, tables/tab3_anomaly.tex, from reports/faz_e/03_table_metrics.csv.

Idempotent: the block sits between the marker comments '% BEGIN PHASE E' and '% END PHASE E', just before \\bottomrule.
Sources: reports/faz_e/03_table_metrics.csv (written by faz_e/03_e1_intervals.py from the one-shot file
reports/faz_e/02_oneshot_cycles.csv.gz; the Phase A D3 row of that file is asserted to reproduce the published table row). [S]
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "paper" / "tables" / "tab3_anomaly.tex"
M = pd.read_csv(ROOT / "reports" / "faz_e" / "03_table_metrics.csv").set_index("rule")
SRC = "[S] faz_e/03_table_metrics.csv (one-shot faz_e/02_oneshot_cycles.csv.gz)"


def row(label, r):
    cells = [f"{r.FAR_010:.1f}", f"{r.det_Sequence_010:.1f}", f"{r.det_Tempo_010:.1f}", f"{r.det_All_010:.1f}", f"{r.det_All_005:.1f}",
             f"{r.auc_Sequence:.3f}", f"{r.auc_Tempo:.3f}", f"{r.auc_All:.3f} $\\pm$ {r.auc_All_sd:.3f}"]
    return f"{label} & " + " & ".join(cells) + f" \\\\ % {SRC}"


block = ["% BEGIN PHASE E", r"\midrule",
         r"\multicolumn{9}{@{}l}{\emph{Evidence-weighted sequence score E1 for B2 (secondary; designed after the primary results were known, run once)}} \\",
         row("E1 sequence score alone ($p_E$)", M.loc["E1 S_E alone"]),
         row("E1 + D2 (Bonferroni, as D3)", M.loc["E1 D3'"]),
         "% END PHASE E"]
t = TAB.read_text(encoding="utf-8")
if "% BEGIN PHASE E" in t:
    a, rest = t.split("% BEGIN PHASE E", 1)
    _, b = rest.split("% END PHASE E", 1)
    t = a.rstrip("\n") + "\n" + "\n".join(block) + b
else:
    assert t.count("\\bottomrule") == 1
    t = t.replace("\\bottomrule", "\n".join(block) + "\n\\bottomrule")
TAB.write_text(t, encoding="utf-8")
print("\n".join(block))

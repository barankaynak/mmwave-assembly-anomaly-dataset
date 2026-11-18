"""Tables for the A4 revision, generated from the verified Phase A4 result files.

Reads only ROOT/reports/faz_a4/*.csv (read-only) and writes LaTeX fragments to paper/tables/.
Every row carries a source-tag comment ([S] = secondary, A4 plan faz_a4/DESIGN_A4.md).
Usage (Git Bash, from ROOT):  "$PY" paper/scripts/make_tables_a4.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A4 = ROOT / "reports" / "faz_a4"
OUT = ROOT / "paper" / "tables"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")
    print("wrote", OUT / name)


# ---------------------------------------------------------------- main text: matched-hardware leakage and Keras
def tab_leak():
    # Phase D (faz_d/DESIGN_D.md): every CNN-GRU run on the RTX 3090. PyTorch S0 = seed 42 (Phase A) + seeds 1-4 (Phase D);
    # the GTX 1650 SUPER runs of A2.2/A4.3a are no longer used. Keras and B2 rows unchanged (A4 per-run file).
    p = pd.read_csv(A4 / "03_leakage_matched_keras_per_run.csv")
    d = pd.read_csv(ROOT / "reports" / "faz_d" / "02_leakage_per_run.csv")
    dd = d[d.model == "cnngru"].copy()
    dd["model"] = "cnngru_3090"
    s1_chk = dd[dd.split == "S1"].sort_values("run")["acc"].to_numpy()
    s1_ref = p[(p.model == "cnngru_remote") & (p.split == "S1")].sort_values("run")["acc"].to_numpy()
    assert len(s1_chk) == 5 and abs(s1_chk - s1_ref).max() < 1e-12, "S1 folds of faz_d and A4 differ"
    db = d[d.model == "b2"].copy()   # B2: S0 seeds 42, 1-4 and S1 folds from the Phase D per-run file (2026-11-18)
    p = pd.concat([p[p.model.isin(["keras"])], dd[["model", "split", "run", "n", "acc", "macro_f1"]],
                   db[["model", "split", "run", "n", "acc", "macro_f1"]]], ignore_index=True)
    g = p.groupby(["model", "split"]).agg(n=("run", "size"), acc=("acc", "mean"), acc_sd=("acc", "std"),
                                         f1=("macro_f1", "mean"), f1_sd=("macro_f1", "std"))
    rows = [("cnngru_3090", "CNN--GRU, PyTorch port (RTX~3090)", "seeds 42, 1--4"),
            ("keras", "CNN--GRU, initial Keras architecture (RTX~3090)", "seed 42"),
            ("b2", "B2, 21 features + random forest (CPU)", "seeds 42, 1--4")]
    src = {"cnngru_3090": "[S] faz_d/02_leakage_per_run.csv (cnngru; S0 seeds 1-4 = Phase D, RTX 3090; S0 seed 42 and S1 = Phase A runs [P])",
           "keras": "[S] faz_a4/03_leakage_matched_keras_per_run.csv (keras; A4.3b, RTX 3090)",
           "b2": "[S] faz_d/02_leakage_per_run.csv (b2; S0 seeds 42, 1-4; S1 = Phase A folds [P])"}
    lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{S0, random window split} & \multicolumn{2}{c}{S1, cycle-grouped (5 folds)} & S0 $-$ S1 \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}",
             r"Model (hardware) & S0 runs & Accuracy (\%) & Macro-F1 & Accuracy (\%) & Macro-F1 & Accuracy (points) \\",
             r"\midrule"]
    for key, lab, seeds in rows:
        s0, s1 = g.loc[(key, "S0")], g.loc[(key, "S1")]
        a0 = pct(s0.acc, 1) + (f" $\\pm$ {pct(s0.acc_sd, 1)}" if s0.n > 1 else "")
        f0 = f"{s0.f1:.3f}" + (f" $\\pm$ {s0.f1_sd:.3f}" if s0.n > 1 else "")
        a1 = f"{pct(s1.acc, 1)} $\\pm$ {pct(s1.acc_sd, 1)}"
        f1 = f"{s1.f1:.3f} $\\pm$ {s1.f1_sd:.3f}"
        d = 100 * (s0.acc - s1.acc)
        dcell = f"{d:+.1f}"
        lines.append(f"{lab} & {seeds} & {a0} & {f0} & {a1} & {f1} & {dcell} \\\\ % {src[key]}")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("tab_leak_matched.tex", "\n".join(lines))


# ---------------------------------------------------------------- main text: unseen-session estimators beside Table 3
def tab_unseen():
    d = pd.read_csv(A4 / "04_unseen_anomaly_table.csv")
    far = pd.read_csv(A4 / "04_unseen_far.csv")
    d = d[(d.segmentation == "pred") & (d.variant == "D3")]
    srcs = [("FazA_S1", "Faz A (S1 OOF)", "Pooled S1 (primary; all six sessions in training)"),
            ("i_S2", "(i) S2 OOF", "(i) Leave one session out (other anomaly sessions seen)"),
            ("ii_s01s02", "(ii) s01+s02", "(ii) s01 + s02 only (no anomaly session in training)")]
    lines = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Normal cycles} & \multicolumn{5}{c}{Detection rate (\%)} & AUROC \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-8}\cmidrule(lr){9-9}",
             r"State estimator trained on & Alarms & FAR (\%) [95\% CI] & 1--3--5 & 1--4--5 & Fast & Slow & All & All \\",
             r"\midrule"]
    for key, fname, lab in srcs:
        r = d[d.state_source == key].set_index("atype")
        f = far[(far.seg == "pred") & (far.source == fname) & (far.variant == "D3") & (far.alpha == 0.1)].iloc[0]
        det = [pct(r.loc[a, "det_a10"]) for a in ["seq135", "seq145", "fast", "slow", "tum anomali"]]
        au = r.loc["tum anomali"]
        lines.append(f"{lab} & {int(f.n_alarm)}/{int(f.n_normal)} & {pct(f.FAR)} [{pct(f.wilson_lo)}--{pct(f.wilson_hi)}] & "
                     + " & ".join(det) + f" & {au.auroc_mean:.3f} $\\pm$ {au.auroc_sd:.3f} \\\\ "
                     f"% [S] faz_a4/04_unseen_anomaly_table.csv, 04_unseen_far.csv ({key}, pred, D3, alpha 0.10)")
    # Phase G (faz_g/DESIGN_G.md, 2026-11-18): E1 + D2 with estimator (ii), one planned run
    g = pd.read_csv(ROOT / "reports" / "faz_g" / "01_e1_ii_summary.csv").set_index("atype")
    nn = int(g.loc["normal", "n_pairs"]); kk = int(round(g.loc["normal", "E1_D3p"] * nn))
    det = [pct(g.loc[a, "E1_D3p"]) for a in ["seq135", "seq145", "fast", "slow", "all"]]
    lines.append(f"(ii) s01 + s02 only, with E1 in place of D1 (one planned run) & {kk}/{nn} & {pct(g.loc['normal', 'E1_D3p'])} "
                 f"[{pct(g.loc['normal', 'lo'])}--{pct(g.loc['normal', 'hi'])}] & " + " & ".join(det) + " & -- \\\\ "
                 "% [S] reports/faz_g/01_e1_ii_summary.csv (plan faz_g/DESIGN_G.md)")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("tab_unseen.tex", "\n".join(lines))


# ---------------------------------------------------------------- supplement: per-session step durations (A4.1c)
def tab_steps():
    s = pd.read_csv(A4 / "01_session_confound_c_steps.csv")
    names = {"session_02(normal)": "s02 (normal, 1--2--5)", "session_03(seq135)": "s03 (order 1--3--5)",
             "session_04(seq145)": "s04 (order 1--4--5)", "session_05(fast)": "s05 (fast)", "session_06(slow)": "s06 (slow)"}
    cols = ["Box1 kalisi", "Box1->sonraki kutu gecisi", "ikinci kutu kalisi", "->Box5 gecisi", "Box5 kalisi", "dongu suresi"]
    lines = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r"Session & Cycles & Expected order & Box~1 dwell & Box~1 $\to$ second box & Second-box dwell & $\to$ Box~5 transit & Box~5 dwell & Cycle duration \\",
             r"\midrule"]
    for _, r in s.iterrows():
        cells = [str(r[c]).replace("[", "[").replace(", ", "--") for c in cols]
        lines.append(f"{names[r.oturum]} & {r.n_dongu} & {r.n_beklenen_dizi} & " + " & ".join(cells)
                     + " \\\\ % [S] faz_a4/01_session_confound_c_steps.csv")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("supp_steps.tex", "\n".join(lines))


# ---------------------------------------------------------------- supplement: opposite-channel controls (A4.1b)
def tab_opposite():
    b = pd.read_csv(A4 / "01_session_confound_b_rates.csv")
    keep = [("D1", ["s02 (normal)", "s05 (fast)", "s06 (slow)", "tempo (s05+s06)", "s03 (seq135)", "s04 (seq145)"]),
            ("D2", ["s02 (normal)", "s03 (seq135)", "s04 (seq145)", "seq (s03+s04)", "s05 (fast)", "s06 (slow)"])]
    lab = {"s02 (normal)": "s02 normal (false-alarm rate)", "s05 (fast)": "s05 fast", "s06 (slow)": "s06 slow",
           "tempo (s05+s06)": "s05 + s06 pooled", "s03 (seq135)": "s03 order 1--3--5", "s04 (seq145)": "s04 order 1--4--5",
           "seq (s03+s04)": "s03 + s04 pooled"}
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Channel & Session & Pairs / cycles & Alarm rate at $\alpha=0.10$ (\%) & Alarm rate at $\alpha=0.05$ (\%) \\",
             r"\midrule"]
    for ch, groups in keep:
        for gname in groups:
            row = {}
            for a in (0.1, 0.05):
                r = b[(b.kanal == ch) & (b.grup == gname) & (abs(b.alpha - a) < 1e-9)].iloc[0]
                if gname.startswith("s02"):
                    row[a] = f"{pct(r.oran)} [W {pct(r.wilson_lo)}--{pct(r.wilson_hi)}]"
                else:
                    row[a] = f"{pct(r.oran)} [B {pct(r.boot_lo)}--{pct(r.boot_hi)}]"
                npair, ncyc = int(r.n_cift), int(r.n_dongu)
            lines.append(f"{ch} & {lab[gname]} & {npair} / {ncyc} & {row[0.1]} & {row[0.05]} \\\\ "
                         f"% [S] faz_a4/01_session_confound_b_rates.csv ({ch}, {gname})")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", ""]
    write("supp_opposite.tex", "\n".join(lines))


# ---------------------------------------------------------------- supplement: tempo channel split (A4.2)
def tab_tempo():
    sp = pd.read_csv(A4 / "02_tempo_channel_split.csv")
    d2 = pd.read_csv(A4 / "02_tempo_secondary_D2alone.csv")
    cells = [("b2", "pred", "B2 states, predicted cycles"), ("tof", "tof", "ToF labels, ToF cycles (oracle)"),
             ("tof", "pred", "ToF labels, predicted cycles"), ("b2", "tof", "B2 states, ToF cycles")]
    lines = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
             r" & & & \multicolumn{3}{c}{D3 alarms at $\alpha=0.10$ raised by} & & \\",
             r"\cmidrule(lr){4-6}",
             r"Cell & Type & D3 detection (\%) & D1 only & D2 only & Both & D2 alone, $\alpha=0.10$ (\%) & D2 alone, FAR (\%) \\",
             r"\midrule"]
    for src, seg, lab in cells:
        dd = d2[(d2.state_source == src) & (d2.segmentation == seg)].iloc[0]
        for at, tl in [("fast", "fast"), ("slow", "slow"), ("tempo", "fast + slow")]:
            r = sp[(sp.state_source == src) & (sp.segmentation == seg) & (sp.atype == at)].iloc[0]
            dlab = lab if at == "fast" else ""
            far = f"{pct(dd.FAR_D2_a10)} [{pct(dd.FAR_D2_a10_wlo)}--{pct(dd.FAR_D2_a10_whi)}]" if at == "fast" else ""
            lines.append(f"{dlab} & {tl} & {pct(r.D3alarm)} ({int(r.n_D3)}/{int(r.n_pairs)}) & {int(r.n_D1_only)} & {int(r.n_D2_only)} & {int(r.n_both)} & "
                         f"{pct(dd['det_' + at + '_D2_a10'])} [{pct(dd['det_' + at + '_D2_a10_lo'])}--{pct(dd['det_' + at + '_D2_a10_hi'])}] & {far} \\\\ "
                         f"% [S] faz_a4/02_tempo_channel_split.csv, 02_tempo_secondary_D2alone.csv ({src} x {seg}, {at})")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", ""]
    write("supp_tempo.tex", "\n".join(lines))


# ---------------------------------------------------------------- supplement: Keras per fold (A4.3b)
def tab_keras_folds():
    p = pd.read_csv(A4 / "03_leakage_matched_keras_per_run.csv")
    s1 = p[p.split == "S1"].pivot(index="run", columns="model", values="macro_f1")
    a1 = p[p.split == "S1"].pivot(index="run", columns="model", values="acc")
    # Phase D: the GTX 1650 SUPER column (A4.3a) is dropped; every CNN-GRU column is an RTX 3090 run.
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"S1 fold & Keras (initial architecture) & PyTorch port & B2 & Keras $-$ B2 (points) \\",
             r"\midrule"]
    for k in s1.index:
        r = s1.loc[k]
        lines.append(f"{k} & {r.keras:.3f} ({pct(a1.loc[k, 'keras'])}) & {r.cnngru_remote:.3f} ({pct(a1.loc[k, 'cnngru_remote'])}) & "
                     f"{r.b2:.3f} ({pct(a1.loc[k, 'b2'])}) & {100 * (r.keras - r.b2):+.2f} \\\\ "
                     f"% [S] faz_a4/03_leakage_matched_keras_per_run.csv (S1 fold {k}; keras A4.3b and cnngru_remote Phase A, both RTX 3090)")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("supp_keras_folds.tex", "\n".join(lines))


# ---------------------------------------------------------------- supplement: unseen-session detail (A4.4)
def tab_unseen_detail():
    d = pd.read_csv(A4 / "04_unseen_anomaly_table.csv")
    srcs = [("FazA_S1", "Pooled S1 (primary)"), ("ii_s01s02", "s01 + s02 only"), ("i_S2", "Leave one session out")]
    lines = [r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
             r" & & & \multicolumn{5}{c}{Detection rate at $\alpha=0.10$ (\%)} & \multicolumn{2}{c}{AUROC} \\",
             r"\cmidrule(lr){4-8}\cmidrule(lr){9-10}",
             r"Cycles & Score & Estimator & 1--3--5 & 1--4--5 & Fast & Slow & All & Sequence & Tempo \\",
             r"\midrule"]
    for seg, segl in [("pred", "Predicted"), ("tof", "ToF")]:
        for var in ["D1", "D2", "D3"]:
            for key, lab in srcs:
                r = d[(d.state_source == key) & (d.segmentation == seg) & (d.variant == var)].set_index("atype")
                det = [pct(r.loc[a, "det_a10"]) for a in ["seq135", "seq145", "fast", "slow", "tum anomali"]]
                lines.append(f"{segl} & {var} & {lab} & " + " & ".join(det)
                             + f" & {r.loc["seq(135+145)", "auroc_mean"]:.3f} & {r.loc["tempo(fast+slow)", "auroc_mean"]:.3f} \\\\ "
                             f"% [S] faz_a4/04_unseen_anomaly_table.csv ({key}, {seg}, {var})")
            lines.append(r"\addlinespace")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", ""]
    write("supp_unseen_detail.tex", "\n".join(lines))


if __name__ == "__main__":
    tab_leak()
    tab_unseen()
    tab_steps()
    tab_opposite()
    tab_tempo()
    tab_keras_folds()
    tab_unseen_detail()

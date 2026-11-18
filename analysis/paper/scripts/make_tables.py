"""Tables T1-T4 as booktabs \\input-able fragments (tabular only; captions live in figures/CAPTIONS.md).
Every number is read from a verified result file; a LaTeX comment after each row names the source and the category
([P] primary, [S] secondary, [D] descriptive). Nothing is recomputed except display rounding."""
import re
import numpy as np, pandas as pd
from style import REP, TAB

def f3(x): return f"{x:.3f}"
def pm(m, s): return f"{m:.3f} $\\pm$ {s:.3f}"
def pmp(m, s): return f"{m*100:.1f} $\\pm$ {s*100:.1f}"  # rates in percent
def f1p(x): return f"{x*100:.1f}"  # rate in percent, one decimal
def thou(n): return format(n, ",").replace(",", "{,}")  # LaTeX thousands separator
def b(s): return f"\\textbf{{{s}}}"

def write(name, lines):
    (TAB / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)

# ------------------------------------------------------------------ T1 sessions
sess = [  # from probe/00_schema.md (per-session summary, loop structure) and CLAUDE.md 2.1 / A2 scenario map
    ("s01", "Training", "randomised, variable", "--", 4683, 56, "5.60", "7"),
    ("s02", "Normal", "1--2--5", "normal", 2274, 51, "5.00", "7"),
    ("s03", "Sequence 1--3--5", "1--3--5", "normal", 2407, 57, "4.60", "7"),
    ("s04", "Sequence 1--4--5", "1--4--5", "normal", 1928, 49, "4.50", "7"),
    ("s05", "Fast", "1--2--5", "fast", 1780, 49, "3.80", "7"),
    ("s06", "Slow", "1--2--5", "slow", 3200, 52, "8.30", "6"),
]
role = {"s01": "state estimation only (not used for anomaly detection)", "s02": "normal: train / calibrate / test", "s03": "anomaly test", "s04": "anomaly test",
        "s05": "anomaly test", "s06": "anomaly test"}
L = [r"\begin{tabular}{@{}llllrrrl@{}}", r"\toprule",
     r"Session & Scenario & Box sequence & Pace & Frames & Cycles & Median cycle (s) & Role in the evaluation \\", r"\midrule"]
for s, sc, seq, pace, fr, cy, med, ppf in sess:
    L.append(f"{s} & {sc} & {seq} & {pace} & {fr:,} & {cy} & {med} & {role[s]} \\\\ % [D] probe/00_schema.md (frames, loops-1, loop duration median); CLAUDE.md 2.1".replace(",", "{,}"))
L += [r"\midrule", r"Total & & & & 16{,}272 & 314 & & \\ % [D] probe/00_schema.md (frames 16272; 320 loops minus 6 loop-0 fragments = 314)", r"\bottomrule", r"\end{tabular}"]
write("tab1_sessions.tex", L)

# ------------------------------------------------------------------ T2 state estimation
st = pd.read_csv(REP / "faz_a/03_state_table.csv")
mn = {"b1": "B1 nearest centroid", "b2": "B2 handcrafted features + random forest", "cnnonly": "CNN-only", "cnngru": "CNN--GRU"}
def row(split, m, prim=False):
    g = st[(st.split == split) & (st.model == m)].set_index("subset")
    cells = [pmp(g.loc["tum"].acc_mean, g.loc["tum"].acc_sd), pm(g.loc["tum"].macro_f1_mean, g.loc["tum"].macro_f1_sd)]
    for sub in ["(a) kutu", "(b) none", "(c) gecis"]:
        if sub in g.index:
            cells.append(pmp(g.loc[sub].acc_mean, g.loc[sub].acc_sd))
        else:
            cells.append("--")
    name = mn[m]
    if prim:
        name, cells = b(name), [b(c) for c in cells]
    tag = "[P]" if split == "S1" else "[P]"
    return f"{name} & " + " & ".join(cells) + f" \\\\ % {tag} faz_a/03_state_table.csv ({split}, {m})"
L = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
     r"Protocol & Model & Accuracy (\%) & Macro-F1 & Box frames (\%) & None frames (\%) & Transition frames (\%) \\", r"\midrule"]
first = True
for split, lab, models in [("S1", "S1 (cycle-grouped, 5 folds)", ["b1", "b2", "cnnonly", "cnngru"]),
                           ("S2", "S2 (leave one session out, 6 folds)", ["b1", "b2", "cnngru"])]:
    if not first:
        L.append(r"\midrule")
    first = False
    L.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{lab}}}}} \\")
    for m in models:
        L.append(" & " + row(split, m, prim=(m == "b2")))
L += [r"\bottomrule", r"\end{tabular}"]
write("tab2_state.tex", L)

# T2b: per-session macro-F1
names6 = {"session_01_training": "s01", "session_02_normal_seq125": "s02", "session_03_normal_seq135": "s03",
          "session_04_normal_seq145": "s04", "session_05_anomaly_fast": "s05", "session_06_anomaly_slow": "s06"}
L = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
     r" & \multicolumn{4}{c}{S1 (cycle-grouped)} & \multicolumn{3}{c}{S2 (leave one session out)} & \\",
     r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
     r"Session & B1 & B2 & CNN & CNN--GRU & B1 & B2 & CNN--GRU & Windows \\", r"\midrule"]
nfr = {"s01": 4679, "s02": 2270, "s03": 2403, "s04": 1924, "s05": 1776, "s06": 3196}  # windowed frames, faz_a/03_state.md (d)
def sv(split, m, sess_full):
    g = st[(st.split == split) & (st.model == m) & (st.subset == "session:" + sess_full)]
    return f3(g.macro_f1_mean.iloc[0]) if len(g) else "--"
for full, s in names6.items():
    L.append(f"{s} & {sv('S1','b1',full)} & {sv('S1','b2',full)} & {sv('S1','cnnonly',full)} & {sv('S1','cnngru',full)} & "
             f"{sv('S2','b1',full)} & {sv('S2','b2',full)} & {sv('S2','cnngru',full)} & {thou(nfr[s])} \\\\ % [P] faz_a/03_state_table.csv (session rows); frame counts from faz_a/03_state.md (d)")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab2b_state_sessions.tex", L)

# ------------------------------------------------------------------ T3 anomaly detection
an = pd.read_csv(REP / "faz_a/04_anomaly_table.csv")
cf = pd.read_csv(REP / "faz_a2/01_conformal_table.csv")
def arow(src, seg, var, label, tag):
    g = an[(an.state_source == src) & (an.segmentation == seg) & (an.variant == var)].set_index("atype")
    far = g.loc["tum anomali"].far_a10
    cells = [f1p(far), f1p(g.loc["seq(135+145)"].det_a10), f1p(g.loc["tempo(fast+slow)"].det_a10), f1p(g.loc["tum anomali"].det_a10),
             f1p(g.loc["tum anomali"].det_a05), f3(g.loc["seq(135+145)"].auroc_mean), f3(g.loc["tempo(fast+slow)"].auroc_mean),
             pm(g.loc["tum anomali"].auroc_mean, g.loc["tum anomali"].auroc_sd)]
    return label, cells, f"{tag} faz_a/04_anomaly_table.csv ({src}, {seg}, {var})"
def crow(method, label, tag):
    g = cf[(cf.cell == "B2 x pred (ana)") & (cf.variant == "D3") & (cf.method == method)]
    def val(alpha, atype, metric):
        q = g[(g.atype == atype) & (g.metric == metric) & ((g.alpha == alpha) if alpha is not None else g.alpha.isna())]
        return q.iloc[0]
    far = val(0.1, "normal", "far")
    farc = f1p(far.value) if far.lo == far.hi else f"{f1p(far.value)} [{f1p(far.lo)}--{f1p(far.hi)}]"
    d = lambda a, t: val(a, t, "det")
    detc = lambda t: f1p(d(0.1, t).value)
    cells = [farc, detc("seq(135+145)"), detc("tempo(fast+slow)"), detc("tum anomali"), f1p(d(0.05, "tum anomali").value),
             f3(val(None, "seq(135+145)", "auroc").value), f3(val(None, "tempo(fast+slow)", "auroc").value),
             pm(val(None, "tum anomali", "auroc").value, val(None, "tum anomali", "auroc_sd").value)]
    return label, cells, f"{tag} faz_a2/01_conformal_table.csv (B2 x pred, D3, {method})"

def emit(rows, bold=None):
    out = []
    for label, cells, src in rows:
        if bold and label == bold:
            label, cells = b(label), [b(c) for c in cells]
        out.append(f" & {label} & " + " & ".join(cells) + f" \\\\ % {src}")
    return out
hdr =[r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
       r" & False-alarm rate (\%) & \multicolumn{4}{c}{Detection rate (\%)} & \multicolumn{3}{c}{AUROC} \\",
       r"\cmidrule(lr){2-2}\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
       r"Detector & Normal ($\alpha{=}0.10$) & Sequence & Tempo & All & All ($\alpha{=}0.05$) & Sequence & Tempo & All \\", r"\midrule"]
def emit2(rows, bold=None):
    out = []
    for label, cells, src in rows:
        if bold and label == bold:
            label, cells = b(label), [b(c) for c in cells]
        out.append(f"{label} & " + " & ".join(cells) + f" \\\\ % {src}")
    return out
L = hdr
L.append(r"\multicolumn{9}{@{}l}{\emph{Deployable cell (B2 states, predicted cycle boundaries), standard conformal $p$}} \\")
L += emit2([arow("b2", "pred", "D0", "D0 cycle duration", "[P]"), arow("b2", "pred", "D1", "D1 Markov sequence", "[P]"),
            arow("b2", "pred", "D2", "D2 step duration", "[P]"), arow("b2", "pred", "D3", "D3 (D1 + D2)", "[P]")], bold="D3 (D1 + D2)")
L.append(r"\midrule")
L.append(r"\multicolumn{9}{@{}l}{\emph{Comparison: D3 with other state sources}} \\")
L += emit2([arow("cnngru", "pred", "D3", "CNN--GRU states", "[P]"), arow("b2", "tof", "D3", "B2 states, ToF cycle boundaries", "[P]"),
            arow("tof", "tof", "D3", "ToF oracle (reference)", "[P]")])
L.append(r"\midrule")
L.append(r"\multicolumn{9}{@{}l}{\emph{Initial pipeline (TFT + $z$-score) on PyTorch CNN--GRU states, same cycles and conformal calibration}} \\")
L += emit2([arow("cnngru", "pred", "OLD_tft", "TFT sequence score", "[P]"), arow("cnngru", "pred", "OLD_z", "$z$-score duration", "[P]"),
            arow("cnngru", "pred", "OLD3", "TFT + $z$-score (max)", "[P]")])
L.append(r"\midrule")
L.append(r"\multicolumn{9}{@{}l}{\emph{Conformal variants for B2 + D3 (secondary; specified after the primary results were known)}} \\")
L += emit2([crow("std", "standard (primary)", "[P]"), crow("smooth", "smoothed $p$", "[S]"),
            crow("cross", "cross-conformal", "[S]"), crow("cross_smooth", "cross-conformal, smoothed $p$", "[S]")])
L += [r"\bottomrule", r"\end{tabular}"]
write("tab3_anomaly.tex", L)

# ------------------------------------------------------------------ T4 latency
txt = (REP / "faz_a2/05_latency.md").read_text(encoding="utf-8")
blk = txt.split("## Ozet (3 kosunun ortalamasi)")[1].split("## Notlar")[0]
rows = {}
for ln in blk.splitlines():
    m = re.match(r"\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*%([\d.]+)\s*\|", ln)
    if m and not ln.startswith("|---"):
        rows[m.group(1)] = tuple(float(m.group(i)) for i in range(2, 6))
print({k: v for k, v in rows.items()})
lab = [("B1", "B1 nearest centroid", "5 centroids"),
       ("B2", "B2 (features + random forest)", "300 trees, 21 features"),
       ("B2_features_only", "\\quad of which: 21 features from frame statistics", "--"),
       ("CNN-GRU (5 frames, full window)", "CNN--GRU, full 5-frame window", "1{,}921{,}446 parameters"),
       ("CNN-GRU streaming (1 new CNN frame + GRU over 5 cached)", "CNN--GRU, streaming (descriptive)", "1{,}921{,}446 parameters"),
       ("Heatmap build per frame (2x48x48)", "Heatmap construction, per frame", "--"),
       ("Frame stats from points (7 feats, approx.)", "Frame statistics from points, per frame (approximate)", "--")]
L = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
     r"Component & Size & Median (ms) & IQR (ms) & 95th percentile (ms) \\", r"\midrule"]
def ms(x): return f"{x:.2f}" if x < 1 else f"{x:.1f}"  # two decimals below 1 ms, one above
for key, name, size in lab:
    med, iqr, p95, share = rows[key]
    L.append(f"{name} & {size} & {ms(med)} & {ms(iqr)} & {ms(p95)} \\\\ % [S] faz_a2/05_latency.md (mean of 3 runs of per-run medians; Ryzen 7 3700X, 1 thread)")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab4_latency.tex", L)

"""Phase G - 01: ONE run of the frozen E1 rule with state estimator (ii) (B2 trained on s01 + s02 only).
Plan: faz_g/DESIGN_G.md. Runs only with E1_II_GO=1 (corresponding author's go-ahead) and only once (start sentinel).
Order: (1) gate - rebuild estimator (ii) cycles + D3 and reproduce the Table 6 row; stop before any E1 score if it fails.
(2) E1 per (cycle, fold) with route/mu from (ii) probabilities of the s02 training normals; D3' = min(1, 2 min(p_E, p_2)).
(3) Tabulate (s02 natural_dev rows dropped before tabulation, count only) and apply the plan's reading rule.
Run:  E1_II_GO=1 bash tools/local_run.sh faz_g/01_e1_estimator_ii.py 2>&1 | tee -a reports/faz_g/01_stdout.log
"""
import os
import sys

if os.environ.get("E1_II_GO") != "1":
    sys.stderr.write("[G-01] refusing to run: set E1_II_GO=1 only with the corresponding author's go-ahead.\n")
    raise SystemExit(2)

import datetime as _dt  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
sys.path.insert(0, str(ROOT / "faz_e"))
import e1_core as E  # noqa: E402
import a2_cycles as ac  # noqa: E402
import fa_common as fc  # noqa: E402

T0 = time.time()
ALPHA = 0.10
B_BOOT, SEED_BOOT = 10_000, 20261118
OUT = ROOT / "reports" / "faz_g"
OUT.mkdir(parents=True, exist_ok=True)
A4 = ROOT / "reports" / "faz_a4"
OOF_II = A4 / "oof" / "b2_s0102__all.csv.gz"
PLAN = ROOT / "faz_g" / "DESIGN_G.md"
PATHS = {"md": OUT / "01_e1_ii.md", "summary": OUT / "01_e1_ii_summary.csv", "cycles": OUT / "01_e1_ii_cycles.csv.gz",
         "folds": OUT / "01_e1_ii_folds.csv", "perfold": OUT / "01_e1_ii_perfold.csv", "readings": OUT / "01_e1_ii_readings.csv",
         "auroc": OUT / "01_e1_ii_auroc.csv"}
PCOL = [f"p{c}" for c in range(6)]
NONE_P = np.array([1.0, 0, 0, 0, 0, 0])

# ------------------------------------------------------------------ preconditions (before any data is loaded)
if [p for p in PATHS.values() if p.exists()]:
    raise SystemExit("[G-01] refusing: outputs already exist; this analysis is computed once.")
_md = (ROOT / "reports" / "faz_e" / "01_synthetic.md").read_text(encoding="utf-8")
_rec = [ln.split("sha256 `")[1].split("`")[0] for ln in _md.splitlines() if "code `e1_core.py` sha256 `" in ln]
_now = E.file_sha256(ROOT / "faz_e" / "e1_core.py")
if len(_rec) != 1 or _rec[0] != _now:
    raise SystemExit(f"[G-01] refusing: e1_core.py sha256 {_now} differs from the Phase E record {_rec}.")
PLAN_SHA = E.file_sha256(PLAN)
SENTINEL = OUT / "01_STARTED.sentinel"
try:
    with open(SENTINEL, "x", encoding="utf-8") as fh:
        fh.write(f"started {_dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}; plan {PLAN_SHA}; e1_core {_now}\n")
except FileExistsError:
    raise SystemExit("[G-01] refusing: the start sentinel exists, so this run was already started once.")
print(f"[G-01] plan sha256 {PLAN_SHA}; e1_core sha256 {_now} (= Phase E record)")


def hr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def fmt(x, nd=3):
    if isinstance(x, (float, np.floating)):
        return "nan" if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def md_table(df, nd=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(fmt(v, nd) for v in r) + " |")
    return "\n".join(out)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cluster_boot(g, col, rng):
    """Cycle-clustered bootstrap of the pair-level alarm rate (percentile 95%)."""
    by = g.groupby(["session", "cycle_idx"])[col].agg(["sum", "size"]).to_numpy(float)
    if len(by) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(by), size=(B_BOOT, len(by)))
    s, n = by[idx, 0].sum(axis=1), by[idx, 1].sum(axis=1)
    return tuple(np.percentile(s / n, [2.5, 97.5]))


# ================================================================== 1. gate: estimator (ii) cycles + D3 reproduce Table 6
hr("1. Gate: estimator (ii) D3 reproduces Table 6")
oof = pd.read_csv(OOF_II)
LAB = ac.labels_from_oof(oof, strict=False)
C, CNT = ac.build_cycles("ii_s01s02", "pred", LAB)
SA, NCAL = ac.score_group(C, verbose=False)
print("cycle counts:", dict(Counter(C["session"])))
print(pd.DataFrame(NCAL).to_string(index=False))

CX = ac.ctx()
mm = CX.ft[["session", "frame_id"]].merge(oof[["session", "frame_id"] + PCOL], on=["session", "frame_id"], how="left",
                                          validate="1:1")
PALL = mm[PCOL].to_numpy(float)
miss = np.isnan(PALL).any(axis=1)
PALL[miss] = NONE_P
ses = mm["session"].to_numpy()
P, DEB, FIDS = {}, {}, {}
for s in CX.SESS:
    assert int(miss[ses == s].sum()) == fc.L - 1, (s, int(miss[ses == s].sum()))
    P[s] = PALL[ses == s]
    assert len(P[s]) == len(CX.SE[s]) == len(LAB[s])
    DEB[s] = ac.debounce_labels(LAB[s])
    FIDS[s] = CX.SE[s]["frame_id"].to_numpy()
CL = C.to_dict("records")
for c in CL:
    s, i0, i1 = c["session"], int(c["start_idx"]), int(c["end_idx"])
    assert i0 >= fc.L - 1
    c.update(E.e1_inputs(DEB[s], P[s], i0, i1))
    assert tuple(v.box for v in c["vis"]) == tuple(c["seq"]), (s, c["cycle_idx"])

# E1 scores (computed now, tabulated only after the gate below passes)
RE, FINFO = E.score_roles(CL, frame_ids_by_session=FIDS)
KEY = ["session", "cycle_idx", "f", "role"]
SAt = SA[SA["role"] == "test"][KEY + ["D1", "D2", "p1", "p2", "p3"]]
T = RE[RE["role"] == "test"].merge(SAt, on=KEY, how="left", validate="1:1")
assert T[["p2", "p3", "p_E"]].notna().all().all()
T["D3p"] = ac.d3(T["p_E"], T["p2"])
TB = E.oneshot_tables(T, ALPHA)
R = TB["rates"].copy()

ref = pd.read_csv(A4 / "04_unseen_anomaly_table.csv")
ref = ref[(ref.state_source == "ii_s01s02") & (ref.segmentation == "pred") & (ref.variant == "D3")].set_index("atype")
far = pd.read_csv(A4 / "04_unseen_far.csv")
far = far[(far.seg == "pred") & (far.source == "(ii) s01+s02") & (far.variant == "D3") & (far.alpha == 0.1)].iloc[0]
gate = {"normal": (float(R.loc[R.atype == "normal", "phaseA_D3"].iloc[0]), float(far.FAR))}
for at in ("seq135", "seq145", "fast", "slow"):
    gate[at] = (float(R.loc[R.atype == at, "phaseA_D3"].iloc[0]), float(ref.loc[at, "det_a10"]))
print("gate (rebuilt, Table 6):", gate)
if not all(abs(a - b) < 5e-4 for a, b in gate.values()):
    raise SystemExit("[G-01] STOP: estimator (ii) D3 not reproduced; nothing tabulated or written.")
print("GATE PASSED")

# ================================================================== 2. results
hr("2. E1 + D2 with estimator (ii), alpha = 0.10")
FOLDS = pd.DataFrame([{k: v for k, v in fi.items() if k != "train_readings"} for fi in FINFO])
print(FOLDS.to_string(index=False))
rng = np.random.default_rng(SEED_BOOT)
Tt = TB["rows"]
rows = []
for at in ("normal", "seq135", "seq145", "fast", "slow", "all"):
    g = Tt[Tt.atype != "normal"] if at == "all" else Tt[Tt.atype == at]
    k, n = int(g["a_D3p"].sum()), len(g)
    if at == "normal":
        lo, hi = wilson(k, n)
    else:
        lo, hi = cluster_boot(g, "a_D3p", rng)
    rows.append(dict(atype=at, n_pairs=n, n_cycles=int(g[["session", "cycle_idx"]].drop_duplicates().shape[0]),
                     E1_D3p=k / n, lo=lo, hi=hi, D3_ii=g["a_D3"].mean(), pE_alone=g["a_pE"].mean(),
                     path_pE_only=(g["ch_E"] & ~g["ch_2"]).mean(), path_p2_only=(~g["ch_E"] & g["ch_2"]).mean(),
                     path_both=(g["ch_E"] & g["ch_2"]).mean(), S_E_min=g["S_E"].min(), S_E_med=g["S_E"].median()))
S = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(S.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"s02 natural_dev rows dropped before tabulation (count only): {TB['n_natural_dev_dropped']}")
print("normal alarms: D3(ii)", len(TB["normal_alarm_sets"]["phaseA_D3"]), "| E1+D2", len(TB["normal_alarm_sets"]["E1_D3p"]))
print(TB["auc_sum"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

s = S.set_index("atype")
n_norm = int(s.loc["normal", "n_pairs"])
far_k = int(round(s.loc["normal", "E1_D3p"] * n_norm))
d3_k = len(TB["normal_alarm_sets"]["phaseA_D3"])
case_A = s.loc["seq135", "E1_D3p"] >= 0.95 and s.loc["seq145", "E1_D3p"] >= 0.95 and far_k <= d3_k + 1
X, Y, Z = 100 * s.loc["seq135", "E1_D3p"], 100 * s.loc["seq145", "E1_D3p"], 100 * far_k / n_norm
lead = (f"With an estimator that never saw the deviation sessions, E1 + D2 detected {X:.1f}% and {Y:.1f}% of sequence "
        f"deviations at a false-alarm rate of {Z:.1f}% ({far_k}/{n_norm}).")
sentence = lead + (" The result does not depend on the estimator having seen these sessions." if case_A else
                   " The full detection of the pooled run therefore depends partly on the estimator having seen these sessions.")
print("\nreading rule (DESIGN_G section 6):", "case (A)" if case_A else "case (B)")
print(sentence)

# ================================================================== 3. outputs
COLS = ["session", "cycle_idx", "start_frame_id", "end_frame_id", "f", "atype", "reading", "reading_c", "S_E", "p_E", "D1", "D2",
        "p1", "p2", "p3", "D3p", "a_D3", "a_D3p", "a_pE", "ch_E", "ch_2", "explain"]
Tt[COLS].rename(columns={"p3": "D3_ii", "D3p": "D3prime"}).to_csv(PATHS["cycles"], index=False, compression="gzip")
S.to_csv(PATHS["summary"], index=False)
FOLDS.to_csv(PATHS["folds"], index=False)
TB["perfold"].to_csv(PATHS["perfold"], index=False)
TB["readings"].to_csv(PATHS["readings"], index=False)
TB["auc"].to_csv(PATHS["auroc"], index=False)
now = _dt.datetime.now().astimezone()
L = ["# Phase G - E1 + D2 with estimator (ii), single run (secondary analysis) [S]", "",
     f"**Computed once:** {now:%Y-%m-%d %H:%M %Z} by `faz_g/01_e1_estimator_ii.py` (E1_II_GO=1; {time.time() - T0:.0f} s, CPU). "
     f"**Plan:** `faz_g/DESIGN_G.md` sha256 `{PLAN_SHA}`. `faz_e/e1_core.py` sha256 `{_now}` (= Phase E record).", "",
     "## 1. Gate", f"Estimator (ii) D3 rebuilt and equal to Table 6 (3 decimals): {gate}", "",
     f"Cycle counts: {dict(Counter(C['session']))}. s02 natural_dev rows dropped before tabulation (count only): "
     f"{TB['n_natural_dev_dropped']}.", "",
     "## 2. Folds", "", md_table(FOLDS, 3), "",
     "## 3. Results at alpha = 0.10", "Rates over (cycle, fold) test pairs. normal: Wilson 95%; others: cycle-clustered bootstrap "
     f"({B_BOOT} resamples, seed {SEED_BOOT}). D3_ii = estimator (ii) with D1 (Table 6).", "", md_table(S, 3), "",
     "AUROC of S_E (fold mean):", "", md_table(TB["auc_sum"], 3), "",
     "Collapsed readings:", "", md_table(TB["readings"], 3), "",
     "## 4. Reading rule (DESIGN_G section 6)", f"**{'Case (A)' if case_A else 'Case (B)'}.** {sentence}", "",
     "## 5. Provenance", f"- Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}; CPU only."]
for p in (ROOT / "faz_e" / "e1_core.py", Path(__file__).resolve(), ROOT / "faz_a2" / "a2_cycles.py", OOF_II, PLAN):
    L.append(f"- `{p.relative_to(ROOT).as_posix()}` sha256 `{E.file_sha256(p)}`")
PATHS["md"].write_text("\n".join(L) + "\n", encoding="utf-8")
for p in PATHS.values():
    print("wrote", p.relative_to(ROOT).as_posix())
print(f"done in {time.time() - T0:.1f} s")

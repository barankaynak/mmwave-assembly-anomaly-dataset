"""Phase E - 02: ONE-SHOT run of E1 on all sessions (DESIGN_E section 5). The only Phase E script allowed to load s03-s06.

Runs only when E1_ONESHOT_GO=1 is set (to be set only with the corresponding author's go-ahead), only after the battery
report reports/faz_e/01_synthetic.md exists, and only once: it writes a start sentinel (reports/faz_e/02_STARTED.sentinel)
before any data is loaded and refuses if the sentinel or any output exists. If the battery recorded a failed acceptance
item (reports/faz_e/01_acceptance.csv), the plan's "Sapmalar" must hold a dated entry "AMENDMENT <item>" for it (plan
section 4: amend on normal data only, re-date, then run). It also pins e1_core.py to the sha256 recorded by the battery.

Order of work (nothing is re-run, re-tuned or re-thresholded afterwards; no other score variant is computed):
  1. Phase A reproduction: B2 x pred cycles of all sessions rebuilt with faz_a2/a2_cycles.py; D0-D2 and p0-p3 must equal
     reports/faz_a/04_cycle_scores.csv.gz row by row, otherwise stop before any E1 score is computed.
  2. Per cycle: visits with posterior mass, collapse; per fold: route/mu (training normals), S_E (frozen multi-match) and
     its traceback, p_E (calibration normals; anomaly cycles are tested in every fold), D3' = min(1, 2 min(p_E, p_2)).
  3. Summary at alpha = 0.10: Phase A D3 (reproduced), E1 D3', p_E alone, alarm path per channel (p_E only / p_2 only /
     both); FAR (s02); detection for seq135, seq145, fast, slow; AUROC of S_E; s03/s04 cycles whose collapsed reading
     contains Box 2. s02 (normal) natural_dev cycles stay in the pool but are dropped before tabulation (count only);
     natural_dev cycles of the anomaly sessions stay in the rates, as in Phase A. The reproduced Phase A D3 rates must
     equal the published ones (3 decimals) before anything is written.
Run (only with the go-ahead), keeping the console output as part of the record:
  E1_ONESHOT_GO=1 bash tools/local_run.sh faz_e/02_e1_oneshot.py 2>&1 | tee -a reports/faz_e/02_oneshot_stdout.log
"""
import os
import sys

if os.environ.get("E1_ONESHOT_GO") != "1":
    sys.stderr.write("[E1-02] refusing to run: this is the one-shot run of DESIGN_E section 5 on s03-s06. "
                     "Set E1_ONESHOT_GO=1 only with the corresponding author's go-ahead.\n")
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
import e1_core as E  # noqa: E402  (unit tests of DESIGN_E 2.4 run at import)
import a2_cycles as ac  # noqa: E402
import fa_common as fc  # noqa: E402

T_START = time.time()
ALPHA = 0.10
DESIGN_SHA = E.assert_design_frozen()
OUT = ROOT / "reports" / "faz_e"
BATTERY_MD = OUT / "01_synthetic.md"
BATTERY_ACC = OUT / "01_acceptance.csv"
BATTERY_FOLDS = OUT / "01_folds.csv"
BATTERY_NORMAL = OUT / "01_normal_rows.csv"
PATHS = {"md": OUT / "02_oneshot.md", "cycles": OUT / "02_oneshot_cycles.csv.gz", "summary": OUT / "02_oneshot_summary.csv",
         "auroc": OUT / "02_oneshot_auroc.csv", "folds": OUT / "02_oneshot_folds.csv", "readings": OUT / "02_oneshot_readings.csv",
         "perfold": OUT / "02_oneshot_perfold.csv"}
PHASE_A_PUBLISHED = {"normal": 0.078, "seq135": 0.254, "seq145": 0.475, "fast": 0.921, "slow": 0.929}  # FAZ A, B2 x pred, D3
SESSION_OF = {"normal": "session_02_normal_seq125 (s02)", "seq135": "s03", "seq145": "s04", "fast": "s05", "slow": "s06"}
PCOL = [f"p{c}" for c in range(6)]
NONE_P = np.array([1.0, 0, 0, 0, 0, 0])

# ------------------------------------------------------------------ preconditions (before any data is loaded)
if not all(p.exists() for p in (BATTERY_MD, BATTERY_ACC, BATTERY_FOLDS, BATTERY_NORMAL)):
    raise SystemExit("[E1-02] refusing: the battery outputs (reports/faz_e/01_synthetic.md, 01_acceptance.csv, 01_folds.csv, "
                     "01_normal_rows.csv) must exist first.")
existing = [p for p in PATHS.values() if p.exists()]
if existing:
    raise SystemExit(f"[E1-02] refusing: one-shot outputs already exist ({[p.name for p in existing]}); it is computed once.")
ACC = pd.read_csv(BATTERY_ACC)
ACC_FAILED = ACC.loc[~ACC["passed"].astype(str).str.lower().eq("true"), "item"].tolist()
SAPMALAR = (ROOT / "faz_e" / "DESIGN_E.md").read_text(encoding="utf-8").split("## Sapmalar", 1)[1]
_missing = [it for it in ACC_FAILED if f"AMENDMENT {it.split()[0]}" not in SAPMALAR]
if _missing:
    raise SystemExit("[E1-02] refusing: failed battery acceptance item(s) without a dated 'AMENDMENT <item>' entry in the "
                     "plan's Sapmalar: " + "; ".join(_missing) + " (plan section 4: amend on normal data only, re-date, then run).")
# pin e1_core.py to the hash recorded by the battery run (verifier finding M4)
_md = BATTERY_MD.read_text(encoding="utf-8")
_rec = [ln.split("sha256 `")[1].split("`")[0] for ln in _md.splitlines() if "code `e1_core.py` sha256 `" in ln]
_now = E.file_sha256(ROOT / "faz_e" / "e1_core.py")
if len(_rec) != 1 or _rec[0] != _now:
    raise SystemExit(f"[E1-02] refusing: e1_core.py sha256 {_now} differs from the battery's record {_rec}; re-run the battery first.")
# start sentinel: written once, before any data is loaded (verifier finding M3); a second attempt is refused
SENTINEL = OUT / "02_STARTED.sentinel"
try:
    with open(SENTINEL, "x", encoding="utf-8") as _fh:
        _fh.write(f"started {_dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}; plan {DESIGN_SHA}; e1_core {_now}\n")
except FileExistsError:
    raise SystemExit(f"[E1-02] refusing: {SENTINEL.name} exists, so the one-shot was already started once.")
print(f"[E1-02] plan sha256 {DESIGN_SHA}; e1_core sha256 {_now}; battery report present; "
      f"failed battery items with a dated amendment: {ACC_FAILED or 'none'}")


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


# ================================================================== 1. Phase A reproduction (all sessions, B2 x pred)
hr("1. Phase A reproduction gate (B2 x pred, all sessions)")
LAB = ac.source_labels(("b2",))
assert "b2" in LAB, "B2 S1 OOF missing or incomplete"
C, CNT = ac.build_cycles("b2", "pred", LAB["b2"])
SA, NCAL = ac.score_group(C, verbose=False)
REF = pd.read_csv(fc.OUT / "04_cycle_scores.csv.gz")
REF = REF[(REF["state_source"] == "b2") & (REF["segmentation"] == "pred")].reset_index(drop=True)
KEY = ["session", "cycle_idx", "f", "role"]
ok, gate = E.compare_to_reference(SA, REF, KEY)
print("gate:", gate)
if not ok:
    raise SystemExit("[E1-02] STOP: Phase A D0-D3 / p0-p3 not reproduced; no E1 score was computed.")
print(f"GATE PASSED: {gate['n_matched']} (cycle, f, role) rows")
N_NATDEV = int(C["natural_dev"].astype(bool).sum())
print(f"cycles: {len(C)}; per session {dict(Counter(C['session']))}; natural_dev cycles in the pool (count only): {N_NATDEV}")

# ================================================================== 2. E1 inputs and scores
hr("2. E1 scores")
CX = ac.ctx()
oof = fc.load_oof("b2", "S1")
mm = CX.ft[["session", "frame_id"]].merge(oof[["session", "frame_id"] + PCOL], on=["session", "frame_id"], how="left",
                                          validate="1:1")
PALL = mm[PCOL].to_numpy(float)
miss = np.isnan(PALL).any(axis=1)
PALL[miss] = NONE_P
ses = mm["session"].to_numpy()
P, DEB, FIDS = {}, {}, {}
for s in CX.SESS:
    assert int(miss[ses == s].sum()) == fc.L - 1, s          # the window-less leading frames only
    P[s] = PALL[ses == s]
    assert len(P[s]) == len(CX.SE[s]) and len(LAB["b2"][s]) == len(CX.SE[s])
    DEB[s] = ac.debounce_labels(LAB["b2"][s])
    FIDS[s] = CX.SE[s]["frame_id"].to_numpy()
CL = C.to_dict("records")
for c in CL:
    s, i0, i1 = c["session"], int(c["start_idx"]), int(c["end_idx"])
    assert i0 >= fc.L - 1
    c.update(E.e1_inputs(DEB[s], P[s], i0, i1))
    assert tuple(v.box for v in c["vis"]) == tuple(c["seq"]), (s, c["cycle_idx"])
RE, FINFO = E.score_roles(CL, frame_ids_by_session=FIDS)
FOLDS = pd.DataFrame([{k: v for k, v in fi.items() if k != "train_readings"} for fi in FINFO])
print(FOLDS.to_string(index=False))

# consistency with the battery (route/mu and the s02 normal test rows depend on s02 normal cycles only)
BF = pd.read_csv(BATTERY_FOLDS)
for col in ("route", "mu_1", "mu_2", "mu_5", "n_train", "n_cal", "cal_max"):
    a, b = FOLDS[col].to_numpy(), BF[col].to_numpy()
    assert (a == b).all() if col == "route" else np.allclose(a.astype(float), b.astype(float), atol=1e-12, rtol=0), col
BN = pd.read_csv(BATTERY_NORMAL)
BN = BN[BN["role"] == "test"].set_index(["cycle_idx", "f"])
NT = RE[(RE["session"] == "session_02_normal_seq125") & (RE["role"] == "test")].set_index(["cycle_idx", "f"])
assert len(NT) == len(BN) and np.allclose(NT.loc[BN.index, "S_E"], BN["S_E"], atol=1e-12, rtol=0)
assert np.allclose(NT.loc[BN.index, "p_E"], BN["p_E"], atol=1e-12, rtol=0)
print("route/mu per fold and the s02 normal test S_E / p_E equal the battery's values")

SAt = SA[SA["role"] == "test"][KEY + ["D1", "D2", "p1", "p2", "p3"]]
T = RE[RE["role"] == "test"].merge(SAt, on=KEY, how="left", validate="1:1")
assert T[["p2", "p3", "p_E"]].notna().all().all()
T["D3p"] = ac.d3(T["p_E"], T["p2"])

# ================================================================== 3. summaries
hr("3. Summary at alpha = 0.10 (rates over (cycle, fold) test pairs, as in Phase A)")
TB = E.oneshot_tables(T, ALPHA)
R = TB["rates"].copy()
R.insert(1, "session", R["atype"].map(SESSION_OF))
R["phaseA_published"] = R["atype"].map(PHASE_A_PUBLISHED)
# the reproduced Phase A D3 rates must equal the published ones before anything is written (verifier finding B1)
for _at, _pub in PHASE_A_PUBLISHED.items():
    _got = float(R.loc[R["atype"] == _at, "phaseA_D3"].iloc[0])
    assert abs(_got - _pub) < 5e-4, f"Phase A D3 for {_at}: reproduced {_got:.4f} vs published {_pub}"
print("reproduced Phase A D3 rates equal the published ones:", PHASE_A_PUBLISHED)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(R.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"normal (s02) natural_dev rows dropped before tabulation (count only): {TB['n_natural_dev_dropped']}")
print("normal alarm sets (cycle_idx):", TB["normal_alarm_sets"])
print("\nAUROC of S_E (per fold, then mean / sd over folds):")
print(TB["auc_sum"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nper fold:")
print(TB["perfold"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nreadings (collapsed) per session:")
print(TB["readings"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

BOX2 = []
for at in ("seq135", "seq145"):
    cs = [c for c in CL if c["atype"] == at]
    w2 = [c for c in cs if any(v.box == 2 for v in c["vis_c"])]
    BOX2.append(dict(atype=at, session=SESSION_OF[at], n_cycles=len(cs), n_collapsed_reading_with_box2=len(w2),
                     readings_with_box2=", ".join(f"{k} x{v}" for k, v in Counter(E.reading(c["vis_c"]) for c in w2).most_common())))
BOX2 = pd.DataFrame(BOX2)
print("\ns03/s04 cycles whose collapsed reading contains Box 2 (the only way a sequence deviation can be missed):")
print(BOX2.to_string(index=False))

rt = R.set_index("atype")
auc = TB["auc_sum"].set_index("group") if len(TB["auc_sum"]) else pd.DataFrame()
seqT = TB["rows"][TB["rows"]["atype"].isin(["seq135", "seq145"])]


def g(at, col):
    return float(rt.loc[at, col]) if at in rt.index else float("nan")


def ch_e(at):
    return g(at, "path_pE_only") + g(at, "path_both")


PRED = pd.DataFrame([
    ("seq135, E1 D3' / p_E alone", "0.98-1.00", f"{g('seq135', 'E1_D3p'):.3f} / {g('seq135', 'pE_alone'):.3f}",
     0.98 <= g("seq135", "E1_D3p") and 0.98 <= g("seq135", "pE_alone")),
    ("seq145, E1 D3' / p_E alone", "0.98-1.00", f"{g('seq145', 'E1_D3p'):.3f} / {g('seq145', 'pE_alone'):.3f}",
     0.98 <= g("seq145", "E1_D3p") and 0.98 <= g("seq145", "pE_alone")),
    ("S_E on sequence cycles: min / median", "min >= 7.86; typical 12-16",
     f"{seqT['S_E'].min():.3f} / {seqT['S_E'].median():.3f}" if len(seqT) else "nan",
     bool(len(seqT)) and seqT["S_E"].min() >= 7.86),
    ("AUROC of S_E, seq(135+145) vs normal", "0.98-1.00",
     f"{auc.loc['seq(135+145)', 'auroc_mean']:.3f}" if "seq(135+145)" in auc.index else "nan",
     "seq(135+145)" in auc.index and auc.loc["seq(135+145)", "auroc_mean"] >= 0.98),
    ("fast, E1 D3'", "0.90-1.00", f"{g('fast', 'E1_D3p'):.3f}", g("fast", "E1_D3p") >= 0.90),
    ("slow, E1 D3'", "0.91-0.94", f"{g('slow', 'E1_D3p'):.3f}", 0.91 <= g("slow", "E1_D3p") <= 0.94),
    ("sequence-channel alarms (2 p_E <= 0.10) on tempo sessions: fast / slow", "fast ~ 0.25, slow ~ 0",
     f"{ch_e('fast'):.3f} / {ch_e('slow'):.3f}", None),
    ("FAR, E1 D3'", "0.078 (4/51), same cycles as Phase A", f"{g('normal', 'E1_D3p'):.3f}; sets {TB['normal_alarm_sets']}",
     TB["normal_alarm_sets"]["E1_D3p"] == TB["normal_alarm_sets"]["phaseA_D3"]),
], columns=["quantity", "prediction (section 6)", "observed", "within"])
print("\npredictions (section 6) vs observed:")
print(PRED.to_string(index=False))

CRIT = pd.DataFrame([
    ("(a) tempo retention: fast >= 0.90 and slow >= 0.90 under E1 D3'", f"fast {g('fast', 'E1_D3p'):.3f}, slow {g('slow', 'E1_D3p'):.3f}",
     g("fast", "E1_D3p") >= 0.90 and g("slow", "E1_D3p") >= 0.90),
    ("(b) FAR unchanged: D3' alarms the same normal cycles as Phase A D3", str(TB["normal_alarm_sets"]),
     TB["normal_alarm_sets"]["E1_D3p"] == TB["normal_alarm_sets"]["phaseA_D3"]),
    ("(c) battery acceptance items (reports/faz_e/01_acceptance.csv; a failed item counts only with its dated AMENDMENT in Sapmalar)",
     ("failed as written: " + "; ".join(ACC_FAILED) + " -> amended before this run (DESIGN_E Sapmalar)") if ACC_FAILED else "none failed",
     not _missing),
    ("seq135 and seq145 >= 0.95 under E1 D3'", f"seq135 {g('seq135', 'E1_D3p'):.3f}, seq145 {g('seq145', 'E1_D3p'):.3f}",
     g("seq135", "E1_D3p") >= 0.95 and g("seq145", "E1_D3p") >= 0.95),
], columns=["criterion (section 7)", "observed", "holds"])
SUCCESS = bool(CRIT["holds"].all())
print("\nreading-rule criteria (section 7):")
print(CRIT.to_string(index=False))
print(f"success as defined in section 7: {SUCCESS}")

# ================================================================== 4. outputs
hr("4. Outputs")
COLS = ["session", "cycle_idx", "start_frame_id", "end_frame_id", "f", "atype", "reading", "reading_c", "S_E", "p_E", "D1", "D2",
        "p1", "p2", "p3", "D3p", "a_D3", "a_D3p", "a_pE", "ch_E", "ch_2", "explain"]
TB["rows"][COLS].rename(columns={"p3": "D3_phaseA", "D3p": "D3prime"}).to_csv(PATHS["cycles"], index=False, compression="gzip")
R.to_csv(PATHS["summary"], index=False)
TB["auc"].to_csv(PATHS["auroc"], index=False)
FOLDS.to_csv(PATHS["folds"], index=False)
TB["readings"].to_csv(PATHS["readings"], index=False)
TB["perfold"].to_csv(PATHS["perfold"], index=False)

now = _dt.datetime.now().astimezone()
L = []
w = L.append
w("# Phase E - E1 one-shot run on s03-s06 (secondary analysis) [S]")
w("")
w(f"**Computed once:** {now:%Y-%m-%d %H:%M %Z} by `faz_e/02_e1_oneshot.py` (E1_ONESHOT_GO=1; run time {time.time() - T_START:.0f} s, CPU). "
  f"**Plan:** `faz_e/DESIGN_E.md`, frozen part sha256 `{DESIGN_SHA}`. Battery report: `reports/faz_e/01_synthetic.md`; "
  f"failed battery acceptance items, each with a dated amendment in the plan before this run: {('; '.join(ACC_FAILED)) if ACC_FAILED else 'none'}.")
w("")
w("**Status.** Secondary analysis, designed after the Phase A results were known, on the posterior-mass quantity that had been looked at on "
  "the deviation sessions (DESIGN_E section 1), validated on synthetic deviations built from normal cycles, then run once. The Phase A rule "
  "(D3, B2 x pred, alpha = 0.10) stays the planned primary result. Given the Phase A reading table, the s03/s04 numbers check that the "
  "decision layer is consistent; they do not test the signal (plan sections 1 and 7).")
w("")
w("## 1. Phase A reproduction gate")
w(f"{gate['n_matched']} of {gate['n_ref']} (cycle, f, role) rows of `reports/faz_a/04_cycle_scores.csv.gz` (b2, pred) reproduced; max |diff| "
  f"D0-D2 {max(gate['maxdiff_D0'], gate['maxdiff_D1'], gate['maxdiff_D2']):.1e}, p0-p3 "
  f"{max(gate['maxdiff_p0'], gate['maxdiff_p1'], gate['maxdiff_p2'], gate['maxdiff_p3']):.1e}. Route/mu per fold and the s02 normal test S_E/p_E "
  f"equal the battery's values. natural_dev cycles in the pool (all sessions): {N_NATDEV}; s02 ones are dropped before the tables "
  f"(count {TB['n_natural_dev_dropped']}), anomaly-session ones stay in the rates as in Phase A.")
w("")
w("## 2. Folds (fitted on s02 training normals; calibration on s02 calibration normals)")
w("")
w(md_table(FOLDS, 3))
w("")
w("## 3. Summary at alpha = 0.10")
w("Rates over (cycle, fold) test pairs; anomaly cycles are tested in every fold. Alarm path: p_E channel = 2 p_E <= 0.10, p_2 channel = 2 p_2 <= 0.10; "
  "D3' alarms if either fires. `phaseA_published` = the Phase A D3 row of the brief.")
w("")
w(md_table(R.rename(columns={"E1_D3p": "E1_D3'"}), 3))
w("")
w("**AUROC of S_E** (per fold, mean and sd over folds):")
w("")
w(md_table(TB["auc_sum"], 3))
w("")
w("**s03/s04 cycles whose collapsed reading contains Box 2** (the only way a sequence deviation can be missed by E1):")
w("")
w(md_table(BOX2, 3))
w("")
w("**Collapsed readings per session** (pairs, cycles, sequence-channel and D3' alarm rates):")
w("")
w(md_table(TB["readings"].rename(columns={"E1_D3p": "E1_D3'"}), 3))
w("")
w("**Per fold:**")
w("")
w(md_table(TB["perfold"].rename(columns={"E1_D3p": "E1_D3'"}), 3))
w("")
w("## 4. Predictions (section 6, written before the run) against the observed values")
w("")
w(md_table(PRED, 3))
w("")
w("## 5. Reading-rule criteria (section 7)")
w("")
w(md_table(CRIT, 3))
w("")
w(f"**Success as defined in section 7: {SUCCESS}.** Reading rule: s03/s04 >= 0.95 is reported as consistent with the mechanism and foreseeable "
  "from the Phase A readings (not as a predicted-and-confirmed test); < 0.95 is a bug-class finding to be investigated on normal data only, "
  "with no changed rule re-run on s03/s04. If (a) fails, E1 is reported as a sequence-only improvement with the tempo loss in numbers; if (b) "
  "fails, the FAR is reported with its SE, without re-tuning. The full table is reported in every case, negative rows included.")
w("")
w("## 6. Files")
for p in PATHS.values():
    w(f"- `{p.relative_to(ROOT).as_posix()}`")
w("")
w("## 7. Provenance")
w(f"- Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}; host {platform.node()}; CPU only.")
for p in (ROOT / "faz_e" / "e1_core.py", Path(__file__).resolve(), ROOT / "faz_a2" / "a2_cycles.py", ROOT / "faz_a" / "fa_common.py",
          fc.OUT / "04_cycle_scores.csv.gz", fc.OUT / "folds_anom.csv", BATTERY_MD, BATTERY_ACC):
    w(f"- `{p.relative_to(ROOT).as_posix()}` sha256 `{E.file_sha256(p)}`")
for k in range(5):
    p = fc.oof_path("b2", "S1", k)
    w(f"- `{p.relative_to(ROOT).as_posix()}` sha256 `{E.file_sha256(p)}`")
PATHS["md"].write_text("\n".join(L) + "\n", encoding="utf-8")
for p in PATHS.values():
    print(f"wrote {p.relative_to(ROOT).as_posix()}")
print(f"done in {time.time() - T_START:.1f} s")

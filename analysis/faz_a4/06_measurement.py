"""Faz A4 / 06 - measurement characterisation (descriptive; 2026-11-10; planned after all other results were known).

No model is trained. Uses the S1 out-of-fold B2 predictions (each frame predicted by a forest that never saw its cycle)
and the 10 Hz ToF labels that trained the models.
  (a) Radar configuration: values read from the released .cfg file and quantities derived from it.
  (b) Step-duration agreement, B2 vs ToF, on ToF cycle boundaries: for every s02-s06 cycle whose debounced B2 visit
      sequence equals the debounced ToF visit sequence, each dwell and transit is compared (B2 minus ToF):
      bias, s.d., 95% limits of agreement (bias +- 1.96 s.d.), median absolute error; also visit onset and offset
      timing error. Cycles flagged natural_dev (s02) are left out. Share of cycles with equal sequences is reported.
  (c) Alarm latency in seconds (B2 x pred, alpha = 0.10, primary calibration) from reports/faz_a3/03_online_alarms.csv:
      seconds from cycle start to the alarm, and for fast cycles seconds after the cycle end. Normal cycles are left out.
  (d) Timing resolution: detection of D3 on uniformly time-scaled normal cycles, copied from reports/faz_e/01_tempo.csv.
Output: reports/faz_a4/06_measurement.md (+ 06_agreement_steps.csv)
Run: tools/local_run.sh faz_a4/06_measurement.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
sys.path.insert(0, str(ROOT / "faz_a2"))
sys.path.insert(0, str(ROOT / "probe"))
import a2_cycles as ac  # noqa: E402

OUT = ROOT / "reports" / "faz_a4"
L = ["# 06 Measurement characterisation (descriptive; 2026-11-10)", ""]

# ----------------------------------------------------------------------------- (a) radar configuration
CFG = Path(os.environ.get("RADAR_CFG", Path(os.environ.get("DATA_PATH", ROOT / "data")).parent / "sensor_config" / "iwr6843isk_radar_profile.cfg"))
txt = CFG.read_text(encoding="utf-8", errors="replace")
cmd = {}
for ln in txt.splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("%"):
        k, *v = ln.split()
        cmd.setdefault(k, []).append(v)
hdr = dict(re.findall(r"^%\s*([^:\n]+):\s*(.+)$", txt, re.M))
p = [float(x) for x in cmd["profileCfg"][0]]
start_ghz, idle_us, adc_start_us, ramp_us, slope = p[1], p[2], p[3], p[4], p[7]
n_adc, fs_ksps = int(p[9]), p[10]
t_adc_us = n_adc / (fs_ksps / 1000.0)
bw_adc = slope * t_adc_us / 1000.0          # GHz swept during sampling
bw_ramp = slope * ramp_us / 1000.0
c = 299792458.0
rres = c / (2 * bw_adc * 1e9)
fr = [float(x) for x in cmd["frameCfg"][0]]
n_loops, period_ms = int(fr[2]), fr[4]
ch = cmd["channelCfg"][0]
rx, tx = int(ch[0]), int(ch[1])
n_rx, n_tx = bin(rx).count("1"), bin(tx).count("1")
cf = cmd["cfarCfg"]
fov = cmd["cfarFovCfg"]
L += ["## (a) Radar configuration", f"Source: {CFG.name}", "",
      f"- start frequency {start_ghz} GHz; slope {slope} MHz/us; ramp end {ramp_us} us (ramp bandwidth {bw_ramp:.2f} GHz)",
      f"- ADC: {n_adc} samples at {fs_ksps} ksps = {t_adc_us:.2f} us; bandwidth used {bw_adc:.2f} GHz; range resolution c/2B = {rres*100:.2f} cm",
      f"- header: range resolution {hdr.get('Range Resolution(m)')} m; max range {hdr.get('Maximum unambiguous Range(m)')} m; "
      f"max radial velocity {hdr.get('Maximum Radial Velocity(m/s)')} m/s; velocity resolution {hdr.get('Radial velocity resolution(m/s)')} m/s; "
      f"azimuth resolution {hdr.get('Azimuth Resolution(deg)')} deg; frame {hdr.get('Frame Duration(msec)')} ms",
      f"- antennas: RX mask {rx} ({n_rx} RX), TX mask {tx} ({n_tx} TX) -> {n_rx*n_tx} virtual channels, azimuth only (no elevation TX)",
      f"- chirps per frame: {n_loops} loops x {len(cmd['chirpCfg'])} chirps; frame period {period_ms} ms",
      f"- CFAR range: {' '.join(cf[0])}; CFAR Doppler: {' '.join(cf[1])} (procDir mode win guard shift cyclic thr_dB peakgroup)",
      f"- CFAR field of view: range {' '.join(fov[0][1:])} m; Doppler {' '.join(fov[1][1:])} m/s",
      f"- clutterRemoval {' '.join(cmd['clutterRemoval'][0])} (static zero-Doppler removal on); multiObjBeamForming {' '.join(cmd['multiObjBeamForming'][0])}",
      ""]

# ----------------------------------------------------------------------------- (b) agreement
LAB = ac.source_labels(sources=("tof", "b2"), verbose=False)

# (b1) visit-level matching (independent of cycle segmentation): each debounced ToF visit is matched to the debounced
# B2 visits of the same box that overlap it. 1 overlap -> matched; 0 -> missed; >1 -> split. Transits are compared
# between consecutive ToF visits that are both matched 1:1 and whose matched B2 visits are also consecutive in B2.
ctx = ac.ctx()
VR, TR, CNT = [], [], []
for s in ctx.SESS:
    tt = ctx.SE[s]["t"].to_numpy()
    vt = ac.visits_of(ac.debounce_labels(LAB["tof"][s]), tt)
    vb = ac.visits_of(ac.debounce_labels(LAB["b2"][s]), tt)
    bidx = {id(v): k for k, v in enumerate(vb)}
    match = []
    for v in vt:
        ov = [w for w in vb if w[0] == v[0] and w[1] < v[2] and w[2] > v[1]]
        match.append(ov[0] if len(ov) == 1 else None)
        CNT.append(dict(session=s, box=v[0], status="matched" if len(ov) == 1 else ("missed" if not ov else "split")))
        if len(ov) == 1:
            w = ov[0]
            VR.append(dict(session=s, step=f"dwell_{v[0]}", tof=v[4] - v[3], b2=w[4] - w[3],
                           onset=w[3] - v[3], offset=w[4] - v[4]))
    for i in range(len(vt) - 1):
        a, b = match[i], match[i + 1]
        if a is None or b is None or bidx[id(b)] - bidx[id(a)] != 1:
            continue
        TR.append(dict(session=s, step=f"transit_{vt[i][0]}_{vt[i+1][0]}",
                       tof=vt[i + 1][3] - vt[i][4], b2=b[3] - a[4]))
VV = pd.concat([pd.DataFrame(VR), pd.DataFrame(TR)], ignore_index=True)
VV["err"] = VV["b2"] - VV["tof"]
C = pd.DataFrame(CNT)
CT, _ = ac.build_cycles("tof", "tof", LAB["tof"])
CB, _ = ac.build_cycles("b2", "tof", LAB["b2"])
key = ["session", "cycle_idx"]
M = CT.merge(CB, on=key, suffixes=("_t", "_b"))
M = M[~M["natural_dev_t"]]
M["same"] = M["seq_t"] == M["seq_b"]
rows, on_rows = [], []
ctx = ac.ctx()
for r in M[M["same"]].itertuples(index=False):
    seq = r.seq_t
    for i, b in enumerate(seq):
        rows.append(dict(session=r.session, step=f"dwell_{b}", tof=r.dw_t[i], b2=r.dw_b[i]))
    for i in range(len(seq) - 1):
        rows.append(dict(session=r.session, step=f"transit_{seq[i]}_{seq[i+1]}", tof=r.gp_t[i], b2=r.gp_b[i]))
S = pd.DataFrame(rows)
S["err"] = S["b2"] - S["tof"]

# visit onset/offset error on the same cycles
for s in sorted(M["session"].unique()):
    df = ctx.SE[s]
    tt = df["t"].to_numpy()
    vt = ac.visits_of(ac.debounce_labels(LAB["tof"][s]), tt)
    vb = ac.visits_of(ac.debounce_labels(LAB["b2"][s]), tt)
    for r in M[(M["session"] == s) & M["same"]].itertuples(index=False):
        i0, i1 = r.start_idx_t, r.end_idx_t
        a = [v for v in vt if i0 <= v[1] < i1]
        b = [v for v in vb if i0 <= v[1] < i1]
        if [v[0] for v in a] != [v[0] for v in b]:
            continue
        for x, y in zip(a, b):
            on_rows.append(dict(session=s, box=x[0], onset=y[3] - x[3], offset=y[4] - x[4]))
O = pd.DataFrame(on_rows)


def agg(x):
    e = x["err"]
    return pd.Series(dict(n=len(e), tof_median=x["tof"].median(), bias=e.mean(), sd=e.std(ddof=1),
                          loa_lo=e.mean() - 1.96 * e.std(ddof=1), loa_hi=e.mean() + 1.96 * e.std(ddof=1),
                          mae_median=e.abs().median(), within_0p1=(e.abs() <= 0.1 + 1e-9).mean()))


A_step = S.groupby("step").apply(agg).reset_index()
A_sess = S.groupby("session").apply(agg).reset_index()
A_all = agg(S)
A_step.to_csv(OUT / "06_agreement_steps.csv", index=False)
share = M.groupby("session")["same"].agg(["sum", "count"])
L += ["## (b) Step-duration agreement, B2 minus ToF (s), ToF cycle boundaries, equal visit sequences only", "",
      f"Cycles with equal debounced sequences: {int(M['same'].sum())} of {len(M)} "
      f"({M['same'].mean():.3f}); natural-deviation cycles left out.", "",
      "Per session (equal / all): " + "; ".join(f"{i}: {int(v['sum'])}/{int(v['count'])}" for i, v in share.iterrows()), "",
      f"All steps: n {int(A_all.n)}, bias {A_all.bias:+.3f}, sd {A_all.sd:.3f}, LoA [{A_all.loa_lo:+.3f}, {A_all.loa_hi:+.3f}], "
      f"median |error| {A_all.mae_median:.3f}, share within 0.1 s {A_all.within_0p1:.3f}", "",
      "```", A_step.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```", "",
      "```", A_sess.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```", ""]
if len(O):
    L += [f"Visit onset error (B2 - ToF): n {len(O)}, median {O.onset.median():+.3f} s, IQR "
          f"[{O.onset.quantile(.25):+.3f}, {O.onset.quantile(.75):+.3f}], share |error| <= 0.1 s {(O.onset.abs() <= 0.1 + 1e-9).mean():.3f}",
          f"Visit offset error (B2 - ToF): median {O.offset.median():+.3f} s, IQR "
          f"[{O.offset.quantile(.25):+.3f}, {O.offset.quantile(.75):+.3f}], share |error| <= 0.1 s {(O.offset.abs() <= 0.1 + 1e-9).mean():.3f}", ""]

VV["kind"] = np.where(VV["step"].str.startswith("dwell"), "dwell", "transit")
B_step = VV.groupby("step").apply(agg).reset_index()
B_sess = VV.groupby(["session", "kind"]).apply(agg).reset_index()
B_kind = VV.groupby("kind").apply(agg).reset_index()
B_step.to_csv(OUT / "06_agreement_visits.csv", index=False)
st = C.groupby(["session", "status"]).size().unstack(fill_value=0)
stt = C["status"].value_counts()
dw = VV[VV.kind == "dwell"]
L += ["## (b1) Visit-level agreement, B2 minus ToF (s), matched by overlap (main measure)", "",
      f"ToF visits: {len(C)}; matched 1:1 {int(stt.get('matched', 0))} ({stt.get('matched', 0)/len(C):.3f}), "
      f"missed {int(stt.get('missed', 0))} ({stt.get('missed', 0)/len(C):.3f}), split {int(stt.get('split', 0))} ({stt.get('split', 0)/len(C):.3f})", "",
      "```", st.to_string(), "```", "",
      "```", B_kind.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```", "",
      "```", B_step.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```", "",
      "```", B_sess.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```", "",
      f"Onset error of matched visits: median {dw.onset.median():+.3f} s, IQR [{dw.onset.quantile(.25):+.3f}, {dw.onset.quantile(.75):+.3f}], "
      f"share |e| <= 0.1 s {(dw.onset.abs() <= 0.1 + 1e-9).mean():.3f}",
      f"Offset error of matched visits: median {dw.offset.median():+.3f} s, IQR [{dw.offset.quantile(.25):+.3f}, {dw.offset.quantile(.75):+.3f}], "
      f"share |e| <= 0.1 s {(dw.offset.abs() <= 0.1 + 1e-9).mean():.3f}", ""]

# ----------------------------------------------------------------------------- (c) alarm latency in seconds
X = pd.read_csv(ROOT / "reports" / "faz_a3" / "03_online_alarms.csv")
X = X[(X.state_source == "b2") & (X.segmentation == "pred") & (X.alarm_lit.astype(bool)) & (X.atype != "normal")].copy()
X["sec_from_start"] = X["frac_lit"] * X["dur_s"]
X["sec_after_end"] = (X["frac_lit"] - 1.0) * X["dur_s"]
L += ["## (c) Alarm latency (B2 x pred, alpha = 0.10, alarmed anomaly pairs)", ""]
for a, g in X.groupby("atype"):
    L.append(f"- {a}: n {len(g)}; seconds from cycle start median {g.sec_from_start.median():.2f} "
             f"[IQR {g.sec_from_start.quantile(.25):.2f}-{g.sec_from_start.quantile(.75):.2f}]; "
             f"share after cycle end {(g.frac_lit > 1).mean():.3f}; seconds after end (those) median "
             f"{g.loc[g.frac_lit > 1, 'sec_after_end'].median() if (g.frac_lit > 1).any() else float('nan'):.2f}"
             + (f"; seconds after wrong-box entry median {g.sec_from_onset_lit.median():.2f}" if a.startswith("seq") else ""))
L.append("")

# ----------------------------------------------------------------------------- (d) timing resolution
T = pd.read_csv(ROOT / "reports" / "faz_e" / "01_tempo.csv")
L += ["## (d) Timing resolution: D3 on uniformly time-scaled normal cycles (from reports/faz_e/01_tempo.csv)", "",
      "```", T[["variant", "n", "det_D3_phaseA", "det_D2_p2_05", "det_D1_05"]].to_string(index=False), "```", ""]

(OUT / "06_measurement.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

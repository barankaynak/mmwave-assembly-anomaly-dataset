"""Gap 4 (OUTLINE s5.3 / s9): data-minimisation numbers, from the raw recordings (read-only) and the radar cfg.

Sources
  raw chip output : <DATA_PATH>/session_*/mmwave_data.csv  (every point the on-chip pipeline emitted, incl. v == 0 points)
  merged          : <DATA_PATH>/session_*/merged_df.csv    (v == 0 points dropped by the capture software; ToF joined)
  radar cfg       : dataset/sensor_config/iwr6843isk_radar_profile.cfg  (parsed below, nothing hard-coded except TI frame-format constants)

Assumptions about the byte format (TI mmWave SDK 03.06 out-of-box demo UART stream, the SDK named in the cfg header):
  frame header 40 B; every TLV = 8 B header + payload; TLV1 detected points = 16 B/point (x, y, z, v float32); TLV7 side info = 4 B/point
  (snr, noise uint16); TLV2 range profile = nRangeBins x 2 B (enabled by guiMonitor logMagRange = 1); TLV6 stats = 24 B (statsInfo = 1);
  total packet padded to a multiple of 32 B. These are documented SDK constants, NOT measured on the wire in this study.
Raw ADC: nChirps/frame x nRx x nAdcSamples x bytes/complex sample (16-bit I + 16-bit Q = 4 B, `adcCfg 2 1`), i.e. what a raw-data capture board would stream.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DATA_PATH", ROOT / "data"))
CFG = Path(os.environ.get("RADAR_CFG", DATA.parent / "sensor_config" / "iwr6843isk_radar_profile.cfg"))
OUT = ROOT / "reports" / "paper_prep"
DT = 0.1

# ------------------------------------------------------------------ cfg
cfg = {}
for line in CFG.read_text().splitlines():
    if line.startswith("%") or not line.strip():
        continue
    parts = line.split()
    cfg.setdefault(parts[0], []).append(parts[1:])
prof = [float(x) for x in cfg["profileCfg"][0]]
n_adc = int(prof[9]); sample_rate_ksps = prof[10]; slope = prof[7]
rx_mask = int(cfg["channelCfg"][0][0]); tx_mask = int(cfg["channelCfg"][0][1])
n_rx = bin(rx_mask).count("1"); n_tx = bin(tx_mask).count("1")
fc_ = [float(x) for x in cfg["frameCfg"][0]]
chirp_start, chirp_end, n_loops, frame_ms = int(fc_[0]), int(fc_[1]), int(fc_[2]), fc_[4]
chirps_per_loop = chirp_end - chirp_start + 1
n_chirps = chirps_per_loop * n_loops
adc_bits_code, adc_fmt = int(cfg["adcCfg"][0][0]), int(cfg["adcCfg"][0][1])
assert adc_bits_code == 2, "adcCfg numADCBits code 2 = 16 bit"
bytes_per_sample = 4        # complex: 16-bit I + 16-bit Q
raw_adc_bytes = n_chirps * n_rx * n_adc * bytes_per_sample
gui = [int(x) for x in cfg["guiMonitor"][0]]      # subFrame, detectedObjects, logMagRange, noiseProfile, azHeatmap, dopplerHeatmap, stats
n_range_bins = 1 << int(np.ceil(np.log2(n_adc)))
cfar_fov = cfg["cfarFovCfg"]
print(f"cfg: nAdc={n_adc} rx={n_rx} tx={n_tx} chirps/loop={chirps_per_loop} loops={n_loops} -> chirps/frame={n_chirps}; frame={frame_ms} ms;"
      f" raw ADC = {n_chirps}x{n_rx}x{n_adc}x{bytes_per_sample} = {raw_adc_bytes} B/frame; guiMonitor={gui}; range bins (FFT)={n_range_bins}; cfarFovCfg={cfar_fov}")
print(f"chirp duration (idle+ramp) = {(prof[2] + prof[4]):.2f} us; active chirp time/frame = {n_chirps*(prof[2]+prof[4])/1000:.1f} ms of {frame_ms:.0f} ms")
bw_mhz = slope * n_adc / sample_rate_ksps * 1000       # slope MHz/us * (n/fs[us]) ; fs in ksps -> samples per us = ksps/1000
bw_mhz = slope * (n_adc / (sample_rate_ksps / 1000.0))
print(f"swept bandwidth over ADC window = {bw_mhz:.0f} MHz -> range resolution = {3e8/(2*bw_mhz*1e6)*100:.2f} cm; FFT bin = 1 per {3e8/(2*bw_mhz*1e6)*100:.2f} cm; 0.5 m = {0.5/(3e8/(2*bw_mhz*1e6)):.1f} bins")


def frame_bytes(n_pts, points_only=False, full_stream=True):
    n_pts = np.asarray(n_pts, float)
    tlv1 = 8 + 16 * n_pts
    tlv7 = 8 + 4 * n_pts
    if points_only:
        return tlv1 + tlv7
    b = 40 + tlv1 + tlv7
    if gui[2]:
        b = b + 8 + 2 * n_range_bins
    if gui[6]:
        b = b + 8 + 24
    return np.ceil(b / 32.0) * 32


def payload_used(n_pts):
    """what the pipeline actually reads per point: x, y, velocity (float32) + snr (uint16)."""
    return 14.0 * np.asarray(n_pts, float)


# ------------------------------------------------------------------ data
rows, tot = [], []
raw_all, mrg_all = [], []
for sdir in sorted(glob.glob(str(DATA / "session_*"))):
    s = Path(sdir).name
    r = pd.read_csv(Path(sdir) / "mmwave_data.csv")
    m = pd.read_csv(Path(sdir) / "merged_df.csv", usecols=["frame_id", "x", "y", "z", "velocity", "range", "azimuth", "elevation", "snr", "timestamp_mmwave"])
    ts = pd.to_datetime(r["timestamp"], format="mixed")
    fr_ts = ts.groupby(r["frame_id"]).first().sort_index()
    dt = np.diff(fr_ts.to_numpy().astype("datetime64[ns]").astype(np.int64)) / 1e9
    k = np.rint(dt / DT).astype(int)                   # radar periods between consecutive recorded frames
    n_rec = len(fr_ts)
    n_empty_inferred = int((k - 1).clip(min=0).sum())
    n_radar = n_rec + n_empty_inferred
    ppf_raw = r.groupby("frame_id").size()
    ppf_mrg = m.groupby("frame_id").size()
    # per radar frame incl. empty ones (0 points)
    ppf_all = np.r_[ppf_raw.to_numpy(), np.zeros(n_empty_inferred, int)]
    rows.append(dict(session=s, n_points_raw=len(r), n_points_merged=len(m), frames_recorded=n_rec, frames_empty_inferred=n_empty_inferred,
                     frames_radar_est=n_radar, empty_share=n_empty_inferred / n_radar, dt_off_grid_share=float((np.abs(dt / DT - k) > 0.25).mean()),
                     ppf_raw_med=float(ppf_raw.median()), ppf_raw_p95=float(ppf_raw.quantile(.95)), ppf_mrg_med=float(ppf_mrg.median()),
                     ppf_mrg_p95=float(ppf_mrg.quantile(.95)), ppf_all_mean=float(ppf_all.mean()),
                     range_gt_0p5=float((r["range"] > 0.5).mean()), z_ne_0=float((r["z"] != 0).mean()), elev_ne_0=float((r["elevation"] != 0).mean()),
                     static_share=float((r["velocity"] == 0).mean())))
    raw_all.append(r.assign(session=s)); mrg_all.append(m.assign(session=s))
    tot.append(ppf_all)
S = pd.DataFrame(rows)
R = pd.concat(raw_all, ignore_index=True)
M = pd.concat(mrg_all, ignore_index=True)
ppf_raw_all = R.groupby(["session", "frame_id"]).size()
ppf_mrg_all = M.groupby(["session", "frame_id"]).size()
ppf_incl_empty = np.concatenate(tot)
n_empty = int(S.frames_empty_inferred.sum()); n_radar = int(S.frames_radar_est.sum()); n_rec = int(S.frames_recorded.sum())

q = lambda v, a: float(np.quantile(v, a))
summ = {}
summ["n_points_raw"] = len(R); summ["n_points_merged"] = len(M)
summ["frac_range_gt_0.5m"] = float((R["range"] > 0.5).mean())
summ["range_max_m"] = float(R["range"].max()); summ["range_p0.25"] = q(R["range"], .0025); summ["range_p50"] = q(R["range"], .5); summ["range_p99.75"] = q(R["range"], .9975)
summ["frac_z_ne_0"] = float((R["z"] != 0).mean()); summ["frac_elev_ne_0"] = float((R["elevation"] != 0).mean())
for c in ("x", "y"):
    summ[f"{c}_central99.5_lo"] = q(R[c], .00125); summ[f"{c}_central99.5_hi"] = q(R[c], .99875)
    summ[f"{c}_min"] = float(R[c].min()); summ[f"{c}_max"] = float(R[c].max())
summ["azimuth_min_deg"] = float(R["azimuth"].min()); summ["azimuth_max_deg"] = float(R["azimuth"].max())
grid = (R["x"].between(-0.5, 0.5)) & (R["y"].between(0.1, 1.0))
summ["frac_inside_heatmap_grid_x[-0.5,0.5]_y[0.1,1.0]"] = float(grid.mean())
summ["frac_velocity_eq_0"] = float((R["velocity"] == 0).mean())
summ["frames_recorded"] = n_rec; summ["frames_empty_inferred"] = n_empty; summ["frames_radar_est"] = n_radar; summ["empty_share"] = n_empty / n_radar
summ["ppf_raw_median_recorded"] = float(ppf_raw_all.median()); summ["ppf_raw_p95_recorded"] = q(ppf_raw_all, .95); summ["ppf_raw_mean_recorded"] = float(ppf_raw_all.mean())
summ["ppf_raw_max"] = int(ppf_raw_all.max())
summ["ppf_merged_median"] = float(ppf_mrg_all.median()); summ["ppf_merged_p95"] = q(ppf_mrg_all, .95); summ["ppf_merged_mean"] = float(ppf_mrg_all.mean())
summ["ppf_incl_empty_median"] = float(np.median(ppf_incl_empty)); summ["ppf_incl_empty_p95"] = q(ppf_incl_empty, .95); summ["ppf_incl_empty_mean"] = float(ppf_incl_empty.mean())

# ------------------------------------------------------------------ bytes
by = []
for label, n in (("median (recorded frames)", summ["ppf_raw_median_recorded"]), ("p95 (recorded frames)", summ["ppf_raw_p95_recorded"]),
                 ("mean over ALL radar frames incl. empty", summ["ppf_incl_empty_mean"]), ("max recorded", summ["ppf_raw_max"])):
    for variant, b in (("V1 full on-chip stream as configured (header+TLV1+TLV7+TLV2 range profile+TLV6 stats, padded to 32 B)", float(frame_bytes(n))),
                       ("V2 point-cloud TLVs only (TLV1+TLV7 incl. 8-B TLV headers)", float(frame_bytes(n, points_only=True))),
                       ("V3 payload the pipeline reads (x, y, v float32 + snr uint16 = 14 B/point)", float(payload_used(n)))):
        by.append(dict(points_per_frame_statistic=label, n_points=n, variant=variant, bytes_per_frame=b, raw_adc_bytes_per_frame=raw_adc_bytes,
                       ratio_raw_over_pointcloud=raw_adc_bytes / b))
BY = pd.DataFrame(by)
print(S.round(4).to_string())
print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in summ.items()})
print(BY.round(2).to_string())
print(f"raw ADC data rate: {raw_adc_bytes*10/1e3:.1f} kB/s = {raw_adc_bytes*10*3600/1e9:.2f} GB/h; V2 at median points: {float(frame_bytes(summ['ppf_raw_median_recorded'], True))*10/1e3:.2f} kB/s")
S.to_csv(OUT / "05_gaps_minimisation_sessions.csv", index=False)
BY.to_csv(OUT / "05_gaps_minimisation_bytes.csv", index=False)
pd.DataFrame([dict(item=k, value=v) for k, v in summ.items()] +
             [dict(item="raw_adc_bytes_per_frame", value=raw_adc_bytes), dict(item="n_chirps_per_frame", value=n_chirps), dict(item="n_rx", value=n_rx),
              dict(item="n_adc_samples", value=n_adc), dict(item="range_bins_fft", value=n_range_bins)]).to_csv(OUT / "05_gaps_minimisation_summary.csv", index=False)

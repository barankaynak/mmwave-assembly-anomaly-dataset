"""Shared loaders for the 3-question probe.

Design rules
- Sessions are DISCOVERED from DATA_PATH (any folder named session_* holding merged_df.csv);
  nothing is hard-coded, so new operators/recordings drop in as new folders.
- Data is read-only. Times are seconds since the session's first radar frame.
- Radar = 10 Hz (100 ms frames). ToF = ~30 Hz (distance_sensors.csv).
- Group key for any split is (session, loop_id); loop_id restarts at 0 in every session.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "probe"
REPORTS.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path(
    os.environ.get(
        "DATA_PATH",
        ROOT / "data",  # set DATA_PATH to the v1.0.0 dataset "data" folder
    )
)

BOXES = [1, 2, 3, 4, 5]
ACTIVE_COLS = [f"box{i}_active" for i in BOXES]
DIST_COLS = [f"box{i}_distance" for i in BOXES]
TOF_ACTIVE_MM = 140.0      # paper's rule: hand in box  <=>  ToF distance < 140 mm
FRAME_PERIOD_S = 0.1       # radar frame period (cfg frameCfg ... 100)
NONE = 0                   # label for "no box active"
MULTI = -1                 # label for ">1 box active" (should not occur)


# --------------------------------------------------------------------------- sessions
def sessions() -> list[str]:
    """All session folder names under DATA_PATH, sorted."""
    return sorted(p.name for p in DATA_PATH.glob("session_*") if (p / "merged_df.csv").exists())


def session_info(name: str) -> dict:
    """Scenario/role parsed from the folder name (no fixed list)."""
    n = name.lower()
    m = re.search(r"seq(\d+)", n)
    if "training" in n:
        scen, role, seq = "training", "train", None
    elif "fast" in n:
        scen, role, seq = "fast", "anomaly_tempo", (1, 2, 5)
    elif "slow" in n:
        scen, role, seq = "slow", "anomaly_tempo", (1, 2, 5)
    elif m:
        seq = tuple(int(c) for c in m.group(1))
        scen = "normal" if seq == (1, 2, 5) else f"seq{m.group(1)}"
        role = "normal" if seq == (1, 2, 5) else "anomaly_sequence"
    else:
        scen, role, seq = "unknown", "unknown", None
    return {"session": name, "scenario": scen, "role": role, "expected_seq": seq}


# --------------------------------------------------------------------------- raw loaders
def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="mixed")


@lru_cache(maxsize=None)
def load_merged(session: str) -> pd.DataFrame:
    """merged_df.csv: one row per radar point, ToF columns joined at frame level."""
    df = pd.read_csv(DATA_PATH / session / "merged_df.csv")
    df["ts_mmwave"] = _to_dt(df["timestamp_mmwave"])
    df["ts_dist"] = _to_dt(df["timestamp_dist"])
    return df


@lru_cache(maxsize=None)
def t0(session: str) -> pd.Timestamp:
    return load_merged(session)["ts_mmwave"].min()


def label_from_active(active: np.ndarray) -> np.ndarray:
    """(n,5) 0/1 matrix -> 0 none, 1..5 box, -1 multi."""
    active = np.asarray(active).astype(int)
    s = active.sum(axis=1)
    lab = np.where(s == 0, NONE, np.where(s == 1, active.argmax(axis=1) + 1, MULTI))
    return lab


@lru_cache(maxsize=None)
def load_points(session: str) -> pd.DataFrame:
    """Per-point table: frame_id, t, x, y, velocity, snr, range, azimuth, label, loop_id."""
    df = load_merged(session)
    out = pd.DataFrame(
        {
            "frame_id": df["frame_id"].to_numpy(),
            "t": (df["ts_mmwave"] - t0(session)).dt.total_seconds().to_numpy(),
            "x": df["x"].to_numpy(),
            "y": df["y"].to_numpy(),
            "velocity": df["velocity"].to_numpy(),
            "snr": df["snr"].to_numpy(float),
            "range": df["range"].to_numpy(),
            "azimuth": df["azimuth"].to_numpy(),
            "label": label_from_active(df[ACTIVE_COLS].to_numpy()),
            "loop_id": df["loop_id"].to_numpy(),
        }
    )
    return out


@lru_cache(maxsize=None)
def load_points_raw(session: str) -> pd.DataFrame:
    """Per-point table from the RAW mmwave_data.csv, which still contains the v == 0 points
    (merged_df.csv dropped exactly those). Frame label/loop_id attached by frame_id from load_frames;
    frames absent from merged_df get label NaN. Column `static` = (velocity == 0)."""
    r = pd.read_csv(DATA_PATH / session / "mmwave_data.csv")
    ts = _to_dt(r["timestamp"])
    f = load_frames(session).set_index("frame_id")
    out = pd.DataFrame(
        {
            "frame_id": r["frame_id"].to_numpy(),
            "t": (ts - t0(session)).dt.total_seconds().to_numpy(),
            "x": r["x"].to_numpy(),
            "y": r["y"].to_numpy(),
            "velocity": r["velocity"].to_numpy(),
            "snr": r["snr"].to_numpy(float),
            "range": r["range"].to_numpy(),
            "azimuth": r["azimuth"].to_numpy(),
        }
    )
    out["static"] = out["velocity"] == 0
    out["label"] = out["frame_id"].map(f["label"])
    out["loop_id"] = out["frame_id"].map(f["loop_id"])
    return out


@lru_cache(maxsize=None)
def load_frames(session: str) -> pd.DataFrame:
    """Per-frame table: frame_id, t, n_points, cx, cy (SNR-weighted centroid), snr_sum, label, loop_id."""
    p = load_points(session)
    w = p["snr"].clip(lower=1e-9)
    g = p.assign(wx=p["x"] * w, wy=p["y"] * w, w=w).groupby("frame_id", sort=True)
    fr = pd.DataFrame(
        {
            "t": g["t"].first(),
            "n_points": g.size(),
            "cx": g["wx"].sum() / g["w"].sum(),
            "cy": g["wy"].sum() / g["w"].sum(),
            "snr_sum": g["w"].sum(),
            "label": g["label"].first(),
            "label_nunique": g["label"].nunique(),
            "loop_id": g["loop_id"].first(),
            "loop_nunique": g["loop_id"].nunique(),
        }
    ).reset_index()
    return fr


@lru_cache(maxsize=None)
def load_tof_raw(session: str) -> pd.DataFrame:
    """distance_sensors.csv (~30 Hz): t, box distances, recorded *_active, label (recorded), label_rule (<140 mm)."""
    d = pd.read_csv(DATA_PATH / session / "distance_sensors.csv")
    ts = _to_dt(d["timestamp"])
    d["t"] = (ts - t0(session)).dt.total_seconds()
    d["label"] = label_from_active(d[ACTIVE_COLS].to_numpy())
    d["label_rule"] = label_from_active((d[DIST_COLS].to_numpy() < TOF_ACTIVE_MM).astype(int))
    return d.sort_values("t", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- segments / loops
def runs(labels: np.ndarray, t: np.ndarray, dt_end: float) -> pd.DataFrame:
    """Run-length encode a label sequence. end = start of next run (last run: t_last + dt_end)."""
    labels = np.asarray(labels)
    t = np.asarray(t, float)
    if len(labels) == 0:
        return pd.DataFrame(columns=["label", "start", "end", "duration_s", "n"])
    change = np.r_[True, labels[1:] != labels[:-1]]
    idx = np.flatnonzero(change)
    starts = t[idx]
    ends = np.r_[t[idx[1:]], t[-1] + dt_end]
    n = np.diff(np.r_[idx, len(labels)])
    return pd.DataFrame({"label": labels[idx], "start": starts, "end": ends, "duration_s": ends - starts, "n": n})


def debounce(seg: pd.DataFrame, min_box_s: float, max_flicker_s: float) -> pd.DataFrame:
    """Clean a run table.
    1) a NONE run shorter than max_flicker_s sandwiched between two runs of the SAME box -> merged (ToF flicker);
    2) box runs shorter than min_box_s -> relabelled NONE (spurious touches);
    3) re-merge adjacent equal labels."""
    s = seg.copy().reset_index(drop=True)
    lab = s["label"].to_numpy().copy()
    dur = s["duration_s"].to_numpy()
    for i in range(1, len(s) - 1):
        if lab[i] == NONE and dur[i] < max_flicker_s and lab[i - 1] == lab[i + 1] and lab[i - 1] > 0:
            lab[i] = lab[i - 1]
    lab = np.where((lab > 0) & (dur < min_box_s), NONE, lab)
    s["label"] = lab
    grp = np.r_[0, np.cumsum(lab[1:] != lab[:-1])]
    out = s.groupby(grp).agg(label=("label", "first"), start=("start", "first"), end=("end", "last"), n=("n", "sum"))
    out["duration_s"] = out["end"] - out["start"]
    return out.reset_index(drop=True)


def tof_segments(session: str, source: str = "tof", min_box_s: float = 0.1, max_flicker_s: float = 0.15) -> pd.DataFrame:
    """Station-visit segments (box dwells + NONE gaps) with loop_id attached (loop of the segment midpoint).
    source='tof' -> raw ~30 Hz ToF (recorded *_active); 'frames' -> 10 Hz frame labels from merged_df."""
    if source == "tof":
        d = load_tof_raw(session)
        seg = runs(d["label"].to_numpy(), d["t"].to_numpy(), dt_end=float(np.median(np.diff(d["t"]))))
    elif source == "frames":
        f = load_frames(session)
        seg = runs(f["label"].to_numpy(), f["t"].to_numpy(), dt_end=FRAME_PERIOD_S)
    else:
        raise ValueError(source)
    seg = debounce(seg, min_box_s=min_box_s, max_flicker_s=max_flicker_s)
    f = load_frames(session)
    mid = (seg["start"] + seg["end"]) / 2
    pos = np.clip(np.searchsorted(f["t"].to_numpy(), mid.to_numpy()), 0, len(f) - 1)
    seg["loop_id"] = f["loop_id"].to_numpy()[pos]
    seg["session"] = session
    return seg


def _is_subsequence(needle, hay) -> bool:
    it = iter(hay)
    return all(any(h == n for h in it) for n in needle)


def loops(session: str, source: str = "tof") -> pd.DataFrame:
    """Per loop_id: start/end/duration, visited box order (debounced), complete flag.
    complete = expected sequence appears in order (scenario sessions) / visits >=1 box and box 5 (training)."""
    info = session_info(session)
    f = load_frames(session)
    seg = tof_segments(session, source=source)
    rows = []
    for lid, g in f.groupby("loop_id", sort=True):
        s = seg[(seg["loop_id"] == lid) & (seg["label"] > 0)]
        order = tuple(int(v) for v in s["label"].to_numpy())
        if info["expected_seq"] is not None:
            complete = _is_subsequence(info["expected_seq"], order)
        else:
            complete = (len(order) >= 2) and (5 in order)
        rows.append(
            {
                "session": session,
                "loop_id": int(lid),
                "start": float(g["t"].min()),
                "end": float(g["t"].max() + FRAME_PERIOD_S),
                "duration_s": float(g["t"].max() + FRAME_PERIOD_S - g["t"].min()),
                "n_frames": int(len(g)),
                "box_order": "-".join(map(str, order)),
                "complete": bool(complete),
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Tiny markdown table writer (no tabulate dependency)."""
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v) if np.isfinite(v) else "nan"
        return str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)

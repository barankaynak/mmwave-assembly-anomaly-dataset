"""Faz A shared helpers (frame table, heatmaps, windows, folds, OOF contract).

Everything here follows faz_a/DESIGN.md (pre-registered). Do not change semantics of the folds
once any model has been trained against them.

Public API
----------
constants : K, L, N_CLASSES, X_MIN.., SEED_S1, SEED_S0, N_FOLDS, OUT, OOF_DIR, CACHE_DIR
frame_table()            -> DataFrame, one row per frame (row index == heatmap row)
heatmaps()               -> float32 (N_frames, 2, 48, 48), cached .npy in $CACHE_DIR
windows()                -> DataFrame, one row per L=5 window (16,248), incl. fold columns
win_rows(win=None)       -> int array (Nw, 5): frame-table row indices of the 5 frames of every window
window_batch(idx, ...)   -> float32 (B, 5, 2, 48, 48); idx = positions into windows()
anom_loops()             -> loop table with anomaly cross-fit fold + role
anom_roles(f)            -> {"train": [..], "cal": [..], "test": [f]}
inner_val_split(groups, seed, frac=0.10, strata=None) -> bool mask (True = validation)
oof_path(model, split, fold), write_oof(...), load_oof(model, split)
state_metrics(y_true, y_pred) -> dict
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe"))
import common  # noqa: E402  (read-only loaders)

# ----------------------------------------------------------------------------- constants
K = 48                      # heatmap grid
L = 5                       # window length (frames)
N_CLASSES = 6               # 0 none, 1..5 boxes
NUM_CHANNELS = 2
X_MIN, X_MAX = -0.5, 0.5    # run_all_pt.py
Y_MIN, Y_MAX = 0.1, 1.0
THETA_DIST = 140.0          # mm
SMOOTH_W = 3                # +-3 frame mode smoothing
N_FOLDS = 5
SEED_S1 = 20261104
SEED_S0 = 42
SEED_ANOM = 20261104
HM_VERSION = "v1"

OUT = ROOT / "reports" / "faz_a"
OOF_DIR = OUT / "oof"
CACHE_DIR = Path(os.environ.get("CACHE_DIR", ROOT / ".cache"))
DIST_COLS = common.DIST_COLS
MODELS = ("cnngru", "cnnonly", "b1", "b2")
SPLITS = ("S1", "S0", "S2")


# ----------------------------------------------------------------------------- labels
def mode_smooth(raw: np.ndarray, w: int = SMOOTH_W, n_classes: int = N_CLASSES) -> np.ndarray:
    """Symmetric +-w mode filter (window clipped at the ends); ties -> smallest class id.
    Identical to run_all_pt.derive_labels (np.unique + argmax)."""
    raw = np.asarray(raw, dtype=np.int64)
    n = len(raw)
    onehot = np.zeros((n + 1, n_classes), dtype=np.int64)
    onehot[1:, :] = np.cumsum(np.eye(n_classes, dtype=np.int64)[raw], axis=0)
    i = np.arange(n)
    lo = np.maximum(0, i - w)
    hi = np.minimum(n, i + w + 1)
    counts = onehot[hi] - onehot[lo]
    return counts.argmax(axis=1).astype(np.int64)      # argmax -> first max -> smallest class


def _raw_labels_from_distance(session: str) -> pd.DataFrame:
    """Per frame: argmin(boxN_distance)+1 if min < 140 mm else 0 (derive_labels, NaN -> inf)."""
    m = common.load_merged(session)
    pf = m.groupby("frame_id", sort=True)[DIST_COLS].min()
    d = np.where(np.isnan(pf.to_numpy(dtype=np.float32)), np.inf, pf.to_numpy(dtype=np.float32))
    mn, am = d.min(axis=1), d.argmin(axis=1)
    raw = np.where(mn < THETA_DIST, am + 1, 0).astype(np.int64)
    return pd.DataFrame({"frame_id": pf.index.to_numpy(), "y_raw_dist": raw})


# ----------------------------------------------------------------------------- frame table
@lru_cache(maxsize=1)
def _frame_table() -> pd.DataFrame:
    parts = []
    for si, s in enumerate(common.sessions()):
        f = common.load_frames(s).copy()
        assert (f["label_nunique"] == 1).all() and (f["loop_nunique"] == 1).all(), s
        assert (f["label"] >= 0).all(), f"multi-active frames in {s}"
        # extra per-frame stats from the point table (v != 0 points, as the old pipeline)
        p = common.load_points(s)
        p = p.assign(absv=p["velocity"].abs(), snr_f=p["snr"].astype(float))
        g = p.groupby("frame_id", sort=True)
        ex = pd.DataFrame(
            {
                "snr_mean": g["snr_f"].mean(),
                "snr_max": g["snr_f"].max(),
                "mean_abs_v": g["absv"].mean(),
                "mean_v": g["velocity"].mean(),
                "std_x": g["x"].std(ddof=0),
                "std_y": g["y"].std(ddof=0),
            }
        ).reset_index()
        f = f.merge(ex, on="frame_id", how="left", validate="1:1")
        # labels: raw from distances (== recorded active == common label, asserted), smoothed per session
        rd = _raw_labels_from_distance(s)
        f = f.merge(rd, on="frame_id", how="left", validate="1:1")
        assert (f["y_raw_dist"].to_numpy() == f["label"].to_numpy()).all(), f"label mismatch {s}"
        f["y_raw"] = f["label"].astype(np.int64)
        f["y_true"] = mode_smooth(f["y_raw"].to_numpy())
        yt = f["y_true"].to_numpy()
        idx = np.flatnonzero(yt[1:] != yt[:-1])           # boundary between frame idx and idx+1
        ch = np.zeros(len(f), dtype=bool)
        for d in (-1, 0, 1, 2):                            # 2 frames before + 2 frames after the boundary
            ch[np.clip(idx + d, 0, len(f) - 1)] = True
        f["near_change"] = ch
        f["session"] = s
        f["session_idx"] = si
        f["frame_idx_in_session"] = np.arange(len(f))
        parts.append(f)
    ft = pd.concat(parts, ignore_index=True)
    ft["loop_id"] = ft["loop_id"].astype(int)
    ft["gid"] = ft.groupby(["session", "loop_id"], sort=True).ngroup().astype(int)
    ft["row"] = np.arange(len(ft))
    cols = [
        "row", "session", "session_idx", "frame_id", "t", "loop_id", "gid", "y_raw", "y_true", "near_change",
        "cx", "cy", "n_points", "snr_sum", "snr_mean", "snr_max", "mean_abs_v", "mean_v", "std_x", "std_y",
        "frame_idx_in_session",
    ]
    return ft[cols]


def frame_table() -> pd.DataFrame:
    """One row per frame (merged_df frames, sorted by session then frame_id); `row` == heatmap row.
    y_raw = ToF label (<140 mm); y_true = +-3-frame mode-smoothed label (within session);
    near_change = frame is one of the 2 frames before / 2 frames after a y_true change boundary."""
    return _frame_table().copy()


# ----------------------------------------------------------------------------- heatmaps
def _hm_path() -> Path:
    return CACHE_DIR / f"heatmaps_{HM_VERSION}_K{K}_L{L}.npy"


def _build_heatmaps() -> np.ndarray:
    """Vectorised build_heatmap of run_all_pt.py: ch0 = max SNR per cell, ch1 = sum of velocities in cell
    (the original divides by a binary 'occupied' indicator, i.e. it is a SUM, not a mean) zeroed where ch0 == 0.
    Layout (N, 2, K, K) indexed [frame, channel, y_idx, x_idx]. float32 accumulation in file order."""
    ft = _frame_table()
    n = len(ft)
    ch0 = np.zeros(n * K * K, dtype=np.float32)
    ch1 = np.zeros(n * K * K, dtype=np.float32)
    for s in common.sessions():
        rows = ft.loc[ft["session"] == s, "row"].to_numpy()
        fids = ft.loc[ft["session"] == s, "frame_id"].to_numpy()
        m = common.load_merged(s)
        pos = np.searchsorted(fids, m["frame_id"].to_numpy())
        assert (fids[pos] == m["frame_id"].to_numpy()).all()
        frow = rows[pos]
        xs = m["x"].to_numpy(dtype=np.float32)
        ys = m["y"].to_numpy(dtype=np.float32)
        snr = m["snr"].to_numpy(dtype=np.float32)
        vel = m["velocity"].to_numpy(dtype=np.float32)
        with np.errstate(invalid="ignore"):
            xi = np.floor((xs - X_MIN) / (X_MAX - X_MIN) * K).astype(np.int64)
            yi = np.floor((ys - Y_MIN) / (Y_MAX - Y_MIN) * K).astype(np.int64)
        ok = (xi >= 0) & (xi < K) & (yi >= 0) & (yi < K)
        flat = frow[ok].astype(np.int64) * (K * K) + yi[ok] * K + xi[ok]
        np.maximum.at(ch0, flat, snr[ok])
        np.add.at(ch1, flat, vel[ok])
    ch0 = ch0.reshape(n, K, K)
    ch1 = ch1.reshape(n, K, K)
    ch1 = np.where(ch0 > 0, ch1, 0.0).astype(np.float32)
    return np.stack([ch0, ch1], axis=1).astype(np.float32)


def heatmaps(mmap: bool = False) -> np.ndarray:
    """(N_frames, 2, 48, 48) float32; built once and cached in $CACHE_DIR (atomic write).
    mmap=True returns a read-only memory map (no RAM copy)."""
    p = _hm_path()
    if not p.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        hm = _build_heatmaps()
        tmp = p.with_name(p.stem + f".tmp{os.getpid()}.npy")
        np.save(tmp, hm)
        os.replace(tmp, p)
        if not mmap:
            return hm
    return np.load(p, mmap_mode="r" if mmap else None)


# ----------------------------------------------------------------------------- windows & folds
@lru_cache(maxsize=1)
def _windows() -> pd.DataFrame:
    ft = _frame_table()
    rows = []
    for s, g in ft.groupby("session", sort=True):
        r = g["row"].to_numpy()
        n = len(r)
        last = np.arange(L - 1, n)
        w = pd.DataFrame(
            {
                "session": s,
                "session_idx": int(g["session_idx"].iloc[0]),
                "row_last": r[last],
                **{f"r{k}": r[last - (L - 1) + k] for k in range(L)},
            }
        )
        rows.append(w)
    w = pd.concat(rows, ignore_index=True)
    last = w["row_last"].to_numpy()
    first = w["r0"].to_numpy()
    for c in ("frame_id", "t", "loop_id", "gid", "y_true", "y_raw"):
        w[c] = ft[c].to_numpy()[last]
    w["loop_first"] = ft["loop_id"].to_numpy()[first]
    w["spans_loops"] = w["loop_first"] != w["loop_id"]
    w["contiguous"] = (ft["frame_id"].to_numpy()[last] - ft["frame_id"].to_numpy()[first]) == (L - 1)
    w["win"] = np.arange(len(w))

    # ---- S1: 5-fold StratifiedGroupKFold, group=(session,loop_id), strata=session
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_S1)
    s1 = np.full(len(w), -1, dtype=int)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(w)), w["session_idx"].to_numpy(), w["gid"].to_numpy())):
        assert (s1[te] == -1).all()
        s1[te] = k
    assert (s1 >= 0).all(), "every window needs exactly one S1 fold"
    w["s1"] = s1
    # ---- S2: leave-one-session-out (fold = session index)
    w["s2"] = w["session_idx"].astype(int)
    # ---- S0: random 80/20 window split stratified by class (random_state 42), as the old paper
    idx = np.arange(len(w))
    _, te0 = train_test_split(idx, test_size=0.2, stratify=w["y_true"].to_numpy(), random_state=SEED_S0)
    s0 = np.zeros(len(w), dtype=int)
    s0[te0] = 1
    w["s0_test"] = s0
    cols = ["win", "session", "session_idx", "frame_id", "t", "loop_id", "gid", "y_true", "y_raw",
            "row_last", "r0", "r1", "r2", "r3", "r4", "loop_first", "spans_loops", "contiguous", "s1", "s2", "s0_test"]
    return w[cols]


def windows() -> pd.DataFrame:
    """Window index table (16,248 rows); window = L=5 consecutive frames, label/loop of its LAST frame.
    r0..r4 = frame-table rows (== heatmap rows) of the window's frames (oldest first).
    s1 = S1 fold (0..4), s2 = LOSO fold (= session_idx), s0_test = 1 for the S0 random 20% test windows."""
    return _windows().copy()


def win_rows(win: pd.DataFrame | None = None) -> np.ndarray:
    w = _windows() if win is None else win
    return w[[f"r{k}" for k in range(L)]].to_numpy(dtype=np.int64)


def window_batch(idx, hm: np.ndarray | None = None, win: pd.DataFrame | None = None) -> np.ndarray:
    """(B, 5, 2, 48, 48) float32 for window positions `idx` (positions/`win` column values of windows()).
    Pass `hm` (heatmaps() array, possibly a memmap) to avoid reloading."""
    w = _windows() if win is None else win
    hm = heatmaps(mmap=True) if hm is None else hm
    r = w[[f"r{k}" for k in range(L)]].to_numpy(dtype=np.int64)[np.asarray(idx)]
    return np.ascontiguousarray(hm[r.reshape(-1)].reshape(len(r), L, NUM_CHANNELS, K, K), dtype=np.float32)


# ----------------------------------------------------------------------------- anomaly cross-fit folds
@lru_cache(maxsize=1)
def _anom_loops() -> pd.DataFrame:
    ft = _frame_table()
    rows = []
    for s in common.sessions():
        lp = common.loops(s)                                  # ToF-based: box_order, complete
        g = ft[ft["session"] == s].groupby("loop_id")
        agg = pd.DataFrame({"first_row": g["row"].min(), "last_row": g["row"].max(), "n_frames": g.size(),
                            "t_start": g["t"].min(), "gid": g["gid"].first()}).reset_index()
        agg["session"] = s
        agg = agg.merge(lp[["loop_id", "box_order", "complete"]], on="loop_id", how="left", validate="1:1")
        rows.append(agg)
    a = pd.concat(rows, ignore_index=True)
    info = {s: common.session_info(s) for s in common.sessions()}
    a["scenario"] = a["session"].map(lambda s: info[s]["scenario"])
    a["role"] = a["session"].map(lambda s: info[s]["role"])
    a["fold"] = -1
    normal = a["role"] == "normal"
    ok = normal & a["complete"].astype(bool)
    ids = a.index[ok].to_numpy()
    perm = np.random.RandomState(SEED_ANOM).permutation(len(ids))        # shuffled
    fold = np.empty(len(ids), dtype=int)
    fold[perm] = np.arange(len(ids)) % N_FOLDS                           # round-robin deal -> balanced, interleaved
    a.loc[ids, "fold"] = fold
    exp = a["session"].map(lambda s: "-".join(map(str, info[s]["expected_seq"])) if info[s]["expected_seq"] else "")
    a["natural_dev"] = (a["role"] != "train") & a["complete"].astype(bool) & (a["box_order"].astype(str) != exp)
    a["is_cycle"] = a["complete"].astype(bool)
    a["use"] = np.select(
        [normal & (a["fold"] >= 0), a["role"].isin(["anomaly_sequence", "anomaly_tempo"]) & a["is_cycle"], a["role"] == "train"],
        ["normal_crossfit", "anomaly_test_all_folds", "state_training_session"],
        default="fragment_dropped",
    )
    cols = ["session", "scenario", "role", "loop_id", "gid", "n_frames", "t_start", "first_row", "last_row",
            "box_order", "is_cycle", "natural_dev", "fold", "use"]
    return a[cols].sort_values(["session", "loop_id"]).reset_index(drop=True)


def anom_loops() -> pd.DataFrame:
    """Per (session, loop_id). `fold` in 0..4 only for s02 normal complete cycles (fragments/partials: -1);
    `use` says what the loop is for. Anomaly sessions s03-s06 are test in EVERY f."""
    return _anom_loops().copy()


def anom_roles(f: int) -> dict:
    """Cross-fit roles for the s02 normal cycles: test=[f], cal=[(f+1)%5,(f+2)%5], train=[(f+3)%5,(f+4)%5]."""
    assert 0 <= f < N_FOLDS
    return {"train": [(f + 3) % N_FOLDS, (f + 4) % N_FOLDS],
            "cal": [(f + 1) % N_FOLDS, (f + 2) % N_FOLDS],
            "test": [f]}


# ----------------------------------------------------------------------------- inner validation split
def inner_val_split(groups, seed: int, frac: float = 0.10, strata=None) -> np.ndarray:
    """Group-wise early-stopping validation mask. `groups`: per-sample group id (e.g. windows()['gid'] of the
    TRAINING windows); `strata`: optional per-sample stratum (e.g. session_idx) -> ~frac of the groups of EACH
    stratum (at least 1) go to validation. Returns bool mask, True = validation sample. Deterministic in seed."""
    groups = np.asarray(groups)
    ug, first = np.unique(groups, return_index=True)
    st = np.zeros(len(ug), dtype=int) if strata is None else np.asarray(strata)[first]
    rng = np.random.RandomState(seed)
    val_groups = []
    for sv in np.unique(st):
        g = ug[st == sv]
        n = max(1, int(round(frac * len(g)))) if len(g) > 1 else 0
        val_groups.extend(rng.permutation(g)[:n].tolist())
    return np.isin(groups, np.array(val_groups))


# ----------------------------------------------------------------------------- OOF contract
OOF_COLS = ["session", "frame_id", "t", "loop_id", "fold", "y_true", "y_raw", "y_pred"] + [f"p{c}" for c in range(N_CLASSES)]


def oof_path(model: str, split: str, fold: int) -> Path:
    assert model in MODELS and split in SPLITS, (model, split)
    return OOF_DIR / f"{model}__{split}__f{int(fold)}.csv.gz"


def write_oof(model: str, split: str, fold: int, wins: pd.DataFrame, proba=None, y_pred=None) -> Path:
    """Write one OOF file. `wins` = rows of windows() predicted in this fold; `proba` = (n,6) class probabilities
    (y_pred = argmax) OR `y_pred` only (-> one-hot probabilities). Atomic write."""
    n = len(wins)
    if proba is None:
        assert y_pred is not None
        y_pred = np.asarray(y_pred, dtype=int)
        proba = np.eye(N_CLASSES, dtype=np.float32)[y_pred]
    else:
        proba = np.asarray(proba, dtype=np.float32)
        assert proba.shape == (n, N_CLASSES), proba.shape
        y_pred = proba.argmax(axis=1) if y_pred is None else np.asarray(y_pred, dtype=int)
    df = pd.DataFrame({"session": wins["session"].to_numpy(), "frame_id": wins["frame_id"].to_numpy(),
                       "t": wins["t"].to_numpy(), "loop_id": wins["loop_id"].to_numpy(), "fold": int(fold),
                       "y_true": wins["y_true"].to_numpy(), "y_raw": wins["y_raw"].to_numpy(), "y_pred": y_pred})
    for c in range(N_CLASSES):
        df[f"p{c}"] = proba[:, c]
    p = oof_path(model, split, fold)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    df[OOF_COLS].to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, p)
    return p


def load_oof(model: str, split: str) -> pd.DataFrame:
    """Concatenate all folds of one (model, split); empty DataFrame (OOF_COLS) if none exist."""
    files = sorted(OOF_DIR.glob(f"{model}__{split}__f*.csv.gz"), key=lambda p: int(p.name.split("__f")[1].split(".")[0]))
    if not files:
        return pd.DataFrame(columns=OOF_COLS)
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


# ----------------------------------------------------------------------------- metrics
def state_metrics(y_true, y_pred) -> dict:
    """accuracy, macro-F1 (over the classes present in y_true; zero_division=0), per-class F1 (0..5, nan if absent)."""
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    present = np.unique(y_true)
    f1s = f1_score(y_true, y_pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    out = {"n": int(len(y_true)), "acc": float(accuracy_score(y_true, y_pred)),
           "macro_f1": float(np.mean(f1s[present]))}
    for c in range(N_CLASSES):
        out[f"f1_{c}"] = float(f1s[c]) if c in present else float("nan")
    return out

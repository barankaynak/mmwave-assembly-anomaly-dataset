"""
run_all_pt.py
-------------
PyTorch port for the IJCIM revision (CPU is QEMU-virtual, no AVX, so TF segfaults).

Produces:
  benchmark_results.json   -- CNN-GRU inference latency       (Table 5)
  ablation_results.json    -- 6-variant ablation               (Table 3)

Usage:
  python run_all_pt.py                  # all
  python run_all_pt.py --only bench
  python run_all_pt.py --only ablate
  python run_all_pt.py --epochs 15      # quick smoke test
  python run_all_pt.py --cpu            # disable GPU

Labels are derived from box{1..5}_distance using theta_dist=140mm + mode over
W=3 frames (matches the regional refinement described in the paper §3.3).
"""

import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ---------- args ----------
ap = argparse.ArgumentParser()
ap.add_argument("--only", choices=["bench", "ablate"], default=None)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--data-root", default="../data")
ARGS = ap.parse_args()

DEVICE = torch.device("cuda" if (torch.cuda.is_available() and not ARGS.cpu) else "cpu")

# ---------- constants (must match notebook 01) ----------
SEQ_LEN = 5
GRID = 48
NUM_CHANNELS = 2
NUM_CLASSES = 6                # No Box + Boxes 1..5
X_MIN, X_MAX = -0.5, 0.5
Y_MIN, Y_MAX = 0.1, 1.0
THETA_DIST = 140.0             # mm
SMOOTH_W = 3                   # symmetric window for mode-smoothing
RANDOM_SEED = 42
N_WARMUP = 50
N_TRIALS = 1000

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------------------------------------------------------- data ----
DIST_COLS = [f"box{i}_distance" for i in (1, 2, 3, 4, 5)]

def load_all(data_root):
    files = glob.glob(os.path.join(data_root, "**", "merged_df.csv"), recursive=True)
    if not files: sys.exit(f"no merged_df.csv under {data_root}")
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d["__src"] = os.path.basename(os.path.dirname(f))
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def derive_labels(df):
    """Per-frame label: argmin(boxN_distance)+1 if min < THETA_DIST else 0 (No Box).
    Then mode-smoothed within a +/-SMOOTH_W frame window."""
    # Per-source-experiment so smoothing doesn't cross experiment boundaries.
    df = df.copy()
    out = np.zeros(len(df), dtype=np.int32)
    idx_offset = 0
    for src, g in df.groupby("__src", sort=False):
        # First reduce per-frame: take min across distance cols using one row per frame.
        per_frame = (g.groupby("frame_id", sort=True)[DIST_COLS]
                       .min().reset_index())
        d = per_frame[DIST_COLS].to_numpy(dtype=np.float32)
        # avoid div0: keep nans as inf
        d = np.where(np.isnan(d), np.inf, d)
        mn = d.min(axis=1)
        am = d.argmin(axis=1)
        raw = np.where(mn < THETA_DIST, am + 1, 0).astype(np.int32)
        # mode smoothing
        sm = raw.copy()
        for i in range(len(raw)):
            lo = max(0, i - SMOOTH_W); hi = min(len(raw), i + SMOOTH_W + 1)
            w = raw[lo:hi]
            vals, cnts = np.unique(w, return_counts=True)
            sm[i] = int(vals[cnts.argmax()])
        per_frame["__label"] = sm
        # map back to all rows of this src
        lab_map = dict(zip(per_frame["frame_id"], per_frame["__label"]))
        mask = (df["__src"] == src).to_numpy()
        out[mask] = df.loc[mask, "frame_id"].map(lab_map).to_numpy(dtype=np.int32)
    df["__label"] = out
    return df


def build_heatmap(points, K):
    hm = np.zeros((K, K, NUM_CHANNELS), dtype=np.float32)
    if points.empty: return hm
    xs = points["x"].to_numpy(dtype=np.float32)
    ys = points["y"].to_numpy(dtype=np.float32)
    snr = points["snr"].to_numpy(dtype=np.float32) if "snr" in points.columns else np.ones(len(points), dtype=np.float32)
    vel = points["velocity"].to_numpy(dtype=np.float32) if "velocity" in points.columns else np.zeros(len(points), dtype=np.float32)
    xi = np.floor((xs - X_MIN) / (X_MAX - X_MIN) * K).astype(int)
    yi = np.floor((ys - Y_MIN) / (Y_MAX - Y_MIN) * K).astype(int)
    m = (xi >= 0) & (xi < K) & (yi >= 0) & (yi < K)
    for u, v, s, w in zip(xi[m], yi[m], snr[m], vel[m]):
        if s > hm[v, u, 0]: hm[v, u, 0] = s
        hm[v, u, 1] += w
    counts = (hm[..., 0] > 0).astype(np.float32)
    hm[..., 1] = np.divide(hm[..., 1], counts, out=np.zeros_like(hm[..., 1]), where=counts > 0)
    return hm


def make_sequences(df, L, K):
    X, y = [], []
    for src, g in df.groupby("__src", sort=False):
        frames = sorted(g["frame_id"].unique())
        per_frame_hm, per_frame_lbl = {}, {}
        for f in frames:
            sub = g[g["frame_id"] == f]
            per_frame_hm[f] = build_heatmap(sub, K)
            per_frame_lbl[f] = int(sub["__label"].mode().iloc[0])
        for i in range(L - 1, len(frames)):
            window = [per_frame_hm[frames[j]] for j in range(i - L + 1, i + 1)]
            X.append(np.stack(window, axis=0))
            y.append(per_frame_lbl[frames[i]])
    X = np.asarray(X, dtype=np.float32)          # (N, L, K, K, 2)
    y = np.asarray(y, dtype=np.int64)
    # NCHW: PyTorch expects channels-first per frame; we keep (N, L, C, K, K)
    X = np.transpose(X, (0, 1, 4, 2, 3))
    return X, y


# ---------------------------------------------------------------- models ----
class CNNBlock(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),           nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),          nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        # compute flattened feature length given K
        with torch.no_grad():
            d = torch.zeros(1, NUM_CHANNELS, K, K)
            self.feat_dim = self.net(d).numel()

    def forward(self, x):
        z = self.net(x)
        return z.flatten(1)


class CNNGRU(nn.Module):
    def __init__(self, L, K, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        self.L = L; self.K = K
        self.cnn = CNNBlock(K)
        self.gru = nn.GRU(input_size=self.cnn.feat_dim, hidden_size=hidden, batch_first=True, dropout=0.0)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden, 64); self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):                                    # (B, L, C, K, K)
        B, L, C, K, _ = x.shape
        z = self.cnn(x.reshape(B * L, C, K, K)).reshape(B, L, -1)
        _, h = self.gru(z)
        h = h.squeeze(0)
        return self.fc2(F.relu(self.fc1(self.drop(h))))


class CNNOnly(nn.Module):
    def __init__(self, L, K, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = CNNBlock(K)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.cnn.feat_dim, 64); self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        last = x[:, -1]                                      # (B, C, K, K)
        z = self.cnn(last)
        return self.fc2(F.relu(self.fc1(self.drop(z))))


def n_params(m): return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- benchmark ----
def run_bench():
    log("=== BENCHMARK ===")
    log(f"device={DEVICE} | torch={torch.__version__} | cuda={torch.cuda.is_available()}")
    model = CNNGRU(SEQ_LEN, GRID).to(DEVICE).eval()
    params = n_params(model)
    log(f"CNN-GRU params (PyTorch reimpl, same architecture): {params:,}")

    def bench(dev_str):
        dev = torch.device(dev_str)
        m = model.to(dev).eval()
        x = torch.randn(1, SEQ_LEN, NUM_CHANNELS, GRID, GRID, device=dev)
        # warmup
        with torch.no_grad():
            for _ in range(N_WARMUP): _ = m(x)
            if dev.type == "cuda": torch.cuda.synchronize()
            times = []
            for _ in range(N_TRIALS):
                t0 = time.perf_counter()
                _ = m(x)
                if dev.type == "cuda": torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        a = np.asarray(times)
        return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
                "p95_ms": float(np.percentile(a, 95)), "std_ms": float(a.std()),
                "n_trials": N_TRIALS}

    out = {"framework": "pytorch", "n_parameters": int(params),
           "input_shape": [1, SEQ_LEN, NUM_CHANNELS, GRID, GRID],
           "cpu": bench("cpu")}
    if torch.cuda.is_available() and not ARGS.cpu:
        out["gpu"] = bench("cuda")
        out["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        out["gpu"] = None

    with open("benchmark_results.json", "w") as f: json.dump(out, f, indent=2)
    log("wrote benchmark_results.json")
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------- ablation ----
def train_eval(model, Xtr, ytr, Xte, yte, epochs, cw_tensor):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss(weight=cw_tensor.to(DEVICE))
    # split train -> train/val 90/10
    Xtr_t, Xv_t, ytr_t, yv_t = train_test_split(Xtr, ytr, test_size=0.1, stratify=ytr, random_state=RANDOM_SEED)
    tr_ds = TensorDataset(torch.from_numpy(Xtr_t), torch.from_numpy(ytr_t))
    va_ds = TensorDataset(torch.from_numpy(Xv_t),  torch.from_numpy(yv_t))
    te_ds = TensorDataset(torch.from_numpy(Xte),   torch.from_numpy(yte))
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    best_val_acc, patience, bad = 0.0, 8, 0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(); logits = model(xb); loss = loss_fn(logits, yb); loss.backward(); opt.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                p = model(xb).argmax(1)
                correct += (p == yb).sum().item(); total += yb.numel()
        va_acc = correct / max(total, 1)
        print(f"  epoch {ep:3d}  val_acc={va_acc:.4f}", flush=True)
        if va_acc > best_val_acc + 1e-4:
            best_val_acc, bad, best_state = va_acc, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early-stop at epoch {ep}", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    correct, total = 0, 0; ys_true, ys_pred = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            p = model(xb).argmax(1)
            correct += (p == yb).sum().item(); total += yb.numel()
            ys_true.append(yb.cpu().numpy()); ys_pred.append(p.cpu().numpy())
    test_acc = correct / max(total, 1)
    yp = np.concatenate(ys_pred); yt = np.concatenate(ys_true)
    rep = classification_report(yt, yp, output_dict=True, zero_division=0)
    return {"test_accuracy": float(test_acc),
            "macro_f1": float(rep["macro avg"]["f1-score"]),
            "weighted_f1": float(rep["weighted avg"]["f1-score"]),
            "n_params": int(n_params(model))}


def run_ablate():
    log("=== ABLATION ===")
    df_raw = load_all(ARGS.data_root)
    log(f"loaded {len(df_raw):,} rows; deriving labels...")
    df = derive_labels(df_raw)
    log(f"label distribution: {dict(zip(*np.unique(df['__label'], return_counts=True)))}")

    variants = [
        ("full_L5_K48",     CNNGRU,   5, 48),
        ("cnn_only_L5_K48", CNNOnly,  5, 48),
        ("full_L3_K48",     CNNGRU,   3, 48),
        ("full_L7_K48",     CNNGRU,   7, 48),
        ("full_L5_K32",     CNNGRU,   5, 32),
        ("full_L5_K64",     CNNGRU,   5, 64),
    ]
    out = []
    for name, cls, L, K in variants:
        try:
            log(f"--- variant {name}  (L={L}, K={K})")
            t0 = time.time()
            X, y = make_sequences(df, L, K)
            log(f"    X.shape={X.shape}  y.shape={y.shape}  classes={np.unique(y)}")
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
            cw_np = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
            cw_t = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            for c, w in zip(np.unique(ytr), cw_np): cw_t[int(c)] = float(w)
            model = cls(L, K)
            res = train_eval(model, Xtr, ytr, Xte, yte, ARGS.epochs, cw_t)
            res.update({"name": name, "L": L, "K": K,
                        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                        "wall_seconds": round(time.time() - t0, 1)})
        except Exception as e:
            res = {"name": name, "L": L, "K": K, "error": str(e)}
            log(f"    !! failed: {e}")
        out.append(res)
        with open("ablation_results.json", "w") as f: json.dump(out, f, indent=2)
        log(f"    wrote ablation_results.json  ({len(out)}/{len(variants)})")
    print(json.dumps(out, indent=2))


def main():
    if ARGS.only is None or ARGS.only == "bench":   run_bench()
    if ARGS.only is None or ARGS.only == "ablate":  run_ablate()
    log("all done.")


if __name__ == "__main__":
    main()

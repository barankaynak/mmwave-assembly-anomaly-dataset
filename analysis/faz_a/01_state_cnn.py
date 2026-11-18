"""Faz A / stage 01 - CNN state models (M1 CNN-GRU, M2 CNN-only), OOF predictions (DESIGN.md).

Port notes (documented deviations from the Keras original / old PyTorch port run_all_pt.py):
  * Architecture = run_all_pt.CNNGRU / CNNOnly (Conv->BN->ReLU->MaxPool x3, GRU(128), Dropout(0.4), Dense64, Dense6).
  * Keras GRU(dropout=0.3, recurrent_dropout=0.2) is mirrored by a Dropout(0.3) on the GRU OUTPUT (DESIGN.md);
    this sits in front of the existing Dropout(0.4) of the port (two stacked dropouts). The recurrent dropout is dropped.
  * Adam(lr=5e-4, eps=1e-7 like Keras) with per-step exponential decay 0.92 ** (step / 8000) (Keras ExponentialDecay, non-staircase).
  * Class-weighted CE for training (balanced, from the training portion of the fold only); validation loss is UNWEIGHTED CE
    (what Keras reports as val_loss). Early stopping: val_loss, patience 25, restore best weights, max 120 epochs.
  * Inner validation (never touches the test fold):
      S1/S2: 10% of the training groups ((session, loop_id)) per session; S0: stratified random 10% of training windows
      (mirrors the old protocol, as S0 is the leakage demo).
Usage:
  python 01_state_cnn.py --model cnngru --split S1 [--folds 0,1] [--max-epochs N] [--device auto|cpu]
  python 01_state_cnn.py --plan all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probe"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fa_common as fa  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

BATCH = 32
LR0 = 5e-4
DECAY_STEPS = 8000
DECAY_RATE = 0.92
PATIENCE = 25
MAX_EPOCHS = 120
EVAL_BATCH = 128
MIN_FREE_GB = 1.5
RUNS_CSV = fa.OUT / "cnn_runs.csv"
SPLIT_ID = {"S1": 1, "S0": 0, "S2": 2}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- models
class CNNBlock(nn.Module):
    def __init__(self, K: int = fa.K):
        super().__init__()
        C = fa.NUM_CHANNELS
        self.net = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            self.feat_dim = self.net(torch.zeros(1, C, K, K)).numel()

    def forward(self, x):
        return self.net(x).flatten(1)


class CNNGRU(nn.Module):
    def __init__(self, K: int = fa.K, hidden: int = 128, n_cls: int = fa.N_CLASSES):
        super().__init__()
        self.cnn = CNNBlock(K)
        self.gru = nn.GRU(self.cnn.feat_dim, hidden, batch_first=True)
        self.drop_gru = nn.Dropout(0.3)      # mirrors Keras GRU dropout=0.3 (on GRU output; see header)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden, 64)
        self.fc2 = nn.Linear(64, n_cls)

    def forward(self, x):                    # (B, L, C, K, K)
        B, L, C, K, _ = x.shape
        z = self.cnn(x.reshape(B * L, C, K, K)).reshape(B, L, -1)
        _, h = self.gru(z)
        h = self.drop_gru(h.squeeze(0))
        return self.fc2(F.relu(self.fc1(self.drop(h))))


class CNNOnly(nn.Module):
    last_only = True

    def __init__(self, K: int = fa.K, n_cls: int = fa.N_CLASSES):
        super().__init__()
        self.cnn = CNNBlock(K)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.cnn.feat_dim, 64)
        self.fc2 = nn.Linear(64, n_cls)

    def forward(self, x):                    # only the last frame is used
        return self.fc2(F.relu(self.fc1(self.drop(self.cnn(x[:, -1])))))


MODELS = {"cnngru": CNNGRU, "cnnonly": CNNOnly}


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- data helpers
class Data:
    """Heatmaps in RAM (float32, ~300 MB); batches are gathered per step by window position."""

    def __init__(self):
        self.win = fa.windows()
        self.hm = torch.from_numpy(np.ascontiguousarray(fa.heatmaps()))
        self.rows = fa.win_rows(self.win)                       # (Nw, 5) heatmap rows
        self.y = self.win["y_true"].to_numpy(dtype=np.int64)

    def batch(self, idx: np.ndarray, last_only: bool) -> torch.Tensor:
        r = self.rows[idx]
        if last_only:
            r = r[:, -1:]
        L = r.shape[1]
        x = self.hm[torch.from_numpy(r.reshape(-1))]
        return x.reshape(len(idx), L, fa.NUM_CHANNELS, fa.K, fa.K)


def pick_device(pref: str) -> torch.device:
    if pref == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as e:  # noqa: BLE001
        log(f"cuda mem_get_info failed ({e}); using CPU")
        return torch.device("cpu")
    log(f"cuda free {free / 2**30:.2f} GB / {total / 2**30:.2f} GB")
    return torch.device("cuda") if free >= MIN_FREE_GB * 2**30 else torch.device("cpu")


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and ("out of memory" in str(e).lower() or "cuda" in str(e).lower()))


@torch.no_grad()
def predict(model, data: Data, idx: np.ndarray, device, last_only: bool, labels: np.ndarray | None = None):
    """Returns (probs (n,6) float32, mean unweighted CE if labels given)."""
    model.eval()
    probs, loss_sum = [], 0.0
    for s in range(0, len(idx), EVAL_BATCH):
        b = idx[s:s + EVAL_BATCH]
        logits = model(data.batch(b, last_only).to(device)).float()
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        if labels is not None:
            loss_sum += F.cross_entropy(logits, torch.from_numpy(labels[s:s + EVAL_BATCH]).to(device), reduction="sum").item()
    p = np.concatenate(probs).astype(np.float32)
    return p, (loss_sum / len(idx) if labels is not None else None)


# ----------------------------------------------------------------------------- fold definition
def fold_indices(split: str, fold: int, data: Data):
    """Returns (train_idx, val_idx, test_idx, seed) as positions into windows()."""
    w = data.win
    n = len(w)
    allidx = np.arange(n)
    seed = fa.SEED_S1 + 1000 * SPLIT_ID[split] + fold
    if split == "S1":
        test = allidx[w["s1"].to_numpy() == fold]
        trn = allidx[w["s1"].to_numpy() != fold]
    elif split == "S2":
        test = allidx[w["s2"].to_numpy() == fold]
        trn = allidx[w["s2"].to_numpy() != fold]
    elif split == "S0":
        test = allidx[w["s0_test"].to_numpy() == 1]
        trn = allidx[w["s0_test"].to_numpy() == 0]
    else:
        raise ValueError(split)
    if split == "S0":
        tr, va = train_test_split(trn, test_size=0.1, stratify=data.y[trn], random_state=fa.SEED_S0)
        tr, va = np.sort(tr), np.sort(va)
    else:
        mask = fa.inner_val_split(w["gid"].to_numpy()[trn], seed, 0.10, strata=w["session_idx"].to_numpy()[trn])
        tr, va = trn[~mask], trn[mask]
    assert len(np.intersect1d(tr, test)) == 0 and len(np.intersect1d(va, test)) == 0
    return tr, va, test, seed


# ----------------------------------------------------------------------------- training
def train_fold(model_name: str, split: str, fold: int, data: Data, device: torch.device, max_epochs: int):
    tr, va, te, seed = fold_indices(split, fold, data)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls = MODELS[model_name]
    model = cls().to(device)
    last_only = getattr(model, "last_only", False)
    npar = n_params(model)

    ytr, yva = data.y[tr], data.y[va]
    cw = np.zeros(fa.N_CLASSES, dtype=np.float32)
    present = np.unique(ytr)
    cw[present] = compute_class_weight("balanced", classes=present, y=ytr)
    log(f"[{model_name} {split} f{fold}] device={device} params={npar:,} train={len(tr)} val={len(va)} test={len(te)} "
        f"| train class counts={np.bincount(ytr, minlength=6).tolist()} cw={np.round(cw, 2).tolist()}")
    wt = torch.from_numpy(cw).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR0, eps=1e-7)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: DECAY_RATE ** (s / DECAY_STEPS))
    gen = torch.Generator().manual_seed(seed)

    best, best_ep, bad, best_state = float("inf"), 0, 0, None
    t_start = time.time()
    ep_run = 0
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        tl, tn = 0.0, 0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            xb = data.batch(b, last_only).to(device)
            yb = torch.from_numpy(data.y[b]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, weight=wt)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
            tn += len(b)
        _, vl = predict(model, data, va, device, last_only, yva)
        ep_run = ep
        improved = vl < best
        if improved:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"[{model_name} {split} f{fold}] ep {ep:3d} train_loss={tl / tn:.4f} val_loss={vl:.4f} "
            f"{'*' if improved else ' '} bad={bad:2d} lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.1f}s")
        if bad >= PATIENCE:
            log(f"[{model_name} {split} f{fold}] early stop at epoch {ep} (best {best_ep}, val_loss {best:.4f})")
            break
    model.load_state_dict(best_state)
    wall = time.time() - t_start

    probs, _ = predict(model, data, te, device, last_only)
    w_te = data.win.iloc[te]
    fa.write_oof(model_name, split, fold, w_te, proba=probs)
    yp = probs.argmax(1)
    m = fa.state_metrics(data.y[te], yp)
    log(f"[{model_name} {split} f{fold}] DONE epochs_run={ep_run} best_epoch={best_ep} best_val_loss={best:.4f} wall={wall:.0f}s "
        f"| TEST n={m['n']} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f}")
    row = {"model": model_name, "split": split, "fold": fold, "epochs_run": ep_run, "best_epoch": best_ep,
           "best_val_loss": round(best, 5), "wall_s": round(wall, 1), "n_params": npar, "device": str(device),
           "n_train": len(tr), "n_val": len(va), "n_test": len(te), "test_acc": round(m["acc"], 5),
           "test_macro_f1": round(m["macro_f1"], 5), "max_epochs": max_epochs}
    append_run(row)
    return row


def append_run(row: dict) -> None:
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(RUNS_CSV, mode="a", header=not RUNS_CSV.exists(), index=False)


def run_job(model_name: str, split: str, folds, data: Data, dev_pref: str, max_epochs: int, skip_existing: bool):
    for f in folds:
        if skip_existing and fa.oof_path(model_name, split, f).exists():
            log(f"[{model_name} {split} f{f}] OOF exists -> skip")
            continue
        device = pick_device(dev_pref)
        try:
            train_fold(model_name, split, f, data, device, max_epochs)
        except Exception as e:  # noqa: BLE001
            if device.type == "cuda" and is_oom(e):
                log(f"[{model_name} {split} f{f}] CUDA failure ({str(e)[:80]}) -> rerun fold on CPU")
                torch.cuda.empty_cache()
                train_fold(model_name, split, f, data, torch.device("cpu"), max_epochs)
            else:
                raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--split", choices=list(SPLIT_ID))
    ap.add_argument("--folds", default=None, help="comma list; default = all folds of the split")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--plan", choices=["all"], default=None)
    ap.add_argument("--no-skip", action="store_true", help="plan mode: retrain even if the OOF file exists")
    a = ap.parse_args()
    torch.set_num_threads(8)
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} threads={torch.get_num_threads()}")
    data = Data()
    log(f"windows={len(data.win)} hm={tuple(data.hm.shape)}")
    nf = {"S1": fa.N_FOLDS, "S0": 1, "S2": 6}
    if a.plan == "all":
        for m, s in [("cnngru", "S1"), ("cnngru", "S0"), ("cnnonly", "S1"), ("cnngru", "S2")]:
            run_job(m, s, range(nf[s]), data, a.device, a.max_epochs, not a.no_skip)
    else:
        assert a.model and a.split, "--model and --split required unless --plan all"
        folds = [int(x) for x in a.folds.split(",")] if a.folds else list(range(nf[a.split]))
        run_job(a.model, a.split, folds, data, a.device, a.max_epochs, False)
    log("all done")


if __name__ == "__main__":
    main()

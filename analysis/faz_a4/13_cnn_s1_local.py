"""Faz A4 / A4.3 (a) - CNN-GRU S1 (5 kat) YEREL GTX 1650 SUPER'da yeniden egitim (esle-donanim sizinti karsilastirmasi).

Kod yolu ve tohum kurallari faz_a/01_state_cnn.py ile AYNI (faz_a2/12_cnn_s0_seeds.py ile ayni yontemle importlib ile ice alinir;
01 ve 12 degistirilmedi): fold_indices('S1', k) (S1 ic dogrulama = grup bazli %10, tohum = SEED_S1 + 1000*1 + k), torch tohumu = ayni
tohum, Adam 5e-4 eps 1e-7 + ustel azalma 0.92^(step/8000), batch 32, sinif agirlikli CE, dogrulama kaybi agirliksiz CE,
EarlyStopping(val_loss, patience 25, restore best), en fazla 120 epoch. 12 ile ayni "train fold" govdesi; yalniz fark (Faz A
dosyalarina yazilmasin diye):
  (1) OOF -> reports/faz_a4/oof/cnngru__S1local__f{k}.csv.gz (fa_common.OOF_COLS sutunlari; fa.write_oof cagrilmaz);
  (2) kosu ozeti -> reports/faz_a4/13_cnn_s1_local_runs.csv;
  (3) cihaz: CUDA; yalniz CUDA OOM/hatasinda CPU'ya dusulur ('device' alaninda belirtilir).
Smoke: --smoke -> gercek dosya yazilmaz (CACHE_DIR/smoke_a4_13), --max-epochs ile kisaltilabilir.
Kullanim: python faz_a4/13_cnn_s1_local.py [--folds 0,1,2,3,4] [--max-epochs N] [--smoke]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
sys.path.insert(0, str(ROOT / "probe"))
import numpy as np
import pandas as pd
import fa_common as fa

spec = importlib.util.spec_from_file_location("state_cnn_01", ROOT / "faz_a" / "01_state_cnn.py")
m01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m01)

import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

log = m01.log
OUT4 = ROOT / "reports" / "faz_a4"
OOF4 = OUT4 / "oof"
RUNS_CSV = OUT4 / "13_cnn_s1_local_runs.csv"
OOF_DIR = OOF4


def oof4_path(fold: int) -> Path:
    return OOF_DIR / f"cnngru__S1local__f{int(fold)}.csv.gz"


def write_oof4(fold: int, wins: pd.DataFrame, proba) -> Path:
    proba = np.asarray(proba, dtype=np.float32)
    assert proba.shape == (len(wins), fa.N_CLASSES)
    y_pred = proba.argmax(axis=1)
    df = pd.DataFrame({"session": wins["session"].to_numpy(), "frame_id": wins["frame_id"].to_numpy(),
                       "t": wins["t"].to_numpy(), "loop_id": wins["loop_id"].to_numpy(), "fold": int(fold),
                       "y_true": wins["y_true"].to_numpy(), "y_raw": wins["y_raw"].to_numpy(), "y_pred": y_pred})
    for c in range(fa.N_CLASSES):
        df[f"p{c}"] = proba[:, c]
    p = oof4_path(fold)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    df[fa.OOF_COLS].to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, p)
    return p


def train_fold_local(fold: int, data, device: torch.device, max_epochs: int):
    model_name, split = "cnngru", "S1"
    tr, va, te, seed = m01.fold_indices(split, fold, data)       # 01 ile ayni: seed = SEED_S1 + 1000*1 + fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = m01.MODELS[model_name]().to(device)
    last_only = getattr(model, "last_only", False)
    npar = m01.n_params(model)
    BATCH, LR0, DECAY_RATE, DECAY_STEPS, PATIENCE = m01.BATCH, m01.LR0, m01.DECAY_RATE, m01.DECAY_STEPS, m01.PATIENCE
    tag = f"[cnngru S1local f{fold}]"

    ytr, yva = data.y[tr], data.y[va]
    cw = np.zeros(fa.N_CLASSES, dtype=np.float32)
    present = np.unique(ytr)
    cw[present] = compute_class_weight("balanced", classes=present, y=ytr)
    log(f"{tag} device={device} seed={seed} params={npar:,} train={len(tr)} val={len(va)} test={len(te)} "
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
        _, vl = m01.predict(model, data, va, device, last_only, yva)
        ep_run = ep
        improved = vl < best
        if improved:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"{tag} ep {ep:3d} train_loss={tl / tn:.4f} val_loss={vl:.4f} "
            f"{'*' if improved else ' '} bad={bad:2d} lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.1f}s")
        if bad >= PATIENCE:
            log(f"{tag} early stop at epoch {ep} (best {best_ep}, val_loss {best:.4f})")
            break
    model.load_state_dict(best_state)
    wall = time.time() - t_start

    probs, _ = m01.predict(model, data, te, device, last_only)
    write_oof4(fold, data.win.iloc[te], probs)
    m = fa.state_metrics(data.y[te], probs.argmax(1))
    log(f"{tag} DONE epochs_run={ep_run} best_epoch={best_ep} best_val_loss={best:.4f} wall={wall:.0f}s "
        f"| TEST n={m['n']} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f}")
    row = {"model": model_name, "split": "S1local", "fold": fold, "seed": seed, "epochs_run": ep_run, "best_epoch": best_ep,
           "best_val_loss": round(best, 5), "wall_s": round(wall, 1), "n_params": npar, "device": str(device),
           "n_train": len(tr), "n_val": len(va), "n_test": len(te), "test_acc": round(m["acc"], 5),
           "test_macro_f1": round(m["macro_f1"], 5), "max_epochs": max_epochs}
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(RUNS_CSV, mode="a", header=not RUNS_CSV.exists(), index=False)
    return row


def main():
    global RUNS_CSV, OOF_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=",".join(str(i) for i in range(fa.N_FOLDS)))
    ap.add_argument("--max-epochs", type=int, default=m01.MAX_EPOCHS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="CPU'ya zorla (yalniz test)")
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    if a.smoke:
        sm = fa.CACHE_DIR / "smoke_a4_13"
        sm.mkdir(parents=True, exist_ok=True)
        RUNS_CSV = sm / "runs.csv"
        OOF_DIR = sm
    torch.set_num_threads(3)
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} threads={torch.get_num_threads()} folds={folds} max_epochs={a.max_epochs}")
    if torch.cuda.is_available():
        log(f"GPU {torch.cuda.get_device_name(0)} free={torch.cuda.mem_get_info()[0] / 2**30:.2f} GB")
    data = m01.Data()
    log(f"windows={len(data.win)} hm={tuple(data.hm.shape)}")
    for f in folds:
        if oof4_path(f).exists() and not a.smoke:
            log(f"[cnngru S1local f{f}] OOF exists -> skip")
            continue
        dev = torch.device("cpu") if (a.cpu or not torch.cuda.is_available()) else torch.device("cuda")
        try:
            train_fold_local(f, data, dev, a.max_epochs)
        except Exception as e:  # noqa: BLE001
            if dev.type == "cuda" and m01.is_oom(e):
                log(f"[cnngru S1local f{f}] CUDA FAILURE/OOM ({str(e)[:120]}) -> rerun this fold on CPU")
                torch.cuda.empty_cache()
                train_fold_local(f, data, torch.device("cpu"), a.max_epochs)
            else:
                raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log("all done")


if __name__ == "__main__":
    main()

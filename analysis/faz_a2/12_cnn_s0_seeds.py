"""Faz A2 / A2.2 (b) - CNN-GRU, S0 (rastgele pencere bolmesi), tohumlar 1,2,3,4 (42 = Faz A'dan, yeniden egitilmez).

Mimari / egitim / erken durdurma / ic dogrulama faz_a/01_state_cnn.py ile AYNI: modeller, Data, predict, pick_device, is_oom ve
sabitler (BATCH, LR0, DECAY_*, PATIENCE, MAX_EPOCHS) 01'den importlib ile ice alinir (01'in __main__ korumasi var; 01 degistirilmedi).
`fold_indices` (S0 dali) ve `train_fold` 01'den BIREBIR kopyadir; yalniz sunlar farkli (Faz A dosyalarina yazilmasin diye):
  (1) test = a2_s0.s0_test_idx(seed) (fa_common satir 234 ile ayni train_test_split cagrisi, random_state=seed);
      ic dogrulama = sinif-tabakali rastgele %10, random_state=seed (01'in S0 dalindaki cagri, SEED_S0 yerine seed);
  (2) model (torch) tohumu = 20261104 + seed (01: 20261104 + 1000*0 + 0 = 20261104, tohum 42 icin);
  (3) OOF -> reports/faz_a2/oof/cnngru__S0s{seed}__f0.csv.gz (a2_s0.write_oof2; fa.write_oof cagrilmaz);
  (4) kosu ozeti -> reports/faz_a2/12_cnn_s0_runs.csv (01'in cnn_runs.csv'sine dokunulmaz);
  (5) cihaz: CUDA; yalniz CUDA OOM/hatasinda CPU'ya dusulur ve log + csv'de 'device' alaninda belirtilir.
Smoke: --smoke  -> hicbir gercek dosya yazilmaz (CACHE_DIR/smoke_a2_12 altina yazar).
Kullanim: python faz_a2/12_cnn_s0_seeds.py [--seeds 1,2,3,4] [--max-epochs N] [--smoke]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
import numpy as np
import pandas as pd
import a2_s0 as s0

fa = s0.fc
spec = importlib.util.spec_from_file_location("state_cnn_01", ROOT / "faz_a" / "01_state_cnn.py")
m01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m01)

import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

log = m01.log
RUNS_CSV = s0.OUT2 / "12_cnn_s0_runs.csv"
OOF_OVERRIDE_DIR = None        # smoke modunda dolu


def fold_indices_s0(seed: int, data):
    w = data.win
    trn, te = s0.s0_masks(w, seed)
    tr, va = s0.inner_val(trn, data.y, seed)                  # 01: train_test_split(trn, 0.1, stratify=y[trn], random_state=SEED_S0)
    assert len(np.intersect1d(tr, te)) == 0 and len(np.intersect1d(va, te)) == 0
    assert len(tr) + len(va) + len(te) == len(w)
    return tr, va, te, s0.MODEL_SEED0 + seed


def train_seed(seed: int, data, device: torch.device, max_epochs: int):
    model_name, split, fold = "cnngru", "S0", 0
    tr, va, te, mseed = fold_indices_s0(seed, data)
    torch.manual_seed(mseed)
    np.random.seed(mseed)
    cls = m01.MODELS[model_name]
    model = cls().to(device)
    last_only = getattr(model, "last_only", False)
    npar = m01.n_params(model)
    BATCH, LR0, DECAY_RATE, DECAY_STEPS, PATIENCE = m01.BATCH, m01.LR0, m01.DECAY_RATE, m01.DECAY_STEPS, m01.PATIENCE
    tag = f"[cnngru S0s{seed}]"

    ytr, yva = data.y[tr], data.y[va]
    cw = np.zeros(fa.N_CLASSES, dtype=np.float32)
    present = np.unique(ytr)
    cw[present] = compute_class_weight("balanced", classes=present, y=ytr)
    log(f"{tag} device={device} model_seed={mseed} params={npar:,} train={len(tr)} val={len(va)} test={len(te)} "
        f"| train class counts={np.bincount(ytr, minlength=6).tolist()} cw={np.round(cw, 2).tolist()}")
    wt = torch.from_numpy(cw).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR0, eps=1e-7)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: DECAY_RATE ** (s / DECAY_STEPS))
    gen = torch.Generator().manual_seed(mseed)

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
    w_te = data.win.iloc[te]
    path = (OOF_OVERRIDE_DIR / f"cnngru__S0s{seed}__f0.csv.gz") if OOF_OVERRIDE_DIR else None
    s0.write_oof2("cnngru", seed, w_te, proba=probs, path=path)
    yp = probs.argmax(1)
    m = fa.state_metrics(data.y[te], yp)
    log(f"{tag} DONE epochs_run={ep_run} best_epoch={best_ep} best_val_loss={best:.4f} wall={wall:.0f}s "
        f"| TEST n={m['n']} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f}")
    row = {"model": model_name, "split": f"S0s{seed}", "seed": seed, "model_seed": mseed, "epochs_run": ep_run,
           "best_epoch": best_ep, "best_val_loss": round(best, 5), "wall_s": round(wall, 1), "n_params": npar,
           "device": str(device), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "test_acc": round(m["acc"], 5), "test_macro_f1": round(m["macro_f1"], 5), "max_epochs": max_epochs}
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(RUNS_CSV, mode="a", header=not RUNS_CSV.exists(), index=False)
    return row


def main():
    global RUNS_CSV, OOF_OVERRIDE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=m01.MAX_EPOCHS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="CPU'ya zorla (yalniz test)")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    assert 42 not in seeds, "tohum 42 Faz A'dan (cnngru__S0__f0); yeniden egitilmez"
    if a.smoke:
        sm = fa.CACHE_DIR / "smoke_a2_12"
        sm.mkdir(parents=True, exist_ok=True)
        RUNS_CSV = sm / "runs.csv"
        OOF_OVERRIDE_DIR = sm
    torch.set_num_threads(3)
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} threads={torch.get_num_threads()} seeds={seeds} max_epochs={a.max_epochs}")
    if torch.cuda.is_available():
        log(f"GPU {torch.cuda.get_device_name(0)} free={torch.cuda.mem_get_info()[0] / 2**30:.2f} GB")
    data = m01.Data()
    log(f"windows={len(data.win)} hm={tuple(data.hm.shape)}")
    s0.assert_seed42_matches_faz_a(data.win)
    log("seed 42 S0 split == Faz A (assert OK)")
    for seed in seeds:
        target = s0.oof2_path("cnngru", seed) if not OOF_OVERRIDE_DIR else OOF_OVERRIDE_DIR / f"cnngru__S0s{seed}__f0.csv.gz"
        if target.exists() and not a.smoke:
            log(f"[cnngru S0s{seed}] OOF exists -> skip")
            continue
        dev = torch.device("cpu") if (a.cpu or not torch.cuda.is_available()) else torch.device("cuda")
        try:
            train_seed(seed, data, dev, a.max_epochs)
        except Exception as e:  # noqa: BLE001
            if dev.type == "cuda" and m01.is_oom(e):
                log(f"[cnngru S0s{seed}] CUDA FAILURE/OOM ({str(e)[:120]}) -> rerun this seed on CPU")
                torch.cuda.empty_cache()
                train_seed(seed, data, torch.device("cpu"), a.max_epochs)
            else:
                raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log("all done")


if __name__ == "__main__":
    main()

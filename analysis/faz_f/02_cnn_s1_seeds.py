"""Faz F / 02 - CNN-GRU (PyTorch port) under S1 with more seeds, RTX 3090 (plan: faz_f/DESIGN_F.md, part F.2).

Imports faz_a4/13_cnn_s1_local.py (which imports faz_a/01_state_cnn.py) unchanged: architecture, training, early stopping,
class weights, batch, learning-rate schedule. Only the seed changes: replica r uses
    seed = SEED_S1 + 1000*1 + fold + 100*r
for the inner validation split (group-wise 10 %) and for torch/numpy. r = 0 is the primary seed (reproduction check).
Outputs (never the Faz A / A4 files):
    reports/faz_f/oof/cnngru__S1r{r}__f{fold}.csv.gz ; reports/faz_f/02_cnn_s1_runs.csv
No CPU fallback: exits if CUDA is missing or the GPU name does not contain --require-gpu.
Usage: python faz_f/02_cnn_s1_seeds.py --replicas 0 --folds 0          (reproduction of the primary fold 0)
       python faz_f/02_cnn_s1_seeds.py --replicas 1,2,3,4               (all folds)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a4_13", ROOT / "faz_a4" / "13_cnn_s1_local.py")
m13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m13)
m01, fa = m13.m01, m13.fa

import torch  # noqa: E402

OUTF = ROOT / "reports" / "faz_f"
_orig_fold_indices = m01.fold_indices
REPLICA = {"r": 0}


def fold_indices_r(split, fold, data):
    assert split == "S1"
    r = REPLICA["r"]
    w = data.win
    tr0, va0, test, seed0 = _orig_fold_indices(split, fold, data)
    seed = seed0 + 100 * r
    if r == 0:
        return tr0, va0, test, seed0
    trn = sorted(set(tr0.tolist()) | set(va0.tolist()))
    import numpy as np
    trn = np.array(trn)
    mask = fa.inner_val_split(w["gid"].to_numpy()[trn], seed, 0.10, strata=w["session_idx"].to_numpy()[trn])
    tr, va = trn[~mask], trn[mask]
    assert len(np.intersect1d(tr, test)) == 0 and len(np.intersect1d(va, test)) == 0
    return tr, va, test, seed


m01.fold_indices = fold_indices_r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", default="1,2,3,4")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=m01.MAX_EPOCHS)
    ap.add_argument("--require-gpu", default="3090")
    a = ap.parse_args()
    if not torch.cuda.is_available() or a.require_gpu not in torch.cuda.get_device_name(0):
        print("no matching GPU; refusing to run on CPU", flush=True)
        return 3
    m13.log(f"GPU {torch.cuda.get_device_name(0)} free={torch.cuda.mem_get_info()[0] / 2**30:.2f} GB torch {torch.__version__}")
    torch.set_num_threads(3)
    data = m01.Data()
    dev = torch.device("cuda")
    for r in [int(x) for x in a.replicas.split(",")]:
        REPLICA["r"] = r
        m13.OOF_DIR = OUTF / "oof"
        m13.RUNS_CSV = OUTF / "02_cnn_s1_runs.csv"
        m13.oof4_path = (lambda fold, r=r: OUTF / "oof" / f"cnngru__S1r{r}__f{int(fold)}.csv.gz")
        for f in [int(x) for x in a.folds.split(",")]:
            if m13.oof4_path(f).exists():
                m13.log(f"r{r} f{f}: OOF exists -> skip")
                continue
            m13.log(f"=== replica {r} fold {f}")
            row = m13.train_fold_local(f, data, dev, a.max_epochs)
            torch.cuda.empty_cache()
    m13.log("all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

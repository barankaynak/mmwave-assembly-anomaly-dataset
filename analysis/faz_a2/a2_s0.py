"""Faz A2 / A2.2 ortak yardimcilar: S0 (rastgele sinif-tabakali %80/20 pencere bolmesi) tohum ailesi + OOF yazimi.

Tohum semasi (DESIGN_A2.md birakmis, burada sabitlenir; B1/B2/CNN-GRU'ya ayni uygulanir):
  * split_seed in {42,1,2,3,4}: test = train_test_split(arange(N), test_size=0.2, stratify=y_true, random_state=split_seed)
    (fa_common._windows satir 234 ile ayni cagri; split_seed=42 Faz A split_s0.csv'yi BIREBIR uretmeli -> assert).
  * ic dogrulama (erken durdurma / B1 tau secimi): egitim pencerelerinden sinif-tabakali rastgele %10, random_state=split_seed
    (01_state_cnn.py S0 dali ile ayni: train_test_split(trn, test_size=0.1, stratify=y[trn], random_state=SEED_S0)).
  * model tohumu (torch baslangic / RF random_state) = 20261104 + split_seed  (her tohum tam bir replikat; S1'de de kat basina degisir).
    NOT: Faz A'nin CNN-GRU S0 tohum-42 kosusu torch tohumu 20261104 kullanmisti (+0); o dosya aynen kullanilir, yeniden egitilmez.
OOF dosyalari: reports/faz_a2/oof/{model}__S0s{seed}__f0.csv.gz (fa_common.OOF_COLS sutunlariyla). Faz A klasorune yazilmaz.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
sys.path.insert(0, str(ROOT / "probe"))
import fa_common as fc  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEEDS = (42, 1, 2, 3, 4)
MODEL_SEED0 = 20261104
OUT2 = ROOT / "reports" / "faz_a2"
OOF2 = OUT2 / "oof"
OOF_A = fc.OOF_DIR            # Faz A OOF (salt okunur)


def s0_test_idx(w: pd.DataFrame, seed: int) -> np.ndarray:
    _, te = train_test_split(np.arange(len(w)), test_size=0.2, stratify=w["y_true"].to_numpy(), random_state=seed)
    return np.sort(te)


def s0_masks(w: pd.DataFrame, seed: int):
    te = s0_test_idx(w, seed)
    m = np.zeros(len(w), dtype=bool)
    m[te] = True
    return np.flatnonzero(~m), te


def inner_val(trn: np.ndarray, y: np.ndarray, seed: int):
    tr, va = train_test_split(trn, test_size=0.1, stratify=y[trn], random_state=seed)
    return np.sort(tr), np.sort(va)


def assert_seed42_matches_faz_a(w: pd.DataFrame) -> None:
    te = s0_test_idx(w, 42)
    ref = pd.read_csv(OOF_A.parent / "split_s0.csv")
    assert len(ref) == len(w) and (ref["win"].to_numpy() == w["win"].to_numpy()).all()
    refte = np.flatnonzero(ref["s0_test"].to_numpy() == 1)
    assert np.array_equal(te, refte), "S0 seed 42 split != Faz A split_s0.csv"
    assert np.array_equal(te, np.flatnonzero(w["s0_test"].to_numpy() == 1)), "S0 seed 42 split != windows()['s0_test']"


def oof2_path(model: str, seed: int) -> Path:
    assert model in ("b1", "b2", "cnngru"), model
    return OOF2 / f"{model}__S0s{int(seed)}__f0.csv.gz"


def write_oof2(model: str, seed: int, wins: pd.DataFrame, proba=None, y_pred=None, path: Path | None = None) -> Path:
    n = len(wins)
    if proba is None:
        y_pred = np.asarray(y_pred, dtype=int)
        proba = np.eye(fc.N_CLASSES, dtype=np.float32)[y_pred]
    else:
        proba = np.asarray(proba, dtype=np.float32)
        assert proba.shape == (n, fc.N_CLASSES)
        y_pred = proba.argmax(axis=1) if y_pred is None else np.asarray(y_pred, dtype=int)
    df = pd.DataFrame({"session": wins["session"].to_numpy(), "frame_id": wins["frame_id"].to_numpy(),
                       "t": wins["t"].to_numpy(), "loop_id": wins["loop_id"].to_numpy(), "fold": 0,
                       "y_true": wins["y_true"].to_numpy(), "y_raw": wins["y_raw"].to_numpy(), "y_pred": y_pred})
    for c in range(fc.N_CLASSES):
        df[f"p{c}"] = proba[:, c]
    p = path or oof2_path(model, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    df[fc.OOF_COLS].to_csv(tmp, index=False, compression="gzip", lineterminator="\n")
    os.replace(tmp, p)
    return p

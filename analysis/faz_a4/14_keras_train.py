"""Faz A4 / A4.3(b) - ORIJINAL Keras CNN-GRU (klasor 4 / dataset/notebooks/01_mmwave_cnn_lstm_model.ipynb), OOF tahminleri.

Calisir: <project>/.venv-tf  (Python 3.12, TensorFlow(-cpu) 2.18.0, Keras 3.9.2, CPU). Dogrudan degil,
faz_a4/14_keras_launch.py uzerinden (ana venv'den) baslatilir.

DEFTERDEN BIREBIR ALINANLAR (cell 2/3 - ikisi ayni; cell 4 bos):
  * Mimari: TimeDistributed(Sequential[Conv2D(32,3x3,relu,same) BN MaxPool2 | Conv2D(64,...) BN MaxPool2 | Conv2D(128,...) BN MaxPool2 | Flatten])
            -> GRU(128, dropout=0.3, recurrent_dropout=0.2) -> Dense(6, softmax).  Girdi (5, 48, 48, 2) kanal-son.
            (Port'taki Dense(64)+Dropout(0.4) ve Conv->BN->ReLU sirasi defterde YOK; burada defterdeki gibi: Conv(relu)->BN.)
  * Optimizer: Adam(ExponentialDecay(initial 5e-4, decay_steps 8000, decay_rate 0.92), staircase=False), eps 1e-7 (Keras varsayilani).
  * Loss: sparse_categorical_crossentropy, metric accuracy; batch 32; epochs 120; shuffle=True (fit varsayilani).
  * Sinif agirliklari: compute_class_weight('balanced') -> dict, fit(class_weight=...) (egitim etiketlerinin icinde bulunan siniflar).
  * Callback: EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True).
  * Girdi olceklemesi YOK (defterde yok); varsayilan Glorot/orthogonal baslangiclari (defterde ozel baslangic yok).
DEFTERDEN FARKLI (DESIGN_A4.md'ye gore / zorunlu):
  * Dogrulama: defterde validation_data = TEST kumesi (X_test_seq) - burada KULLANILMAZ; ic dogrulama Faz A kurali
    (S1: egitim gruplarinin (session,loop_id) %10'u, oturum-tabakali, fa.inner_val_split; S0: sinif-tabakali rastgele %10, random_state=42)
    = faz_a/01_state_cnn.py fold_indices'in BIREBIR kopyasi. Sinif agirliklari egitim kismindan (dogrulama ayrildiktan sonra), 01 gibi.
  * ModelCheckpoint(val_accuracy) YOK: defter en iyi val_accuracy'li modeli yukluyordu; DESIGN_A4 yalniz EarlyStopping restore_best (val_loss) ister.
  * Heatmap girdisi: Faz A onbellegi (fa_common.heatmaps(); kanal 1 = hucre basina hiz TOPLAMI, defter hucre ortalamasi) - DESIGN: "ayni heatmap girdileri".
  * Etiket/pencere/kat: fa_common.windows() (y_true = +-3 kare mod; defterin 'refined regional' etiketinden farkli ama Faz A ile ayni).
  * Tohum: keras.utils.set_random_seed(20261104 + 1000*split_id + fold) (01'deki model tohumu kurali; defterde tohum yok).
  * TF is parcacigi: intra 4 / inter 2.
OOF: reports/faz_a4/oof/keras__S1__f{k}.csv.gz, keras__S0__f0.csv.gz (fa_common OOF_COLS sozlesmesi; write_oof 'keras' modelini
     kabul etmedigi icin ayni sutunlarla burada yazilir). Kosu ozeti: reports/faz_a4/14_keras_runs.csv.
Kullanim: python 14_keras_train.py [--runs S1:0,S1:1,S1:2,S1:3,S1:4,S0:0] [--max-epochs N] [--smoke] [--force]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))

import numpy as np
import pandas as pd
import fa_common as fa  # noqa: E402
import tensorflow as tf  # noqa: E402

import os as _os  # thread counts from env (2026-11-05: machine dedicated overnight; see DESIGN_A4 Sapmalar)
tf.config.threading.set_intra_op_parallelism_threads(int(_os.environ.get("TF_INTRA", "4")))
tf.config.threading.set_inter_op_parallelism_threads(int(_os.environ.get("TF_INTER", "2")))
import keras  # noqa: E402
from keras.layers import (BatchNormalization, Conv2D, Dense, Flatten, GRU, Input, MaxPooling2D,  # noqa: E402
                          TimeDistributed)
from keras.models import Model, Sequential  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

MODEL = "keras"
OUT4 = ROOT / "reports" / "faz_a4"
OOF4 = OUT4 / "oof"
RUNS_CSV = OUT4 / "14_keras_runs.csv"
SPLIT_ID = {"S1": 1, "S0": 0}
BATCH, EPOCHS, PATIENCE = 32, 120, 25
SMOKE_DIR = fa.CACHE_DIR / "smoke_a4_14"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- model (defter, cell 2)
def build_model(L: int = fa.L, K: int = fa.K, C: int = fa.NUM_CHANNELS, n_cls: int = fa.N_CLASSES):
    cnn_base = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(K, K, C)), BatchNormalization(), MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"), BatchNormalization(), MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu", padding="same"), BatchNormalization(), MaxPooling2D((2, 2)), Flatten(),
    ], name="cnn_base")
    video_input = Input(shape=(L, K, K, C), name="video_input")
    encoded_frames = TimeDistributed(cnn_base, name="time_distributed_cnn")(video_input)
    encoded_sequence = GRU(128, dropout=0.3, recurrent_dropout=0.2, name="gru_layer")(encoded_frames)
    output_layer = Dense(n_cls, activation="softmax", name="output_layer")(encoded_sequence)
    model = Model(inputs=video_input, outputs=output_layer, name="cnn_lstm_model")
    lr = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0005, decay_steps=8000, decay_rate=0.92)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------------------------------------------------------- folds (faz_a/01_state_cnn.py fold_indices, BIREBIR)
def fold_indices(split: str, fold: int, win: pd.DataFrame):
    w = win
    y = w["y_true"].to_numpy(dtype=np.int64)
    n = len(w)
    allidx = np.arange(n)
    seed = fa.SEED_S1 + 1000 * SPLIT_ID[split] + fold
    if split == "S1":
        test = allidx[w["s1"].to_numpy() == fold]
        trn = allidx[w["s1"].to_numpy() != fold]
    elif split == "S0":
        test = allidx[w["s0_test"].to_numpy() == 1]
        trn = allidx[w["s0_test"].to_numpy() == 0]
    else:
        raise ValueError(split)
    if split == "S0":
        tr, va = train_test_split(trn, test_size=0.1, stratify=y[trn], random_state=fa.SEED_S0)
        tr, va = np.sort(tr), np.sort(va)
    else:
        mask = fa.inner_val_split(w["gid"].to_numpy()[trn], seed, 0.10, strata=w["session_idx"].to_numpy()[trn])
        tr, va = trn[~mask], trn[mask]
    assert len(np.intersect1d(tr, test)) == 0 and len(np.intersect1d(va, test)) == 0
    return tr, va, test, seed


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self):
        self.win = fa.windows()
        self.hm = fa.heatmaps(mmap=True)                     # (N, 2, 48, 48) float32 [frame, ch, y, x]
        self.rows = fa.win_rows(self.win)                    # (Nw, 5)
        self.y = self.win["y_true"].to_numpy(dtype=np.int64)

    def X(self, idx: np.ndarray) -> np.ndarray:
        """(B, 5, 48, 48, 2) float32, channels-last (H = y bins, W = x bins as in the notebook)."""
        r = self.rows[np.asarray(idx)]
        x = np.asarray(self.hm[r.reshape(-1)])               # (B*5, 2, 48, 48)
        x = x.reshape(len(r), fa.L, fa.NUM_CHANNELS, fa.K, fa.K)
        return np.ascontiguousarray(x.transpose(0, 1, 3, 4, 2), dtype=np.float32)


class EpochLog(keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__()
        self.tag, self.best, self.best_ep, self.t0 = tag, float("inf"), 0, time.time()
        self.ep_t = None

    def on_epoch_begin(self, epoch, logs=None):
        self.ep_t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        vl = logs["val_loss"]
        star = vl < self.best
        if star:
            self.best, self.best_ep = vl, epoch + 1
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        log(f"{self.tag} ep {epoch + 1:3d} train_loss={logs['loss']:.4f} train_acc={logs['accuracy']:.4f} val_loss={vl:.4f} "
            f"val_acc={logs['val_accuracy']:.4f} {'*' if star else ' '} best_ep={self.best_ep} lr={lr:.2e} "
            f"{time.time() - self.ep_t:.1f}s (elapsed {time.time() - self.t0:.0f}s)")


def write_oof_keras(split: str, fold: int, wins: pd.DataFrame, proba: np.ndarray, out_dir: Path) -> Path:
    y_pred = proba.argmax(axis=1)
    df = pd.DataFrame({"session": wins["session"].to_numpy(), "frame_id": wins["frame_id"].to_numpy(),
                       "t": wins["t"].to_numpy(), "loop_id": wins["loop_id"].to_numpy(), "fold": int(fold),
                       "y_true": wins["y_true"].to_numpy(), "y_raw": wins["y_raw"].to_numpy(), "y_pred": y_pred})
    for c in range(fa.N_CLASSES):
        df[f"p{c}"] = proba[:, c].astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{MODEL}__{split}__f{int(fold)}.csv.gz"
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    df[fa.OOF_COLS].to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, p)
    return p


def train_run(split: str, fold: int, data: Data, max_epochs: int, smoke_n: int, out_dir: Path, runs_csv: Path):
    tr, va, te, seed = fold_indices(split, fold, data.win)
    if smoke_n:
        rs = np.random.RandomState(0)
        tr, va, te = (rs.choice(a, min(len(a), n), replace=False) for a, n in ((tr, smoke_n), (va, smoke_n // 3), (te, smoke_n // 3)))
    tag = f"[keras {split} f{fold}]"
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = build_model()
    npar = int(model.count_params())

    ytr, yva = data.y[tr], data.y[va]
    present = np.unique(ytr)
    cwv = compute_class_weight("balanced", classes=present, y=ytr)
    cw = {int(c): float(v) for c, v in zip(present, cwv)}
    log(f"{tag} seed={seed} params={npar:,} train={len(tr)} val={len(va)} test={len(te)} "
        f"| train class counts={np.bincount(ytr, minlength=6).tolist()} cw={[round(cw.get(c, 0.0), 2) for c in range(6)]}")
    t_load = time.time()
    Xtr, Xva = data.X(tr), data.X(va)
    log(f"{tag} arrays built Xtr={Xtr.shape} Xva={Xva.shape} in {time.time() - t_load:.1f}s")

    el = EpochLog(tag)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1)
    t_start = time.time()
    hist = model.fit(Xtr, ytr, epochs=max_epochs, batch_size=BATCH, validation_data=(Xva, yva), callbacks=[es, el],
                     class_weight=cw, verbose=0)
    wall = time.time() - t_start
    ep_run = len(hist.history["loss"])
    best_val = float(min(hist.history["val_loss"]))
    best_ep = int(np.argmin(hist.history["val_loss"])) + 1
    # restore_best kontrolu: geri yuklenen agirliklarla dogrulama kaybi == en iyi val_loss olmali
    chk_loss = float(model.evaluate(Xva, yva, batch_size=128, verbose=0)[0])
    log(f"{tag} restore check: val_loss with restored weights={chk_loss:.5f} vs best={best_val:.5f}")
    del Xtr, Xva

    Xte = data.X(te)
    probs = model.predict(Xte, batch_size=128, verbose=0).astype(np.float32)
    del Xte
    w_te = data.win.iloc[te]
    write_oof_keras(split, fold, w_te, probs, out_dir)
    m = fa.state_metrics(data.y[te], probs.argmax(1))
    log(f"{tag} DONE epochs_run={ep_run} best_epoch={best_ep} best_val_loss={best_val:.4f} wall={wall:.0f}s "
        f"| TEST n={m['n']} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f}")
    row = {"model": MODEL, "split": split, "fold": fold, "seed": seed, "epochs_run": ep_run, "best_epoch": best_ep,
           "best_val_loss": round(best_val, 5), "restored_val_loss": round(chk_loss, 5), "wall_s": round(wall, 1),
           "s_per_epoch": round(wall / ep_run, 1), "n_params": npar, "device": "cpu",
           "n_train": len(tr), "n_val": len(va), "n_test": len(te), "test_acc": round(m["acc"], 5),
           "test_macro_f1": round(m["macro_f1"], 5), "max_epochs": max_epochs, "tf": tf.__version__, "keras": keras.__version__}
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(runs_csv, mode="a", header=not runs_csv.exists(), index=False)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="S1:0,S1:1,S1:2,S1:3,S1:4,S0:0")
    ap.add_argument("--max-epochs", type=int, default=EPOCHS)
    ap.add_argument("--smoke", action="store_true", help="kucuk altkume, gercek dosyalara yazmaz (CACHE_DIR/smoke_a4_14)")
    ap.add_argument("--smoke-n", type=int, default=600)
    ap.add_argument("--force", action="store_true", help="OOF varsa da yeniden egit")
    a = ap.parse_args()
    out_dir, runs_csv = (SMOKE_DIR, SMOKE_DIR / "runs.csv") if a.smoke else (OOF4, RUNS_CSV)
    log(f"python {sys.version.split()[0]} tf {tf.__version__} keras {keras.__version__} numpy {np.__version__} "
        f"devices={[d.device_type for d in tf.config.list_physical_devices()]} smoke={a.smoke} max_epochs={a.max_epochs}")
    data = Data()
    log(f"windows={len(data.win)} hm={tuple(data.hm.shape)} out={out_dir}")
    for spec in a.runs.split(","):
        split, f = spec.split(":")
        f = int(f)
        target = out_dir / f"{MODEL}__{split}__f{f}.csv.gz"
        if target.exists() and not (a.force or a.smoke):
            log(f"[keras {split} f{f}] OOF exists -> skip")
            continue
        train_run(split, f, data, a.max_epochs, a.smoke_n if a.smoke else 0, out_dir, runs_csv)
    log("all done")


if __name__ == "__main__":
    main()

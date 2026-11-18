"""Faz A / 05 - ESKI BORU HATTI PROTOKOL ALTINDA (DESIGN.md "OLD (eski boru hatti)").

Durum kaynagi (ToF oracle + varsa cnngru S1 OOF) -> state-change dizisi (predicted_box, prev_box, prev_box_duration_s;
notebook 03_tft_anomaly_v2.ipynb) -> her anomali capraz-uyum kati f icin:
  * TFT (pytorch-forecasting; hedef = kutu numarasi, encoder 24, tahmin 1, hidden 32, 2 head, dropout 0.1, lr 1e-3,
    grad clip 0.1, <= 50 epoch, patience 10) YALNIZ s02 EGITIM dongulerinin gecislerinden egitilir; erken durdurma
    egitim dongulerinin ic bolmesiyle (kal/test'e hic bakilmaz);
  * gecis skoru = |gercek - tahmin| (Denk. 9); sure skoru = prev_box basina Z (egitim-dongusu istatistikleri);
  * donguye max ile toplanir (04'un dongu anahtarlariyla).
Cikti: reports/faz_a/05_old.md, 05_old_cycle_scores.csv.gz (04'un `add_old` sozlesmesi: anahtar + f + old_tft + old_z;
       ayrica takma adlar old_state_score / old_dur_score), 05_old_transitions.csv.gz, 05_old_fits.csv.

04_cycle_anomaly.py `import` EDILMEZ (modul-duzeyinde tum asamayi kosar ve ciktilarini yeniden yazar); bunun yerine dosyanin
AST'sinden yalniz `LAB = source_labels()` satirina kadarki ust-duzey tanimlari (fonksiyonlar, sabitler, kare tablosu) calistirilir:
dongu segmentasyonu / anahtarlari 04 ile BIREBIR ayni koddan gelir.

Kullanim: python faz_a/05_old_pipeline.py [--sources tof,cnngru] [--folds 0,1,2,3,4] [--max_epochs 50] [--seq raw|deb] [--tag ""]
"""
import argparse
import ast
import math
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probe"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import fa_common as fc
from common import md_table

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--sources", default="tof,cnngru")
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--max_epochs", type=int, default=50)
ap.add_argument("--seq", default="raw", choices=["raw", "deb"], help="raw = notebook (per-frame state changes, none dahil); deb = 04'un debounce'u uygulanmis etiketlerden")
ap.add_argument("--tag", default="", help="cikti dosya soneki (duyarlilik kosulari icin)")
ap.add_argument("--normalizer", default="softplus", choices=["softplus", "none"], help="hedef normalizer (notebook: softplus); none = duyarlilik")
ap.add_argument("--inner_frac", type=float, default=0.20)
args = ap.parse_args()
FOLDS = [int(x) for x in args.folds.split(",")]
WANT = args.sources.split(",")
TAG = args.tag
FULL = (FOLDS == list(range(5)) and args.max_epochs == 50)

# ---- TFT sabitleri (notebook)
ENC, PRED = 24, 1
HID, HEADS, DROP, LR, CLIP, PATIENCE = 32, 2, 0.1, 1e-3, 0.1, 10
BATCH = 64
THR_TFT, THR_Z = 0.5, 1.0
SEED = 20261104
PUBLISHED = {  # makale / brifing: gecis duzeyi "% esik ustu" (state, dur, OR); Z gecis sayisi
    "normal": dict(state=9.09, dur=21.27, orr=29.34, nz=221),
    "seq135": dict(state=float("nan"), dur=float("nan"), orr=31.16, nz=180),
    "seq145": dict(state=float("nan"), dur=float("nan"), orr=75.14, nz=156),
    "fast": dict(state=25.29, dur=38.89, orr=float("nan"), nz=198),
    "slow": dict(state=52.05, dur=58.10, orr=float("nan"), nz=315),
}

LINES = []


def out(s=""):
    LINES.append(s)
    print(s, flush=True)


# ------------------------------------------------------------------ 04'un ust-duzey tanimlari (AST ile; main'e kadar)
def load_04():
    p = Path(__file__).resolve().parent / "04_cycle_anomaly.py"
    tree = ast.parse(p.read_text(encoding="utf8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LAB" for t in node.targets):
            break
        body.append(node)
    ns = {"__file__": str(p), "__name__": "mod04"}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(p), "exec"), ns)
    return ns


M4 = load_04()
SESS, SE, build_cycles, debounce_labels, runs_of = M4["SESS"], M4["SE"], M4["build_cycles"], M4["debounce_labels"], M4["runs_of"]
DT = M4["DT"]
print(f"[bilgi] 04 tanimlari yuklendi (modul calistirilmadi). oturumlar={SESS}")
ATYPE_OF = {"session_02": "normal", "session_03": "seq135", "session_04": "seq145", "session_05": "fast", "session_06": "slow"}
T_START_SKIP = 4   # OOF tahminleri her oturumun 5. karesinden baslar (notebook'ta da dizi ilk tahminle baslar)

# ------------------------------------------------------------------ pytorch-forecasting
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

torch.set_num_threads(4)
try:
    import pytorch_optimizer  # noqa: F401
    OPTIM = "ranger"
except Exception:
    OPTIM = "adam"
print(f"[bilgi] torch {torch.__version__}, optimizer = {OPTIM} (ranger = pytorch-forecasting varsayilani; pytorch_optimizer yoksa adam)")


# ------------------------------------------------------------------ state-change dizisi
def state_changes(lab, tt, start):
    """Notebook `extract_state_changes_and_features`: her durum degisiminde bir kayit; prev_box_duration_s = biten durumun suresi
    (degisim zamanlari farki); ilk kayit: prev_box 0, sure 0. `start`: dizinin basladigi kare (ilk tahmin)."""
    lab = np.asarray(lab)[start:]
    t = np.asarray(tt)[start:]
    rows = []
    last_box, last_t = -999, float("nan")
    for i, (b, ti) in enumerate(zip(lab, t)):
        if b != last_box:
            dur = ti - last_t if last_box != -999 else 0.0
            rows.append((start + i, int(b), int(last_box) if last_box != -999 else 0, float(dur)))
            last_box, last_t = int(b), float(ti)
    df = pd.DataFrame(rows, columns=["frame_idx", "predicted_box", "prev_box", "prev_box_duration_s"])
    df["time_idx"] = np.arange(len(df))
    return df


def cycle_membership(C, df):
    """Her gecis (kare indeksi) icin icinde bulundugu dongu satiri (start_idx <= idx < end_idx) -> cycle sutunlari."""
    cid = np.full(len(df), -1, dtype=int)
    for k, c in enumerate(C):
        m = (df["frame_idx"].to_numpy() >= c["start_idx"]) & (df["frame_idx"].to_numpy() < c["end_idx"])
        cid[m] = k
    return cid


# ------------------------------------------------------------------ TFT yardimcilari
def make_df(d):
    d = d.copy()
    d["group_id"] = pd.Series(["series"] * len(d), dtype=object, index=d.index)
    d["prev_box_str"] = d["prev_box"].astype(int).astype(str).astype(object)
    d["predicted_box"] = d["predicted_box"].astype(np.float32)
    d["prev_box_duration_s"] = d["prev_box_duration_s"].astype(np.float32)
    d["time_idx"] = d["time_idx"].astype(np.int64)
    return d[["time_idx", "group_id", "predicted_box", "prev_box_str", "prev_box_duration_s"]].reset_index(drop=True)


def ts_dataset(df, transformation):
    return TimeSeriesDataSet(
        df, time_idx="time_idx", target="predicted_box", group_ids=["group_id"],
        max_encoder_length=ENC, min_encoder_length=ENC, max_prediction_length=PRED, min_prediction_length=PRED,
        static_categoricals=[], static_reals=[], time_varying_known_categoricals=[], time_varying_known_reals=[],
        time_varying_unknown_categoricals=["prev_box_str"], time_varying_unknown_reals=["prev_box_duration_s"],
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation=transformation, center=False),
        categorical_encoders={"prev_box_str": NaNLabelEncoder(add_nan=True), "group_id": NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True, allow_missing_timesteps=True)


def build_train_ds(df_tr):
    """notebook: GroupNormalizer(softplus, center=False). softplus ters-donusumu hedef 0 (none) icin -inf uretebilir;
    sonlu degilse en yakin sadik yedek: transformation=None (center=False), 'port farki' olarak raporlanir."""
    for tr in (("softplus", None) if args.normalizer == "softplus" else (None,)):
        try:
            ds = ts_dataset(df_tr, tr)
            ok = True
            dl = ds.to_dataloader(train=False, batch_size=256, num_workers=0)
            for xb, yb in dl:
                for v in list(xb.values()) + [yb[0]]:
                    if torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all()):
                        ok = False
                if not ok:
                    break
            if ok:
                return ds, tr
            print(f"[uyari] normalizer {tr}: sonlu olmayan deger -> yedege geciliyor")
        except Exception as e:  # noqa
            print(f"[uyari] normalizer {tr}: {type(e).__name__}: {str(e)[:150]} -> yedege geciliyor")
    raise RuntimeError("TimeSeriesDataSet kurulamadi")


def fit_tft(df_all, n_tr, seed, tmpdir):
    """df_all = [egitim satirlari (time_idx < n_tr)] + [ic-dogrulama satirlari]; ic dogrulama hedefleri time_idx >= n_tr
    (encoder baglami egitim satirlarina kadar geri uzanir: gozlenen baglam)."""
    pl.seed_everything(seed, workers=False)
    train_ds, tr = build_train_ds(df_all[df_all["time_idx"] < n_tr])
    n_train_samples = len(train_ds)
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_all, min_prediction_idx=n_tr, stop_randomization=True)
    n_val_samples = len(val_ds)
    info = dict(n_train_rows=int(n_tr), n_val_rows=int(len(df_all) - n_tr), n_train_samples=int(n_train_samples),
                n_val_samples=int(n_val_samples), normalizer=str(tr))
    if n_train_samples < 1 or n_val_samples < 1:
        info.update(status="yetersiz_ornek", epochs=0, best_val=float("nan"))
        return None, train_ds, info
    train_dl = train_ds.to_dataloader(train=True, batch_size=BATCH, num_workers=0, shuffle=True)
    val_dl = val_ds.to_dataloader(train=False, batch_size=BATCH * 2, num_workers=0, shuffle=False)
    es = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=PATIENCE, mode="min")
    ck = ModelCheckpoint(dirpath=tmpdir, filename="tft-best", monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="cpu", devices=1, gradient_clip_val=CLIP, callbacks=[es, ck],
                         logger=False, enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmpdir,
                         enable_checkpointing=True)
    kw = dict(learning_rate=LR, hidden_size=HID, attention_head_size=HEADS, dropout=DROP, hidden_continuous_size=HID // 2,
              output_size=1, loss=QuantileLoss(quantiles=[0.5]), log_interval=-1, reduce_on_plateau_patience=4)
    tft = TemporalFusionTransformer.from_dataset(train_ds, optimizer=OPTIM, **kw)
    trainer.fit(tft, train_dl, val_dl)
    best = ck.best_model_path
    model = TemporalFusionTransformer.load_from_checkpoint(best) if best else tft
    info.update(status="ok", epochs=int(trainer.current_epoch), best_val=float(ck.best_model_score) if ck.best_model_score is not None else float("nan"),
                n_params=int(sum(p.numel() for p in model.parameters())))
    return model, train_ds, info


def predict_series(model, train_ds, df):
    """df (make_df) -> {time_idx: tahmin}; time_idx < ENC icin tahmin yok (notebook ile ayni)."""
    if len(df) <= ENC:
        return {}
    ds = TimeSeriesDataSet.from_dataset(train_ds, df, predict=False, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=BATCH * 2, num_workers=0, shuffle=False)
    res = model.predict(dl, mode="prediction", return_x=True, return_index=True,
                        trainer_kwargs=dict(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                                            enable_model_summary=False, enable_checkpointing=False))
    pred = res.output.detach().cpu().numpy().reshape(len(ds), -1)[:, 0]
    tix = res.x["decoder_time_idx"][:, 0].detach().cpu().numpy()
    return dict(zip(tix.astype(int).tolist(), pred.tolist()))


def duration_stats(tr_df):
    """compute_duration_table.py ile ayni: oturum ilk kaydi (time_idx==0 & sure==0) atilir; prev_box basina mean/std(ddof=1),
    std >= 1e-6, count >= 3."""
    d = tr_df[~((tr_df["time_idx"] == 0) & (tr_df["prev_box_duration_s"] == 0.0))]
    g = d.groupby("prev_box")["prev_box_duration_s"].agg(mean_duration="mean", std_duration="std", count="count").reset_index()
    g["std_duration"] = np.maximum(g["std_duration"].fillna(1e-6), 1e-6)
    return g[g["count"] >= 3].set_index("prev_box")


def zscores(df, st):
    z = np.full(len(df), np.nan)
    for i, (pb, du) in enumerate(zip(df["prev_box"].to_numpy(), df["prev_box_duration_s"].to_numpy())):
        if pb in st.index and not (i == 0):   # oturumun ilk kaydi (sure 0) skorlanmaz
            z[i] = (du - st.loc[pb, "mean_duration"]) / st.loc[pb, "std_duration"]
    return z


# ------------------------------------------------------------------ durum kaynaklari
LAB_ALL = M4["source_labels"]()
SOURCES = [s for s in WANT if s in LAB_ALL]
for s in WANT:
    if s not in LAB_ALL:
        print(f"[uyari] kaynak {s}: etiket yok / OOF eksik -> atlandi")
if not SOURCES:
    print("[hata] hic kaynak yok")
    sys.exit(1)
print(f"[bilgi] kaynaklar: {SOURCES}; seq = {args.seq}; folds = {FOLDS}; max_epochs = {args.max_epochs}")

SRCN = {"tof": "ToF oracle", "cnngru": "CNN-GRU", "cnnonly": "CNN-only", "b1": "B1", "b2": "B2"}
fit_rows, trans_all, cyc_all, z_tabs = [], [], [], []
t00 = time.time()
for src in SOURCES:
    labs = LAB_ALL[src]
    # kaynak/oturum basina state-change dizisi (segmentasyondan bagimsiz)
    SEQ = {}
    for s in SESS:
        lab = labs[s]
        if args.seq == "deb":
            lab = debounce_labels(lab)
        SEQ[s] = state_changes(lab, SE[s]["t"].to_numpy(), 0 if src == "tof" else T_START_SKIP)
    print(f"[bilgi] {src}: gecis sayilari = " + ", ".join(f"{s.split('_')[1]}:{len(SEQ[s])}" for s in SESS))
    for seg in ("pred", "tof"):
        C, _ = build_cycles(src, seg, labs)
        Cl = C.to_dict("records")
        # gecis -> dongu
        memb = {}
        for s in SESS:
            Cs = [c for c in Cl if c["session"] == s]
            idxs = [i for i, c in enumerate(Cl) if c["session"] == s]
            m = cycle_membership(Cs, SEQ[s])
            memb[s] = np.where(m >= 0, np.array(idxs + [-1])[m], -1)
        for f in FOLDS:
            ro = fc.anom_roles(f)
            # --- egitim dizisi: s02 egitim donguleri (zaman sirasi) + ic dogrulama donguleri (sonda)
            s02 = "session_02_normal_seq125" if "session_02_normal_seq125" in SESS else [s for s in SESS if s.startswith("session_02")][0]
            d02 = SEQ[s02].copy()
            d02["cyc"] = memb[s02]
            tr_c = [i for i, c in enumerate(Cl) if c["session"] == s02 and c["atype"] == "normal" and c["fold"] in ro["train"]]
            cal_c = [i for i, c in enumerate(Cl) if c["session"] == s02 and c["atype"] == "normal" and c["fold"] in ro["cal"]]
            te_c = [i for i, c in enumerate(Cl) if c["session"] == s02 and c["atype"] == "normal" and c["fold"] == f]
            dtr = d02[d02["cyc"].isin(tr_c)].copy()
            vm = fc.inner_val_split(dtr["cyc"].to_numpy(), seed=SEED + 100 * f + (1 if seg == "tof" else 2), frac=args.inner_frac)
            dfit = pd.concat([dtr[~vm], dtr[vm]], ignore_index=True)
            n_tr = int((~vm).sum())
            # Z istatistikleri: TUM egitim-dongusu gecisleri (ic dogrulama dahil; Z-score egitim istatistigidir). time_idx = s02 oturum sirasi
            st = duration_stats(dtr)
            dfit_t = dfit.copy()
            dfit_t["time_idx"] = np.arange(len(dfit_t))
            tmp = tempfile.mkdtemp(prefix="tft05_")
            t0 = time.time()
            model, train_ds, info = fit_tft(make_df(dfit_t), n_tr, SEED + 10 * f, tmp)
            info.update(state_source=src, segmentation=seg, f=f, n_train_cycles=len(tr_c), n_inner_val_cycles=int(dtr.loc[vm, "cyc"].nunique()),
                        secs=round(time.time() - t0, 1), z_prev_boxes=",".join(map(str, st.index.tolist())),
                        z_counts=",".join(map(str, st["count"].tolist())))
            fit_rows.append(info)
            print(f"[fit] {src}/{seg}/f{f}: " + ", ".join(f"{k}={v}" for k, v in info.items() if k not in ("z_prev_boxes", "z_counts")), flush=True)
            zt = st.reset_index().assign(state_source=src, segmentation=seg, f=f)
            z_tabs.append(zt)
            # --- skorlama: her oturum tam dizisi
            for s in SESS:
                d = SEQ[s].copy()
                d["cyc"] = memb[s]
                tft_sc = np.full(len(d), np.nan)
                pred_v = np.full(len(d), np.nan)
                if model is not None:
                    pr = predict_series(model, train_ds, make_df(d))
                    for ti, pv in pr.items():
                        pred_v[ti] = pv
                    tft_sc = np.abs(d["predicted_box"].to_numpy() - pred_v)
                d["tft_pred"] = pred_v
                d["tft_score"] = tft_sc
                d["z"] = zscores(d, st)
                d["session"] = s
                d["state_source"], d["segmentation"], d["f"] = src, seg, f
                rl = np.full(len(d), "", dtype=object)
                at = np.full(len(d), "", dtype=object)
                cf = np.full(len(d), -1)
                for k in np.unique(d["cyc"][d["cyc"] >= 0]):
                    c = Cl[k]
                    mk = d["cyc"].to_numpy() == k
                    at[mk] = c["atype"]
                    cf[mk] = c["fold"]
                    if c["atype"] == "normal":
                        rl[mk] = "train" if k in tr_c else ("cal" if k in cal_c else ("test" if k in te_c else "?"))
                    else:
                        rl[mk] = "test"
                d["role"], d["atype"], d["cyc_fold"] = rl, at, cf
                trans_all.append(d)
            # --- dongu skorlari (cal + test)
            dd = pd.concat([x for x in trans_all[-len(SESS):]], ignore_index=True)
            for k, c in enumerate(Cl):
                is_norm = c["atype"] == "normal"
                if is_norm:
                    role = "train" if k in tr_c else ("cal" if k in cal_c else ("test" if k in te_c else "?"))
                    if role not in ("cal", "test"):
                        continue
                else:
                    role = "test"
                g = dd[(dd["cyc"] == k) & (dd["session"] == c["session"])]
                ot = g["tft_score"].dropna()
                oz = g["z"].dropna().abs()
                cyc_all.append(dict(state_source=src, segmentation=seg, session=c["session"], cycle_idx=c["cycle_idx"],
                                    start_frame_id=c["start_frame_id"], f=f, role=role, atype=c["atype"],
                                    old_tft=(float(ot.max()) if len(ot) else float("nan")),
                                    old_z=(float(oz.max()) if len(oz) else float("nan")),
                                    n_trans=len(g), n_tft=len(ot), n_z=len(oz)))
    print(f"[bilgi] {src} bitti; gecen sure {time.time() - t00:.0f} s", flush=True)

FIT = pd.DataFrame(fit_rows)
TR = pd.concat(trans_all, ignore_index=True)
CY = pd.DataFrame(cyc_all)
ZT = pd.concat(z_tabs, ignore_index=True)
CY["old_state_score"] = CY["old_tft"]
CY["old_dur_score"] = CY["old_z"]
# 04 sozlesmesi: anahtar + f basina TEK satir
KEY = ["state_source", "segmentation", "session", "cycle_idx", "start_frame_id", "f"]
assert not CY.duplicated(KEY).any(), "dongu anahtari tekrar ediyor"
if FULL:
    CY.to_csv(fc.OUT / f"05_old_cycle_scores{TAG}.csv.gz", index=False, compression="gzip")
    TR.drop(columns=["tft_pred"]).to_csv(fc.OUT / f"05_old_transitions{TAG}.csv.gz", index=False, compression="gzip")
    FIT.to_csv(fc.OUT / f"05_old_fits{TAG}.csv", index=False)
else:
    print("[bilgi] kismi/duman kosusu: cikti dosyalari YAZILMADI (yalniz stdout)")


# ------------------------------------------------------------------ analiz
def pct(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return 100.0 * float((x > 0).mean()) if len(x) else float("nan")


def rate_stats(g):
    """g: gecis satirlari -> (n_tft, %state>0.5, n_z, %|Z|>1, n_or, %OR)."""
    a = g["tft_score"].to_numpy(float)
    z = np.abs(g["z"].to_numpy(float))
    ma, mz = ~np.isnan(a), ~np.isnan(z)
    mo = ma & mz
    return dict(n_tft=int(ma.sum()), pct_state=100 * float((a[ma] > THR_TFT).mean()) if ma.any() else float("nan"),
                n_z=int(mz.sum()), pct_dur=100 * float((z[mz] > THR_Z).mean()) if mz.any() else float("nan"),
                n_or=int(mo.sum()), pct_or=100 * float(((a[mo] > THR_TFT) | (z[mo] > THR_Z)).mean()) if mo.any() else float("nan"))


def msd(v, d=2):
    v = np.asarray([x for x in v if not np.isnan(x)], float)
    if not len(v):
        return "nan"
    return f"{v.mean():.{d}f}" + (f" ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else "")


out(f"# 05 — Eski boru hatti (CNN-GRU/ToF -> TFT |Δ| + Z-score) protokol altinda (Faz A)")
out()
out("`faz_a/05_old_pipeline.py` tarafindan uretilir; DESIGN.md 'OLD (eski boru hatti)'. Tum sayilar stdout'a da yazilir.")
out()
out("## Uygulama ve protokol")
out(f"- Durum kaynaklari: {', '.join(SRCN[s] for s in SOURCES)}" + (f" (istenip OOF'u/etiketi olmayan atlandi: {', '.join(SRCN[s] for s in WANT if s not in SOURCES)})" if len(SOURCES) < len(WANT) else "") + ".")
out(f"- **State-change dizisi** (notebook `extract_state_changes_and_features`): oturum boyunca kare etiketindeki her degisim bir kayit (none = 0 dahil); `prev_box_duration_s` = biten durumun suresi; ilk kayit prev_box = 0, sure = 0. Dizi turu = `{args.seq}` ({'debounce UYGULANMAMIS ham kare etiketleri (notebook ile ayni)' if args.seq == 'raw' else '04 debounce uygulanmis etiketlerden'}); tahmin kaynaklarinda dizi 5. kareden baslar (OOF penceresi), ToF'ta 1. kareden.")
out("- **Dongu eslemesi:** `04_cycle_anomaly.py` fonksiyonlari (`build_cycles`, `debounce_labels`; AST ile yuklendi, 04 kosmaz) ile **ayni** dongu sinirlari/anahtarlari; bir gecis, yeni durumunun basladigi karenin icinde bulundugu donguye aittir. Dongu skoru = icindeki gecislerin max'i (04 anahtari: `(state_source, segmentation, session, cycle_idx, start_frame_id)` + `f`).")
out("- **TFT:** pytorch-forecasting " + __import__("pytorch_forecasting").__version__ + f"; hedef = kutu numarasi (float), encoder {ENC}, tahmin {PRED}, hidden {HID}, {HEADS} head, dropout {DROP}, lr {LR}, grad clip {CLIP}, max {args.max_epochs} epoch, early-stop patience {PATIENCE} (min_delta 1e-5), tek nicel (0.5) `QuantileLoss`, `reduce_on_plateau_patience=4`, batch {BATCH}; unknown categorical = `prev_box`, unknown real = `prev_box_duration_s`; `GroupNormalizer(softplus, center=False)` (sonlu degilse yedek: `transformation=None`; kullanilan asagida), `add_relative_time_idx/target_scales/encoder_length`. Optimizer: **{OPTIM}**" + (" (pytorch-forecasting varsayilani `ranger` icin `pytorch_optimizer` kurulu degil; Adam notebook'un `ranger`'indan **port farkidir**)." if OPTIM == "adam" else " (varsayilan)."))
out("- **Egitim verisi (her f):** s02 **egitim** dongulerinin gecisleri zaman sirasiyla birlestirilir (egitim katlari = `anom_roles(f)['train']`, 2 kat ≈ 20 dongu); bunlarin `inner_frac`" + f" = {args.inner_frac:.2f}'si (dongu bazli `fa_common.inner_val_split`) **sona** eklenip ic dogrulama hedefleri olur (encoder baglami egitim satirlarina geri uzanir; kal/test hicbir yerde yok). Erken durdurma + checkpoint secimi yalniz bu ic dogrulama kaybiyla. Tum dizilerde **tek sabit `group_id`** (notebook'taki oturum-basina grup kimlikleri `GroupNormalizer` icin bilinmeyen grup sorunu yaratir).")
out(f"- **Cikarim:** her oturumun TAM gecis dizisi (gozlenen baglam; komsu katlarin donguleri encoder'a girebilir — DESIGN: etiket sizintisi degil). `time_idx < {ENC}` olan gecislere TFT skoru verilemez (notebook ile ayni): bu gecisler dongu skoruna girmez; hepsi bu gecislerden olusan donguler `old_tft = NaN` alir.")
out("- **Sure Z-skoru** (`compute_duration_table.py` ile ayni mantik): `prev_box` basina mean/std (ddof 1, std >= 1e-6, count >= 3) yalniz **egitim-dongusu** gecislerinden (ic dogrulama donguleri dahil); oturum ilk kaydi skorlanmaz; istatistigi olmayan `prev_box` -> NaN. Dongu skoru = max |Z|.")
out(f"- Tohumlar: bolme {SEED}; TFT `seed_everything({SEED} + 10f)`; ic bolme `{SEED} + 100f + (1|2)`.")
out()

# --- fit tablosu
out("## TFT egitim ozeti (her segmentasyon × f; veri-acligi)")
ft_ = FIT.copy()
ft_["sure_s"] = ft_["secs"]
show = ft_[["state_source", "segmentation", "f", "n_train_cycles", "n_inner_val_cycles", "n_train_rows", "n_val_rows", "n_train_samples", "n_val_samples", "epochs", "best_val", "normalizer", "status", "sure_s"]]
out(md_table(show, "{:.4f}"))
nm = FIT["normalizer"].value_counts().to_dict()
out(f"\nKullanilan normalizer: {nm}. Egitim ornek sayisi: min {FIT['n_train_samples'].min()}, medyan {FIT['n_train_samples'].median():.0f}, maks {FIT['n_train_samples'].max()}; ic-dogrulama ornegi medyan {FIT['n_val_samples'].median():.0f}. Epoch (erken durdurma) medyan {FIT['epochs'].median():.0f}/{args.max_epochs}.")
out()
out("### Z istatistik referanslari (egitim-dongusu gecisleri; prev_box:count)")
zs = ZT.groupby(["state_source", "segmentation", "prev_box"])["count"].agg(["min", "max"]).reset_index()
out(md_table(zs.rename(columns={"min": "count_min_over_f", "max": "count_max_over_f"}), "{:.0f}"))
out()

# --- TFT trivial taban: TFT veri-acligi kaniti
out("## TFT trivial tahminciden daha iyi mi? (veri-acligi tanisi)")
out("Test-dongusu gecislerinde (s02 normal, havuzlanmis; TFT skoru olan) TFT MAE'si ile iki trivial tahminci: (a) egitim ortalamasi, (b) **Markov-modu**: egitim-dongusu gecislerinde `prev_box` icin en sik yeni durum. MAE = ortalama |gercek − tahmin| (Denk. 9).")
def trivial_table(TRx):
    rows = []
    for (src, seg), g in TRx.groupby(["state_source", "segmentation"]):
        mae_t, mae_m, mae_mk, n = [], [], [], 0
        gg = g[(g["session"].str.startswith("session_02")) & (g["role"] == "test") & g["tft_score"].notna()]
        for f, gf in gg.groupby("f"):
            trn = g[(g["session"].str.startswith("session_02")) & (g["f"] == f) & (g["role"] == "train")]
            mu = trn["predicted_box"].mean()
            mode = trn.groupby("prev_box")["predicted_box"].agg(lambda x: x.value_counts().idxmax())
            pm = gf["prev_box"].map(mode).fillna(mu)
            mae_t.append(gf["tft_score"].mean())
            mae_m.append((gf["predicted_box"] - mu).abs().mean())
            mae_mk.append((gf["predicted_box"] - pm).abs().mean())
            n += len(gf)
        if n:
            rows.append(dict(kaynak=SRCN[src], seg=seg, n_gecis=n, TFT_MAE=float(np.mean(mae_t)), egitim_ortalamasi_MAE=float(np.mean(mae_m)),
                             Markov_modu_MAE=float(np.mean(mae_mk))))
    return pd.DataFrame(rows)


RT = trivial_table(TR)
if len(RT):
    out(md_table(RT, "{:.3f}"))
out()

# --- gecis duzeyi % esik ustu (makale metrigi)
out("## Gecis duzeyinde '% esik ustu' (eski makale metrigi; esikler sabit: |Δ| > 0.5, |Z| > 1.0)")
out("Normal: s02 **test** dongulerinin gecisleri, 5 f havuzlanmis (her normal gecis, kendi donguyu hic gormeyen bir modelle tam bir kez skorlanir). Anomali oturumlari: her f'nin kendi TFT/istatistigiyle oturumdaki **tum** gecisler; 5 f ortalamasi ± sd. `n` = skoru olan gecis sayisi (TFT ilk 24 gecis icin skor veremez). Yayinlanmis degerler sizintili CNN-GRU tahmin dizisi + normal veride egitim=dogrulama=test ile elde edilmisti; **ToF-oracle satiri ust sinirdir, birebir kiyas degildir**.")
tab = []
pub_scen = ["normal", "seq135", "seq145", "fast", "slow"]
S_OF = {v: k for k, v in {"session_02": "normal", "session_03": "seq135", "session_04": "seq145", "session_05": "fast", "session_06": "slow"}.items()}
for src in SOURCES:
    for seg in ("tof", "pred"):
        for sc in pub_scen:
            sess = [s for s in SESS if s.startswith(S_OF[sc])][0]
            g = TR[(TR["state_source"] == src) & (TR["segmentation"] == seg) & (TR["session"] == sess)]
            if sc == "normal":
                gt = g[g["role"] == "test"]
                r = rate_stats(gt)
                r.update(f_sd="")
                vals = {k: r[k] for k in r}
                tab.append(dict(kaynak=SRCN[src], seg=seg, senaryo=sc, n_tft=r["n_tft"], pct_state=f"{r['pct_state']:.2f}", n_z=r["n_z"],
                                pct_dur=f"{r['pct_dur']:.2f}", n_or=r["n_or"], pct_or=f"{r['pct_or']:.2f}",
                                yayinlanan=f"{PUBLISHED[sc]['state']}/{PUBLISHED[sc]['dur']}/{PUBLISHED[sc]['orr']}"))
            else:
                rs = [rate_stats(gf) for _, gf in g.groupby("f")]
                tab.append(dict(kaynak=SRCN[src], seg=seg, senaryo=sc,
                                n_tft=f"{np.mean([r['n_tft'] for r in rs]):.0f}", pct_state=msd([r['pct_state'] for r in rs]),
                                n_z=f"{np.mean([r['n_z'] for r in rs]):.0f}", pct_dur=msd([r['pct_dur'] for r in rs]),
                                n_or=f"{np.mean([r['n_or'] for r in rs]):.0f}", pct_or=msd([r['pct_or'] for r in rs]),
                                yayinlanan=f"{PUBLISHED[sc]['state']}/{PUBLISHED[sc]['dur']}/{PUBLISHED[sc]['orr']}"))
TT = pd.DataFrame(tab)
TT = TT.rename(columns={"pct_state": "%state>0.5", "pct_dur": "%|Z|>1", "pct_or": "%OR", "yayinlanan": "yayinlanan state/dur/OR (nan = makalede yok)"})
out(md_table(TT))
out()
out("Okuma notu: normal satirlarindaki `%OR`, yayinlanan **29.3**'un egitim-disi (test dongusu) karsiligidir. " + "; ".join(
    f"{r['kaynak']}/{r['seg']}: state {r['%state>0.5']} %, |Z|>1 {r['%|Z|>1']} %, OR {r['%OR']} %" for _, r in TT[TT['senaryo'] == 'normal'].iterrows()) + ". (Tanim geregi normal dagilimda |Z| > 1 ~%32 isaretler.)")
out()

# --- dongu duzeyi OLD skorlari (04'e girdi)
out("## Dongu duzeyi OLD skorlari (04'e girdi; cal + test rolleri)")
out("Dongu skoru = icindeki gecislerin max'i. Asagida yalniz 05'in kendi ozeti (tam AUROC/AUPRC/FAR/tespit tablosu `04_anomaly.md`'de, conformal p-degerleriyle).")
cs = []
for (src, seg), g in CY.groupby(["state_source", "segmentation"]):
    te = g[g["role"] == "test"]
    for col, nm_ in (("old_tft", "OLD TFT"), ("old_z", "OLD Z")):
        row = dict(kaynak=SRCN[src], seg=seg, skor=nm_)
        nn = te[te["atype"] == "normal"][col]
        row["normal n"] = int(nn.notna().sum()) ; row["normal NaN"] = int(nn.isna().sum()); row["normal medyan"] = float(nn.median())
        for at in ("seq135", "seq145", "fast", "slow"):
            aa = te[te["atype"] == at][col]
            row[f"{at} medyan"] = float(aa.median())
        # AUROC havuzlanmis (kat-ici degil; ozet): normal test vs her anomali, f ortalamasi
        aus = []
        for at in ("seq135", "seq145", "fast", "slow"):
            v = []
            for f, gf in te.groupby("f"):
                n_ = gf[(gf["atype"] == "normal")][col].dropna().to_numpy()
                p_ = gf[gf["atype"] == at][col].dropna().to_numpy()
                if len(n_) > 1 and len(p_) > 0:
                    v.append(roc_auc_score(np.r_[np.zeros(len(n_)), np.ones(len(p_))], np.r_[n_, p_]))
            aus.append(float(np.mean(v)) if v else float("nan"))
        for at, a in zip(("seq135", "seq145", "fast", "slow"), aus):
            row[f"AUROC {at}"] = a
        cs.append(row)
CS = pd.DataFrame(cs)
out(md_table(CS, "{:.3f}"))
out()
out("(AUROC burada kat basina hesaplanip ortalanmistir; resmi sonuc `04_anomaly.md`.)")
out()
out("## Yorum (sayilardan)")
if len(RT):
    for _, r in RT.iterrows():
        out(f"- {r['kaynak']}/{r['seg']}: TFT MAE = {r['TFT_MAE']:.3f}; egitim-ortalamasi tahmincisi {r['egitim_ortalamasi_MAE']:.3f}; Markov-modu tahmincisi {r['Markov_modu_MAE']:.3f} -> TFT Markov-modunu "
            + ("**geciyor**" if r['TFT_MAE'] < r['Markov_modu_MAE'] else "**gecemiyor**") + f" (n = {r['n_gecis']} gecis).")
out(f"- Egitim ornegi sayisi {FIT['n_train_samples'].min()}-{FIT['n_train_samples'].max()} (DESIGN'in ~36 tahmini 3 gecis/dongu varsayiyordu; none durumu dahil ham dizide ~6 gecis/dongu oldugundan daha fazla, ama yine de ~70 ornek: TFT icin veri-acligi vardir). Erken durdurma epoch medyani {FIT['epochs'].median():.0f}/{args.max_epochs}.")
out("- TFT MAE'si Markov-modundan kotuyse 'state' kanali TFT'nin ogrendigi bir seyden degil, dizideki kutu-numarasi farkindan gelir; sira bilgisi D1 (Markov −log p) ile aynidir.")
out("- Sabit esikler (0.5 / 1.0) conformal esikle degil, makaledeki gibi uygulandi; conformal karsiligi `04_anomaly.md` OLD satirlarindadir.")
if TAG == "":
    out()
    out("## Duyarlilik kosulari (ayni protokol; ayri dosyalar; asagidaki sayilar o kosularin ciktilarindan okunur)")
    for tg, desc in (("_none", "`--normalizer none` (softplus hedef donusumu yok; TFT'nin cokmesi donusumden mi?)"),
                     ("_deb", "`--seq deb` (state-change dizisi 04'un debounce'lu etiketlerinden; ham kare-flip gurultusu temizlenmis)")):
        pf, pt = fc.OUT / f"05_old_fits{tg}.csv", fc.OUT / f"05_old_transitions{tg}.csv.gz"
        if not (pf.exists() and pt.exists()):
            out(f"- {desc}: dosya yok (kosulmadi).")
            continue
        F2 = pd.read_csv(pf)
        T2 = pd.read_csv(pt)
        R2 = trivial_table(T2)
        out(f"- **{desc}**: egitim ornegi {F2['n_train_samples'].min()}-{F2['n_train_samples'].max()}, epoch medyan {F2['epochs'].median():.0f}, ic-val kaybi medyan {F2['best_val'].median():.3f} (min {F2['best_val'].min():.3f}, maks {F2['best_val'].max():.3f}).")
        for _, r in R2.iterrows():
            out(f"  - {r['kaynak']}/{r['seg']}: TFT MAE {r['TFT_MAE']:.3f} | egitim-ortalamasi {r['egitim_ortalamasi_MAE']:.3f} | Markov-modu {r['Markov_modu_MAE']:.3f} (n = {r['n_gecis']}).")
        gn = T2[(T2["segmentation"] == "tof") & T2["session"].str.startswith("session_02") & (T2["role"] == "test")]
        for src, gs in gn.groupby("state_source"):
            r_ = rate_stats(gs)
            out(f"  - {SRCN[src]}/tof normal test: %state>0.5 = {r_['pct_state']:.2f}, %|Z|>1 = {r_['pct_dur']:.2f}, %OR = {r_['pct_or']:.2f} (n_or = {r_['n_or']}).")
out(f"- Toplam sure {time.time() - t00:.0f} s.")

if FULL:
    (fc.OUT / f"05_old{TAG}.md").write_text("\n".join(LINES), encoding="utf8")
    print(f"[bilgi] yazildi: {fc.OUT / f'05_old{TAG}.md'}")

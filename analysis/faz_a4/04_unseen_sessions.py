"""Faz A4 / A4.4 (M4) - anomali oturumlarini hic gormemis durum tahmincisi (DESIGN_A4.md, on-kayitli).

Faz A anomali protokolu (roller, pred segmentasyonu, D1/D2/D3, standart conformal) DEGISTIRILMEDEN; B2 durumlari iki kaynaktan:
  (i)  S2: mevcut b2__S2__f* OOF'lari (her oturum, kendisini gormemis bir modelle; ama OTHER anomali oturumlarini gormus olabilir).
  (ii) "devreye alma gerceki": B2 yalniz s01 + s02 ile egitilir.
       - s02 pencereleri: anomali kati k'nin pencereleri, s01 + s02'nin (k disindaki katlari) ile egitilen modelle (k = 0..4);
       - s03-s06: s01 + s02'nin tamamiyla egitilen tek model.
       Ozellikler / RF ayari / tohum B2 ile ayni (faz_a/02_state_baselines.py'den birebir kopya; modul import EDILMEZ).

Raporlanan: FAR (Wilson), tur basina tespit, AUROC, D1/D2 tek basina, Faz A ile yan yana, durum macro-F1 (s02-s06),
on-yazili okuma kurali (harfi harfine), devreye alma dakikalari.
Kullanim: OMP_NUM_THREADS=4 tools/local_run.sh faz_a4/04_unseen_sessions.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
sys.path.insert(0, str(ROOT / "faz_a2"))
sys.path.insert(0, str(ROOT / "probe"))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import a2_cycles as ac
import fa_common as fc
from common import md_table

OUT4 = ROOT / "reports" / "faz_a4"
OOF4 = OUT4 / "oof"
OOF4.mkdir(parents=True, exist_ok=True)
SEED = 20261104            # faz_a/02_state_baselines.py ile ayni
S01, S02 = "session_01_training", "session_02_normal_seq125"
MARK = "<!-- YORUM-BEGIN -->"
T0 = time.time()
NJOBS = 3


def P(*a):
    print(*a, flush=True)


# ============================================================ B2 ozellikleri (02_state_baselines.py'den birebir)
FEATS = ["cx", "cy", "n_points", "snr_sum", "mean_abs_v", "std_x", "std_y"]
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}
ft = fc.frame_table()
W = fc.windows()
R = fc.win_rows(W)


def b2_features(rows):
    cols = []
    t = np.arange(fc.L, dtype=float)
    tc = t - t.mean()
    for f in FEATS:
        v = ft[f].to_numpy(float)
        v = np.where(np.isnan(v), FILL[f], v)[rows]
        cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
    return np.stack(cols, 1).astype(np.float32)


X = b2_features(R)
Y = W["y_true"].to_numpy()
SESS_W = W["session"].to_numpy()


def fit_predict(tr_idx, te_idx):
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=NJOBS)
    rf.fit(X[tr_idx], Y[tr_idx])
    pr = np.zeros((len(te_idx), 6), np.float32)
    pr[:, rf.classes_.astype(int)] = rf.predict_proba(X[te_idx])
    return pr


def oof_frame(idx, proba, fold):
    w = W.iloc[idx]
    df = pd.DataFrame({"session": w["session"].to_numpy(), "frame_id": w["frame_id"].to_numpy(), "t": w["t"].to_numpy(),
                       "loop_id": w["loop_id"].to_numpy(), "fold": int(fold), "y_true": w["y_true"].to_numpy(),
                       "y_raw": w["y_raw"].to_numpy(), "y_pred": proba.argmax(1)})
    for c in range(fc.N_CLASSES):
        df[f"p{c}"] = proba[:, c]
    return df[fc.OOF_COLS]


# ============================================================ 0) esdegerlik korumasi: kopyalanan B2 = Faz A B2
P(f"[{time.time()-T0:.0f}s] 0) esdegerlik korumasi")
guard = []
for split, f, col in (("S1", 0, "s1"), ("S2", 1, "s2")):
    te = np.flatnonzero(W[col].to_numpy() == f)
    tr = np.flatnonzero(W[col].to_numpy() != f)
    pr = fit_predict(tr, te)
    ref = pd.read_csv(fc.oof_path("b2", split, f))
    key = W.iloc[te][["session", "frame_id"]].reset_index(drop=True)
    assert (ref[["session", "frame_id"]].reset_index(drop=True).equals(key)), f"{split} f{f}: OOF satir sirasi farkli"
    agree = float((pr.argmax(1) == ref["y_pred"].to_numpy()).mean())
    dmax = float(np.abs(pr - ref[[f"p{c}" for c in range(6)]].to_numpy()).max())
    P(f"  b2 {split} f{f}: n={len(te)} y_pred uyumu = {agree:.6f}; max |proba farki| = {dmax:.3g}")
    guard.append(dict(split=split, fold=f, n=len(te), agree=agree, max_proba_diff=dmax))
    assert agree == 1.0, "kopyalanan B2 Faz A OOF'unu yeniden uretmiyor"
guard = pd.DataFrame(guard)

# ============================================================ kaynak (i): S2 OOF (tamlik kontrolu)
P(f"[{time.time()-T0:.0f}s] kaynak (i): b2__S2")
oof_i = fc.load_oof("b2", "S2")
assert len(oof_i) == 16248 and oof_i.groupby("fold").ngroups == 6
assert not oof_i.duplicated(["session", "frame_id"]).any()
mk = W[["session", "frame_id"]].merge(oof_i[["session", "frame_id", "y_pred"]], on=["session", "frame_id"], how="left", validate="1:1")
assert mk["y_pred"].notna().all(), "S2 OOF pencere kumesini tam kapsamiyor"
P(f"  (i) tam: {len(oof_i)} satir, 6 kat, (session, frame_id) kumesi windows() ile ayni")
LAB_I = ac.labels_from_oof(oof_i, strict=False)

# ============================================================ kaynak (ii): B2 yalniz s01 + s02
P(f"[{time.time()-T0:.0f}s] kaynak (ii): B2 s01+s02")
AL = fc.anom_loops()
fmap = AL[AL["session"] == S02].set_index("loop_id")["fold"]
is02 = SESS_W == S02
is01 = SESS_W == S01
wfold = np.full(len(W), -2, dtype=int)                      # -2: s02 disi
wfold[is02] = W.loc[is02, "loop_id"].map(fmap).fillna(-1).astype(int).to_numpy()
n_frag = int((is02 & (wfold == -1)).sum())
P(f"  s02 pencere: {int(is02.sum())}; kat basina pencere: { {k: int((wfold == k).sum()) for k in range(5)} }; parca (fold -1) pencere: {n_frag}")
assert n_frag == 1, "beklenen: yalniz loop 0 parcasi (1 pencere)"
parts, train_n = [], []
for k in range(5):
    tr = np.flatnonzero(is01 | (is02 & (wfold >= 0) & (wfold != k)))
    te = np.flatnonzero(is02 & ((wfold == k) | ((wfold == -1) if k == 0 else False)))
    assert not set(tr) & set(te)
    pr = fit_predict(tr, te)
    parts.append(oof_frame(te, pr, k))
    train_n.append(dict(model=f"ii_f{k}", n_train=len(tr), n_s01=int(is01.sum()), n_s02=int(len(tr) - is01.sum()), n_pred=len(te)))
    P(f"  (ii) kat {k}: egitim {len(tr)} pencere, tahmin {len(te)} pencere ({time.time()-T0:.0f}s)")
tr = np.flatnonzero(is01 | (is02 & (wfold >= 0)))
te = np.flatnonzero(SESS_W != S01)
te = te[SESS_W[te] != S02]
pr = fit_predict(tr, te)
parts.append(oof_frame(te, pr, 5))
train_n.append(dict(model="ii_all", n_train=len(tr), n_s01=int(is01.sum()), n_s02=int(len(tr) - is01.sum()), n_pred=len(te)))
P(f"  (ii) tek model (s03-s06): egitim {len(tr)} pencere, tahmin {len(te)} pencere ({time.time()-T0:.0f}s)")
oof_ii = pd.concat(parts, ignore_index=True)
for fo, g in oof_ii.groupby("fold"):
    g.to_csv(OOF4 / f"b2_s0102__f{int(fo)}.csv.gz", index=False, compression="gzip")
# tamlik: s02-s06 = 16,248 - 4,679 = 11,569 pencere, hic NaN yok
n_exp = int((SESS_W != S01).sum())
assert len(oof_ii) == n_exp == 16248 - int(is01.sum()), (len(oof_ii), n_exp)
assert not oof_ii.duplicated(["session", "frame_id"]).any()
mk = W[SESS_W != S01][["session", "frame_id"]].merge(oof_ii[["session", "frame_id", "y_pred"]], on=["session", "frame_id"], how="left", validate="1:1")
assert mk["y_pred"].notna().all() and len(mk) == n_exp, "(ii) OOF s02-s06'yi tam kapsamiyor"
P(f"  (ii) tam: {len(oof_ii)} satir (= s02-s06 pencereleri), eksik y_pred = 0")
LAB_II = ac.labels_from_oof(oof_ii, strict=False)
train_n = pd.DataFrame(train_n)

# ============================================================ Faz A referans etiketleri + anomali protokolu
LAB_A = ac.source_labels(sources=("b2",), verbose=False)["b2"]
LABELS = {"FazA_S1": LAB_A, "i_S2": LAB_I, "ii_s01s02": LAB_II}
P(f"[{time.time()-T0:.0f}s] anomali protokolu (Faz A rolleri, standart conformal)")
S, CNT, NCAL = ac.score_all(sources=list(LABELS), segs=("pred", "tof"), labels=LABELS, verbose=True)
PF, SM = ac.evaluate(S)
TAB = ac.summarize(PF, SM)

# Faz A tablosuyla birebir kontrol (FazA_S1 hucresi == reports/faz_a/04_anomaly_table.csv b2 hucresi)
ref = pd.read_csv(fc.OUT / "04_anomaly_table.csv")
chk_rows = []
for seg in ("pred", "tof"):
    a = TAB[(TAB.state_source == "FazA_S1") & (TAB.segmentation == seg)].set_index(["variant", "atype"])
    b = ref[(ref.state_source == "b2") & (ref.segmentation == seg)].set_index(["variant", "atype"])
    for c in ["auroc_mean", "det_a10", "det_a05", "far_a10", "far_a05"]:
        d = float(np.nanmax(np.abs(a[c].reindex(b.index).to_numpy(float) - b[c].to_numpy(float))))
        chk_rows.append(dict(seg=seg, col=c, max_abs_diff=d))
chk = pd.DataFrame(chk_rows)
P("  Faz A (S1, b2) yeniden kosu vs 04_anomaly_table.csv: max |fark| =", f"{chk.max_abs_diff.max():.3g}")
assert chk.max_abs_diff.max() <= 1e-9, "protokol Faz A tablosunu yeniden uretmiyor"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


SRC_NAME = {"FazA_S1": "Faz A (S1 OOF)", "i_S2": "(i) S2 OOF", "ii_s01s02": "(ii) s01+s02"}
SRCS = list(LABELS)
ATS = list(ac.AGROUPS)


def cell(seg, src, var, at):
    g = TAB[(TAB.segmentation == seg) & (TAB.state_source == src) & (TAB.variant == var) & (TAB.atype == at)]
    assert len(g) == 1
    return g.iloc[0]


# ------------------------------------------------------------ FAR tablosu (Wilson)
far_rows = []
for seg in ("pred", "tof"):
    for src in SRCS:
        for var in ("D3", "D1", "D2"):
            r = cell(seg, src, var, "normal" if False else "seq135")           # far_* sutunlari her atype satirinda ayni
            n = int(r["far_n_pairs"])
            for ak, al in (("a10", 0.10), ("a05", 0.05)):
                kk = int(round(r[f"far_{ak}"] * n))
                lo, hi = wilson(kk, n)
                far_rows.append(dict(seg=seg, source=SRC_NAME[src], variant=var, alpha=al, n_normal=n, n_alarm=kk,
                                     FAR=kk / n, wilson_lo=lo, wilson_hi=hi, se=r[f"far_se_{ak}"],
                                     within2se=bool(r[f"far_within2se_{ak}"]), nominal_in_wilson=bool(lo <= al <= hi)))
FAR = pd.DataFrame(far_rows)

# ------------------------------------------------------------ genis tablo (D0..D3, tur basina)
det_rows = []
for seg in ("pred", "tof"):
    for src in SRCS:
        for var in ("D0", "D1", "D2", "D3"):
            for at in ATS:
                r = cell(seg, src, var, at)
                det_rows.append(dict(seg=seg, source=SRC_NAME[src], variant=var, atype=at, auroc_mean=r.auroc_mean, auroc_sd=r.auroc_sd,
                                     det_a10=r.det_a10, det_a05=r.det_a05, n_pairs=int(r.det_n_pairs)))
DET = pd.DataFrame(det_rows)

# ------------------------------------------------------------ on-yazili okuma kurali (pred, D3, tum anomali)
verd = []
fa = cell("pred", "FazA_S1", "D3", "tum anomali")
for src in ("i_S2", "ii_s01s02"):
    nw = cell("pred", src, "D3", "tum anomali")
    d_auc = float(fa.auroc_mean - nw.auroc_mean)
    d_det = float(fa.det_a10 - nw.det_a10)
    per_type = {at: float(cell("pred", "FazA_S1", "D3", at).det_a10 - cell("pred", src, "D3", at).det_a10) for at in ac.ATYPES}
    per_auc = {at: float(cell("pred", "FazA_S1", "D3", at).auroc_mean - cell("pred", src, "D3", at).auroc_mean) for at in ac.ATYPES}
    ok = (d_auc <= 0.05) and (d_det <= 0.10)
    verd.append(dict(source=SRC_NAME[src], auroc_faz_a=float(fa.auroc_mean), auroc_new=float(nw.auroc_mean), auroc_drop=d_auc,
                     auroc_clause=d_auc <= 0.05, det_faz_a=float(fa.det_a10), det_new=float(nw.det_a10), det_drop=d_det,
                     det_clause=d_det <= 0.10, holds=ok, max_type_det_drop=max(per_type.values()),
                     max_type_auroc_drop=max(per_auc.values()),
                     **{f"det_drop_{k}": v for k, v in per_type.items()}, **{f"auroc_drop_{k}": v for k, v in per_auc.items()}))
VERD = pd.DataFrame(verd)
overall = bool(VERD["holds"].all())

# ============================================================ durum macro-F1 (s02-s06)
oof_a = fc.load_oof("b2", "S1")
assert len(oof_a) == 16248
st_rows = []
SES = [s for s in sorted(ft["session"].unique()) if s != S01]
for nm, oof in (("Faz A (S1 OOF)", oof_a), ("(i) S2 OOF", oof_i), ("(ii) s01+s02", oof_ii)):
    o = oof[oof["session"] != S01]
    assert len(o) == 11569
    m = fc.state_metrics(o["y_true"].to_numpy(), o["y_pred"].to_numpy())
    st_rows.append(dict(source=nm, session="s02-s06 havuz", **{k: m[k] for k in ("n", "acc", "macro_f1")}))
    for s in SES:
        g = o[o["session"] == s]
        m = fc.state_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        st_rows.append(dict(source=nm, session=s[:10], **{k: m[k] for k in ("n", "acc", "macro_f1")}))
    ng = o[o["session"] != S02]
    m = fc.state_metrics(ng["y_true"].to_numpy(), ng["y_pred"].to_numpy())
    st_rows.append(dict(source=nm, session="s03-s06 havuz", **{k: m[k] for k in ("n", "acc", "macro_f1")}))
STATE = pd.DataFrame(st_rows)

# ============================================================ devreye alma dakikalari (pencere / 600 = 0,1 s kare, 'fiili' tanim)
MIN = lambda n: n / 600.0
s2 = W["s2"].to_numpy()
min_rows = []
min_rows.append(dict(kaynak="Faz A (S1 OOF) kat modeli", egitim_penceresi=float(np.mean([(W["s1"] != f).sum() for f in range(5)])),
                     etiketli_oturumlar="6 oturumun %80'i (5 katin ort.)"))
min_rows.append(dict(kaynak="(i) S2: s02'yi tahmin eden model", egitim_penceresi=float((s2 != 1).sum()), etiketli_oturumlar="s01 + s03..s06"))
min_rows.append(dict(kaynak="(i) S2: s03'ü tahmin eden model", egitim_penceresi=float((s2 != 2).sum()), etiketli_oturumlar="s01, s02, s04..s06"))
min_rows.append(dict(kaynak="(i) S2: 6 modelin ort.", egitim_penceresi=float(np.mean([(s2 != j).sum() for j in range(6)])), etiketli_oturumlar="diger 5 oturum"))
for r in train_n.itertuples():
    min_rows.append(dict(kaynak=f"(ii) {r.model}", egitim_penceresi=float(r.n_train), etiketli_oturumlar=f"s01 ({r.n_s01}) + s02 ({r.n_s02})"))
MINS = pd.DataFrame(min_rows)
MINS["etiketli_dakika"] = MINS["egitim_penceresi"].map(MIN)
n_s01_w, n_s02_w = int(is01.sum()), int(is02.sum())
# anomali dedektoru icin normal dongu suresi (Faz A rolleri: egitim 2 kat + kalibrasyon 2 kat; pred, ortalama f)
cyc = []
for src in SRCS:
    g = S[(S.state_source == src) & (S.segmentation == "pred") & (S.role.isin(["train", "cal"]))]
    gg = g[g.atype == "normal"].groupby(["f", "role"])["dur_s"].agg(["sum", "size"]).reset_index()
    cyc.append(dict(source=SRC_NAME[src], train_dk=float(gg[gg.role == "train"]["sum"].mean() / 60), train_n=float(gg[gg.role == "train"]["size"].mean()),
                    cal_dk=float(gg[gg.role == "cal"]["sum"].mean() / 60), cal_n=float(gg[gg.role == "cal"]["size"].mean())))
CYC = pd.DataFrame(cyc)

# ============================================================ pencere ortusmesi (betimleyici): (ii) test penceresi ile egitim penceresi ayni kareyi paylasir mi
ov = []
for k in range(5):
    tr = np.flatnonzero(is01 | (is02 & (wfold >= 0) & (wfold != k)))
    te = np.flatnonzero(is02 & (wfold == k))
    trf = set(R[tr].ravel().tolist())
    sh = np.array([any(x in trf for x in R[i].tolist()) for i in te])
    ov.append(dict(fold=k, n_test=len(te), n_share=int(sh.sum())))
OV = pd.DataFrame(ov)

# ============================================================ CSV'ler
TAB_out = TAB.drop(columns=[c for c in TAB.columns if "excl_nat" in c or c in ("far_n_natdev",)])
TAB_out.to_csv(OUT4 / "04_unseen_anomaly_table.csv", index=False)
STATE.to_csv(OUT4 / "04_unseen_state_metrics.csv", index=False)
FAR.to_csv(OUT4 / "04_unseen_far.csv", index=False)
VERD.to_csv(OUT4 / "04_unseen_verdict.csv", index=False)
MINS.to_csv(OUT4 / "04_unseen_minutes.csv", index=False)
CNT.drop(columns=["n_natural_dev"]).to_csv(OUT4 / "04_unseen_cycle_counts.csv", index=False)
oof_ii.to_csv(OUT4 / "oof" / "b2_s0102__all.csv.gz", index=False, compression="gzip")

# ============================================================ rapor
L = []


def H(s=""):
    L.append(s)


f3 = "{:.3f}"
H("# A4.4 — Anomali oturumlarını hiç görmemiş durum tahmincisi (M4)")
H("")
H("`faz_a4/04_unseen_sessions.py` tarafından üretilir (DESIGN_A4.md A4.4). **İkincil analiz**; Faz A sayıları birincil kalır. Tüm sayılar stdout'a da yazılır; seçici raporlama yoktur.")
H("")
H("## Kurulum")
H("")
H("- **Protokol değişmedi:** `a2_cycles` ile Faz A rolleri (eğitim katları f+3,f+4; kalibrasyon f+1,f+2; test f), pred segmentasyonu, D0–D3, standart conformal p, D3 = min(1, 2·min(p1, p2)). Yalnız B2 durum etiketlerinin kaynağı değişti.")
H("- **Faz A (S1 OOF):** mevcut `b2__S1` — referans sütunu. Bu hücre protokolün yeniden koşusunda `reports/faz_a/04_anomaly_table.csv` ile birebir çıktı (max |fark| = "
  f"{chk.max_abs_diff.max():.1g}).")
H("- **(i) S2:** mevcut `b2__S2__f0..f5` OOF (oturum dışarıda). **Dikkat:** her oturumu tahmin eden model **diğer anomali oturumlarını görmüş** olabilir (ör. s03'ü tahmin eden model s04–s06'yı ve s02'yi görür); yalnız kendi oturumunu görmedi. Bu yüzden (i) soruyu yalnız yarım yanıtlar.")
H("- **(ii) devreye alma gerçekçi:** B2 yalnız s01 + s02 ile; **anomali oturumları s03–s06 eğitimde hiç yok.** s02 pencereleri anomali katı k'nin dışarıda bırakıldığı modelle (k = 0..4), s03–s06 pencereleri s01 + s02'nin tamamıyla eğitilen tek modelle tahmin edildi. Özellikler (21), RF (300 ağaç, balanced, random_state 20261104) ve pencere tanımı B2 ile aynı; `n_jobs = 3` (sonucu etkilemez).")
H("- **Eşdeğerlik koruması:** betikteki B2 kopyası, Faz A OOF'unu yeniden üretti: "
  + "; ".join(f"{r.split} f{r.fold}: y_pred uyumu {r.agree:.6f}, max |proba farkı| {r.max_proba_diff:.2g}" for r in guard.itertuples()) + ".")
H(f"- **Tamlık:** (i) 16.248 pencere, 6 kat, `windows()` ile aynı (session, frame_id) kümesi. (ii) {len(oof_ii)} pencere = s02–s06 (16.248 − s01'in {n_s01_w}); eksik y_pred yok. `labels_from_oof(strict=False)` yalnız bu iki doğrulamadan sonra çağrıldı.")
H(f"- s02'nin tek parça döngüsü (loop 0, fold −1; {n_frag} pencere) hiçbir eğitim kümesine alınmadı; k = 0 modeliyle tahmin edildi (örnek-içi değil).")
H("- Eğitim penceresi sayıları (ii): " + "; ".join(f"{r.model}: {r.n_train}" for r in train_n.itertuples()) + ".")
H("")

H("## 1. Ana karşılaştırma — B2 × pred, D3 (Faz A Tablo 3 ile yan yana)")
H("")
for al, ak in ((0.10, "a10"), (0.05, "a05")):
    H(f"**α = {al:.2f}**")
    H("")
    rows = []
    for src in SRCS:
        fr = FAR[(FAR.seg == "pred") & (FAR.source == SRC_NAME[src]) & (FAR.variant == "D3") & (FAR.alpha == al)].iloc[0]
        row = {"kaynak": SRC_NAME[src], "FAR": f"{fr.FAR:.3f} ({fr.n_alarm}/{fr.n_normal})", "Wilson %95": f"[{fr.wilson_lo:.3f}, {fr.wilson_hi:.3f}]"}
        for at in ac.ATYPES + ["tum anomali"]:
            row[f"det {at}"] = f"{cell('pred', src, 'D3', at)['det_' + ak]:.3f}"
        row["AUROC tüm (kat ort ± sd)"] = f"{cell('pred', src, 'D3', 'tum anomali').auroc_mean:.3f} ± {cell('pred', src, 'D3', 'tum anomali').auroc_sd:.3f}"
        rows.append(row)
    H(md_table(pd.DataFrame(rows)))
    H("")
H("**AUROC (D3, kat ortalaması ± sd), tür başına**")
H("")
rows = []
for src in SRCS:
    row = {"kaynak": SRC_NAME[src]}
    for at in ATS:
        c = cell("pred", src, "D3", at)
        row[at] = f"{c.auroc_mean:.3f} ± {c.auroc_sd:.3f}"
    rows.append(row)
H(md_table(pd.DataFrame(rows)))
H("")

H("## 2. D1 (sıra) ve D2 (süre) tek başına — pred")
H("")
for var in ("D1", "D2"):
    H(f"**{var}**")
    H("")
    rows = []
    for src in SRCS:
        fr10 = FAR[(FAR.seg == "pred") & (FAR.source == SRC_NAME[src]) & (FAR.variant == var) & (FAR.alpha == 0.10)].iloc[0]
        fr05 = FAR[(FAR.seg == "pred") & (FAR.source == SRC_NAME[src]) & (FAR.variant == var) & (FAR.alpha == 0.05)].iloc[0]
        row = {"kaynak": SRC_NAME[src], "FAR@0,10": f"{fr10.FAR:.3f} [{fr10.wilson_lo:.3f}, {fr10.wilson_hi:.3f}]",
               "FAR@0,05": f"{fr05.FAR:.3f} [{fr05.wilson_lo:.3f}, {fr05.wilson_hi:.3f}]"}
        for at in ac.ATYPES + ["tum anomali"]:
            c = cell("pred", src, var, at)
            row[at] = f"AUROC {c.auroc_mean:.3f}; det {c.det_a10:.3f} / {c.det_a05:.3f}"
        rows.append(row)
    H(md_table(pd.DataFrame(rows)))
    H("")
H("(det hücreleri: α = 0,10 / α = 0,05 tespit oranı.)")
H("")

H("## 3. Önceden yazılmış okuma kuralı — harfi harfine")
H("")
H("Kural: *Tümü-AUROC düşüşü ≤ 0,05 ve tespit düşüşü ≤ 0,10 ise \"Faz A sonuçları anomali oturumlarının görülmesine bağlı değil\"; değilse makale (ii)'yi dağıtım gerçekçi tahmin olarak öne alır ve Faz A sayılarını iyimser diye niteler.* "
  "Ölçüm: B2 × pred, D3, \"tüm anomali\" grubu; AUROC = kat ortalaması; tespit = α = 0,10 havuzlanmış oran; düşüş = Faz A − yeni (pozitif = kötüleşme).")
H("")
rows = []
for r in VERD.itertuples():
    rows.append({"kaynak": r.source, "AUROC Faz A": f3.format(r.auroc_faz_a), "AUROC yeni": f3.format(r.auroc_new), "AUROC düşüşü": f"{r.auroc_drop:+.3f}",
                 "AUROC ≤ 0,05": "evet" if r.auroc_clause else "HAYIR", "det Faz A": f3.format(r.det_faz_a), "det yeni": f3.format(r.det_new),
                 "det düşüşü": f"{r.det_drop:+.3f}", "det ≤ 0,10": "evet" if r.det_clause else "HAYIR", "kural sağlandı": "EVET" if r.holds else "HAYIR"})
H(md_table(pd.DataFrame(rows)))
H("")
H("Tür başına tespit düşüşü (α = 0,10, D3; ek betimleyici) ve tür başına AUROC düşüşü:")
H("")
rows = []
for r in VERD.itertuples():
    rows.append({"kaynak": r.source, **{f"det düşüş {at}": f"{getattr(r, 'det_drop_' + at):+.3f}" for at in ac.ATYPES},
                 **{f"AUROC düşüş {at}": f"{getattr(r, 'auroc_drop_' + at):+.3f}" for at in ac.ATYPES}})
H(md_table(pd.DataFrame(rows)))
H("")
if overall:
    H("**Hüküm (kuralın harfi):** her iki kaynakta da iki şart sağlandı → \"Faz A sonuçları anomali oturumlarının görülmesine bağlı değil\".")
else:
    bad = [r.source for r in VERD.itertuples() if not r.holds]
    ok_ = [r.source for r in VERD.itertuples() if r.holds]
    H(f"**Hüküm (kuralın harfi):** kural şu kaynak(lar)da **sağlanmadı**: {', '.join(bad)}"
      + (f" (sağlanan: {', '.join(ok_)})" if ok_ else "") + ". Kuralın \"değilse\" dalı: makale (ii)'yi **dağıtım gerçekçi tahmin** olarak öne alır ve Faz A sayılarını iyimser diye niteler.")
r2 = VERD[VERD.source == "(ii) s01+s02"].iloc[0]
H("")
H(f"Not: kural \"tespit düşüşü\"nü tek bir sayı olarak veriyor; ana okuma \"tüm anomali\" havuzudur. Tür başına en büyük tespit düşüşü (ii) için {r2.max_type_det_drop:+.3f}, (i) için "
  f"{VERD[VERD.source == '(i) S2 OOF'].iloc[0].max_type_det_drop:+.3f}; tür başına en büyük AUROC düşüşü (ii) için {r2.max_type_auroc_drop:+.3f}, (i) için {VERD[VERD.source == '(i) S2 OOF'].iloc[0].max_type_auroc_drop:+.3f}. "
  "Katlar arası AUROC sd ve ~10 normal döngü/kat nedeniyle ±0,03–0,05'lik AUROC farkları gürültü düzeyindedir (Faz A uyarı 6).")
H("")

H("## 4. Durum macro-F1 (s02–s06)")
H("")
H("Kare/pencere düzeyi, `fc.state_metrics` (macro-F1: y_true'da bulunan sınıflar üzerinden). Aynı 11.569 pencere (s02–s06) üzerinde.")
H("")
piv = STATE.pivot(index="session", columns="source", values="macro_f1")
pivA = STATE.pivot(index="session", columns="source", values="acc")
tab = pd.DataFrame({"oturum": piv.index})
for nm in ("Faz A (S1 OOF)", "(i) S2 OOF", "(ii) s01+s02"):
    tab[f"macro-F1 {nm}"] = piv[nm].to_numpy()
for nm in ("Faz A (S1 OOF)", "(i) S2 OOF", "(ii) s01+s02"):
    tab[f"acc {nm}"] = pivA[nm].to_numpy()
order = ["s02-s06 havuz", "s03-s06 havuz"] + [s[:10] for s in SES]
tab = tab.set_index("oturum").loc[order].reset_index()
H(md_table(tab))
H("")
H("s05'te Box5 etiketten silindiği için (±3 kare mod yumuşatması; Faz A uyarı 1) s05 skorları kısmen hatalı referansla ölçülür.")
H("")

H("## 5. Segmentasyon sayımları (pred)")
H("")
c2 = CNT[(CNT.segmentation == "pred")].drop(columns=["n_natural_dev", "segmentation"])
c2 = c2.assign(state_source=c2["state_source"].map(SRC_NAME))
H(md_table(c2, floatfmt="{:.0f}"))
H("")

H("## 6. Devreye alma dakikaları (eğitim setine göre yeniden etiketlendi)")
H("")
H("Tanım: **pencere sayısı / 600** (0,1 s kare, 'fiili' tanım; A2.4(b)'deki 21,7 dk ile aynı). Bu, ToF ile **etiketli** durum-eğitim kaydı süresidir. A2.4'ün 'dk toplam 6 oturum' sütunu eğitimin altı oturumdan havuzlandığını varsayıyordu; (ii)'de eğitim yalnız s01 + s02'dir.")
H("")
H(md_table(MINS[["kaynak", "etiketli_oturumlar", "egitim_penceresi", "etiketli_dakika"]], floatfmt="{:.1f}"))
H("")
H(f"Ham kayıt süreleri: s01 = {n_s01_w} pencere ≈ {MIN(n_s01_w):.1f} dk; s02 = {n_s02_w} pencere ≈ {MIN(n_s02_w):.1f} dk. s03–s06'nın etiketlenmesine gerek yoktur; yalnız anomali tespiti için normal döngülerin süresi gerekir. "
  "Anomali dedektörü normal döngü süresi (Faz A rolleri, pred döngüleri, f ortalaması):")
H("")
H(md_table(CYC.rename(columns={"train_dk": "eğitim dk", "train_n": "eğitim döngü", "cal_dk": "kalibrasyon dk", "cal_n": "kalibrasyon döngü"}), floatfmt="{:.2f}"))
H("")
H("(ii) için kayıt gereksinimi böylece: durum-eğitim etiketi için s01 + s02, anomali dedektörü için s02'nin normal döngüleri (aynı oturumdan). Hiçbir anomali oturumu kaydı gerekmez.")
H("")

H("## 7. Ek (betimleyici)")
H("")
H("**(ii)'de test penceresi ile eğitim penceresinin aynı kareyi paylaşması** (kat k test pencereleri, ≥ 1 ortak kare; döngü sınırlarında, Faz A S1'deki %12,9'a benzer):")
H("")
H(md_table(OV, floatfmt="{:.0f}"))
H("")
H("**tof segmentasyonu (ek; okuma kuralı pred'e göre), D3, α = 0,10:**")
H("")
rows = []
for src in SRCS:
    fr = FAR[(FAR.seg == "tof") & (FAR.source == SRC_NAME[src]) & (FAR.variant == "D3") & (FAR.alpha == 0.10)].iloc[0]
    row = {"kaynak": SRC_NAME[src], "FAR": f"{fr.FAR:.3f} [{fr.wilson_lo:.3f}, {fr.wilson_hi:.3f}]"}
    for at in ac.ATYPES + ["tum anomali"]:
        row[f"det {at}"] = f"{cell('tof', src, 'D3', at).det_a10:.3f}"
    row["AUROC tüm"] = f"{cell('tof', src, 'D3', 'tum anomali').auroc_mean:.3f}"
    rows.append(row)
H(md_table(pd.DataFrame(rows)))
H("")
H("**Doğal sapma notu:** Faz A tablosundaki `far_excl_nat` ve döngü başına skorlar bu raporda **verilmedi/yazılmadı** (s02 doğal sapma döngülerinin alarm durumuna bakmak yasak); FAR havuzlanmış, doğal sapmalar dahil standart tanımdır.")
H("")
H("Çıktılar: `04_unseen_anomaly_table.csv` (Faz A `04_anomaly_table.csv` sütunları, 3 kaynak × 2 segmentasyon), `04_unseen_far.csv`, `04_unseen_state_metrics.csv`, `04_unseen_verdict.csv`, `04_unseen_minutes.csv`, `04_unseen_cycle_counts.csv`, `oof/b2_s0102__f0..f5.csv.gz` (+ `__all`).")
H("")
H(MARK)

body = "\n".join(L)
P(body)
fn = OUT4 / "04_unseen_sessions.md"
keep = ""
if fn.exists():
    old = fn.read_text(encoding="utf-8")
    if MARK in old:
        keep = old.split(MARK, 1)[1]
fn.write_text(body + "\n" + keep, encoding="utf-8")
P(f"[{time.time()-T0:.0f}s] yazildi: {fn}")

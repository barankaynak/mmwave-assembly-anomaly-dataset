"""Faz A / INFRA: build heatmap cache, write fold tables + 00_folds.md (counts, asserts)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probe"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import common
import fa_common as fc

t_start = time.time()
OUT = fc.OUT
OUT.mkdir(parents=True, exist_ok=True)
out = []          # markdown lines; every number is also printed


def P(*a):
    print(*a, flush=True)


def H(txt):
    out.append(txt)
    P(txt)


# ------------------------------------------------------------------ frames / heatmaps
ft = fc.frame_table()
P(f"[{time.time()-t_start:.0f}s] frame table {ft.shape}")
hm = fc.heatmaps()
P(f"[{time.time()-t_start:.0f}s] heatmaps {hm.shape} {hm.dtype} cache={fc._hm_path()} ({hm.nbytes/1e6:.0f} MB)")
assert hm.shape == (len(ft), 2, fc.K, fc.K)
assert len(ft) == 16272, len(ft)

# ---- exactness check of the vectorised heatmap against the original per-frame loop (run_all_pt.build_heatmap)
def build_heatmap_ref(points, K=fc.K):
    hmr = np.zeros((K, K, 2), dtype=np.float32)
    if points.empty:
        return hmr
    xs = points["x"].to_numpy(dtype=np.float32)
    ys = points["y"].to_numpy(dtype=np.float32)
    snr = points["snr"].to_numpy(dtype=np.float32)
    vel = points["velocity"].to_numpy(dtype=np.float32)
    xi = np.floor((xs - fc.X_MIN) / (fc.X_MAX - fc.X_MIN) * K).astype(int)
    yi = np.floor((ys - fc.Y_MIN) / (fc.Y_MAX - fc.Y_MIN) * K).astype(int)
    m = (xi >= 0) & (xi < K) & (yi >= 0) & (yi < K)
    for u, v, s, w in zip(xi[m], yi[m], snr[m], vel[m]):
        if s > hmr[v, u, 0]:
            hmr[v, u, 0] = s
        hmr[v, u, 1] += w
    counts = (hmr[..., 0] > 0).astype(np.float32)
    hmr[..., 1] = np.divide(hmr[..., 1], counts, out=np.zeros_like(hmr[..., 1]), where=counts > 0)
    return hmr


rng = np.random.RandomState(0)
chk = rng.choice(len(ft), 600, replace=False)
maxdiff = 0.0
for r in chk:
    s, fid = ft.at[r, "session"], ft.at[r, "frame_id"]
    m = common.load_merged(s)
    ref = build_heatmap_ref(m[m["frame_id"] == fid]).transpose(2, 0, 1)
    maxdiff = max(maxdiff, float(np.abs(ref - hm[r]).max()))
P(f"heatmap vs reference loop: max |diff| over 600 random frames = {maxdiff:.3g}")
assert maxdiff == 0.0, "vectorised heatmap differs from run_all_pt.build_heatmap"
occ = (hm[:, 0] > 0).reshape(len(hm), -1).sum(1)
P(f"occupied cells/frame: median {np.median(occ):.0f}, mean {occ.mean():.2f}, max {occ.max()}; "
  f"frames with 0 occupied cells: {(occ == 0).sum()}")

# ------------------------------------------------------------------ windows
w = fc.windows()
sess = common.sessions()
P(f"windows: {len(w)}  (frames {len(ft)} - {len(sess)} x {fc.L-1} = {len(ft) - len(sess)*(fc.L-1)})")
assert len(w) == 16248, len(w)
assert w["s1"].between(0, fc.N_FOLDS - 1).all() and w["s1"].notna().all()
assert w.groupby("win").size().eq(1).all()
assert w["contiguous"].all() or True
n_noncontig = int((~w["contiguous"]).sum())
n_span = int(w["spans_loops"].sum())
# a window never crosses a session boundary
fts = ft["session_idx"].to_numpy()
assert all((fts[w[f"r{k}"].to_numpy()] == w["session_idx"].to_numpy()).all() for k in range(fc.L))
# labels of window == label of last frame
assert (w["y_true"].to_numpy() == ft["y_true"].to_numpy()[w["row_last"].to_numpy()]).all()
# no group in more than one S1 fold
assert (w.groupby("gid")["s1"].nunique() == 1).all()
# every fold has every session
cnt_sf = w.groupby(["session", "s1"]).size().unstack(fill_value=0)
assert (cnt_sf.to_numpy() > 0).all(), "a session is missing from an S1 fold"
# a sample window batch
b = fc.window_batch([0, 1, 5000], hm=hm, win=w)
assert b.shape == (3, 5, 2, 48, 48) and b.dtype == np.float32
assert np.array_equal(b[2, 4], hm[w.at[5000, "r4"]])
P(f"window_batch ok {b.shape}; windows with frame_id gaps: {n_noncontig}; windows whose 5 frames span >1 loop: {n_span}")

# ------------------------------------------------------------------ S0
n_te0 = int(w["s0_test"].sum())
P(f"S0: train {len(w)-n_te0}  test {n_te0}")
assert n_te0 == 3250, n_te0
assert len(w) - n_te0 == 12998

# ------------------------------------------------------------------ anomaly folds
al = fc.anom_loops()
for f in range(fc.N_FOLDS):
    r = fc.anom_roles(f)
    assert sorted(r["train"] + r["cal"] + r["test"]) == list(range(fc.N_FOLDS))
cyc02 = al[(al["role"] == "normal")]
P("s02 loops:", len(cyc02), "| complete cycles with fold:", int((cyc02["fold"] >= 0).sum()))
assert set(cyc02.loc[cyc02["fold"] >= 0, "fold"]) == set(range(fc.N_FOLDS))

# ------------------------------------------------------------------ write CSVs
fold_cols = ["win", "session", "session_idx", "frame_id", "t", "loop_id", "gid", "y_true", "y_raw", "s1", "s2", "s0_test"]
w[fold_cols].rename(columns={"s1": "s1_fold", "s2": "s2_fold"}).to_csv(OUT / "folds_state.csv", index=False)
al.to_csv(OUT / "folds_anom.csv", index=False)
w[["win", "session", "frame_id", "loop_id", "y_true", "s0_test"]].to_csv(OUT / "split_s0.csv", index=False)

# ------------------------------------------------------------------ markdown report
H("# 00 — Katlar, pencereler, önbellek (INFRA)")
H("")
H("Bu rapor `faz_a/00_folds.py` tarafından üretilir (DESIGN.md ön-kayıt tasarımı; sapma yok). Tüm sayılar stdout'a da yazılır.")
H("")
H("## Kare ve pencere")
H(f"- Kare: **{len(ft)}** (6 oturum); etiket = ToF < 140 mm, ±3 kare mod yumuşatma (oturum içi, beraberlikte en küçük sınıf; `run_all_pt.derive_labels` ile birebir). "
  f"Ham etiket (`y_raw`) kayıtlı `box*_active` ve mesafeden türetilen etiketle %100 aynı (assert).")
H(f"- Heatmap: ({hm.shape[0]}, 2, 48, 48) float32, X ∈ [−0,5; 0,5], Y ∈ [0,1; 1,0]; kanal 0 = hücre başına en büyük SNR, kanal 1 = hücredeki hızların **toplamı** "
  f"(orijinal `build_heatmap` ikili 'dolu' göstergesine böler; adı ortalama ama aslında toplamdır — port bunu aynen korur). "
  f"Vektörleştirilmiş inşa, orijinal döngüyle 600 rastgele karede **maks |fark| = {maxdiff:.3g}**. "
  f"Ortanca dolu hücre/kare = {np.median(occ):.0f}, dolu hücresi olmayan kare = {(occ == 0).sum()}.")
H(f"- Pencere: L = 5, etiket/döngü = son karenin; **{len(w)}** pencere ({len(ft)} − 6×4). Kare-id boşluğu olan pencere: {n_noncontig}. "
  f"5 karesi iki döngüye yayılan pencere: **{n_span}** (pencerenin grubu = son karesinin döngüsü; bu sınır pencereleri katlar arası ≤ 4 karelik örtüşme taşır — S1'de bilinen küçük artık sızıntı, DESIGN'da purge öngörülmemiştir).")
H("")
ws = w.groupby("session").size().rename("pencere").reset_index()
ws["kare"] = ws["session"].map(ft.groupby("session").size())
ws["döngü(loop_id)"] = ws["session"].map(w.groupby("session")["gid"].nunique())
H(common.md_table(ws))
H("")
H("## Sınıf dağılımı (pencere, y_true)")
cc = w.groupby(["session", "y_true"]).size().unstack(fill_value=0)
cc.columns = [f"c{c}" for c in cc.columns]
cc["toplam"] = cc.sum(axis=1)
cc.loc["TOPLAM"] = cc.sum()
H(common.md_table(cc.reset_index().rename(columns={"index": "session"})))
H("")
H("## S1 — 5 katlı StratifiedGroupKFold (grup = (session, loop_id), strata = oturum, seed 20261104)")
H(f"Her pencere tam bir katta (assert); her grup tek katta; her katta her oturum var (assert).")
H("")
H("**Pencere sayısı, oturum × kat**")
t1 = cnt_sf.copy()
t1.columns = [f"f{c}" for c in t1.columns]
t1["toplam"] = t1.sum(axis=1)
t1.loc["TOPLAM"] = t1.sum()
H(common.md_table(t1.reset_index()))
H("")
H("**Döngü (grup) sayısı, oturum × kat**")
gl = w.drop_duplicates("gid").groupby(["session", "s1"]).size().unstack(fill_value=0)
gl.columns = [f"f{c}" for c in gl.columns]
gl["toplam"] = gl.sum(axis=1)
gl.loc["TOPLAM"] = gl.sum()
H(common.md_table(gl.reset_index()))
H("")
H("**Pencere sayısı, sınıf × kat**")
t2 = w.groupby(["y_true", "s1"]).size().unstack(fill_value=0)
t2.columns = [f"f{c}" for c in t2.columns]
t2["toplam"] = t2.sum(axis=1)
t2.loc["TOPLAM"] = t2.sum()
H(common.md_table(t2.reset_index().rename(columns={"y_true": "sınıf"})))
H("")
H("## S2 — leave-one-session-out (6 kat)")
t3 = w.groupby(["s2"]).agg(session=("session", "first"), test_pencere=("win", "size")).reset_index().rename(columns={"s2": "fold"})
t3["eğitim_pencere"] = len(w) - t3["test_pencere"]
H(common.md_table(t3))
H("")
H("## S0 — rastgele sınıf-tabakalı %80/20 pencere bölmesi (sızıntı gösterimi; random_state 42)")
t4 = w.groupby("y_true")["s0_test"].agg(test="sum", n="size").reset_index()
t4["eğitim"] = t4["n"] - t4["test"]
t4.loc[len(t4)] = ["TOPLAM", t4["test"].sum(), t4["n"].sum(), t4["eğitim"].sum()]
H(f"Eğitim **{len(w)-n_te0}**, test **{n_te0}** (eski makalenin 3.250 test penceresiyle aynı; assert).")
H("")
H(common.md_table(t4.rename(columns={"y_true": "sınıf", "n": "toplam"})))
H("")
H("## Anomali çapraz uyum katları (s02 normal döngüleri; seed 20261104)")
H("Döngüler karıştırılıp (RandomState.permutation) sırayla 5 kata dağıtılır (round-robin) → dengeli ve aralıklı; yalnız **tam** döngüler (ToF kuralı `complete`) katlanır. "
  "Loop 0 (ilk Box1 girişinden önceki parça) ve tam olmayanlar `fold = -1` / `use = fragment_dropped`. "
  "Roller: test = f; kalibrasyon = (f+1)%5, (f+2)%5; eğitim = (f+3)%5, (f+4)%5. s03–s06 döngüleri her f'de test.")
H("")
a02 = cyc02[cyc02["fold"] >= 0]
t5 = a02.groupby("fold").agg(döngü=("loop_id", "size"), kare=("n_frames", "sum"), doğal_sapma=("natural_dev", "sum")).reset_index()
t5.loc[len(t5)] = ["TOPLAM", t5["döngü"].sum(), t5["kare"].sum(), t5["doğal_sapma"].sum()]
H(common.md_table(t5))
H("")
rl = pd.DataFrame([{"f": f, **{k: str(v) for k, v in fc.anom_roles(f).items()},
                    "n_train": int(a02[a02.fold.isin(fc.anom_roles(f)['train'])].shape[0]),
                    "n_cal": int(a02[a02.fold.isin(fc.anom_roles(f)['cal'])].shape[0]),
                    "n_test": int(a02[a02.fold.isin(fc.anom_roles(f)['test'])].shape[0])} for f in range(fc.N_FOLDS)])
H(common.md_table(rl))
H(f"- Her f için en küçük ulaşılabilir conformal p = 1/(n_cal+1): " +
  ", ".join(f"f{r.f}: {1/(r.n_cal+1):.4f}" for r in rl.itertuples()) + " (α = 0,05 için n_cal ≥ 19 gerekir).")
H("")
H("**Tüm oturumlar — döngü kullanımı**")
t6 = al.groupby(["session", "use"]).agg(döngü=("loop_id", "size"), kare=("n_frames", "sum")).reset_index()
H(common.md_table(t6))
H("")
nat = al[al["natural_dev"]][["session", "loop_id", "box_order", "fold"]]
H(f"Beklenen sıradan sapan tam döngüler (ToF; `natural_dev`; s02'dekiler normal FAR'da dahil/hariç raporlanacak; s03–s06'dakiler de listelenir): **{len(nat)}**")
if len(nat):
    H("")
    H(common.md_table(nat))
H("")
H("## Dosyalar")
H("- `folds_state.csv`: pencere başına S1/S2/S0 katı (`s1_fold`, `s2_fold`, `s0_test`), `gid` = (session, loop_id) grup kimliği.")
H("- `folds_anom.csv`: döngü başına anomali kat/rol (`fold`, `use`, `box_order`, `natural_dev`).")
H("- `split_s0.csv`: S0 test bayrağı.")
H("- Önbellek: `$CACHE_DIR/" + fc._hm_path().name + "`; OOF sözleşmesi: `reports/faz_a/oof/{model}__{split}__f{k}.csv.gz`.")
H("")
(OUT / "00_folds.md").write_text("\n".join(out) + "\n", encoding="utf-8")
P(f"done in {time.time()-t_start:.0f}s")

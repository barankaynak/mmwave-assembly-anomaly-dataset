"""Faz A2 / A2.0 - YEREL = UZAK boru hatti denkligi (DESIGN_A2.md 'A2.0').

1) heatmap onbellegini yerelde kur; orijinal run_all_pt.build_heatmap dongusuyle 600 rastgele karede karsilastir (max |fark| = 0)
2) kare / pencere / dongu sayilari (16.272 / 16.248 / S0 3.250) ve katlar: folds_state.csv, folds_anom.csv, split_s0.csv
   yerelde yeniden uretilip Faz A dosyalariyla (bayt bayt ve deger deger) karsilastirilir
3) B2 S1 kat 0 ve 1 yerelde yeniden egitilir (02_state_baselines.py ile ayni ozellik/tohum/hiperparametre), OOF ile karsilastirilir
Cikti: reports/faz_a2/00_equivalence.md (PASS/FAIL). 00b_cycles_selftest.py kendi bolumunu bu dosyanin sonuna ekler.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
sys.path.insert(0, str(ROOT / "probe"))
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

import common
import fa_common as fc

OUT2 = ROOT / "reports" / "faz_a2"
OUT2.mkdir(parents=True, exist_ok=True)
MARK = "<!-- SELFTEST-BEGIN -->"
t0 = time.time()
LINES = []


def P(*a):
    print(*a, flush=True)


def H(s=""):
    LINES.append(s)
    P(s)


checks = {}          # ad -> (bool, aciklama)


def check(name, ok, detail=""):
    checks[name] = (bool(ok), detail)
    P(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ------------------------------------------------------------------ 1) heatmap
ft = fc.frame_table()
P(f"[{time.time()-t0:.0f}s] frame table {ft.shape}")
cache_existed = fc._hm_path().exists()
hm = fc.heatmaps()
P(f"[{time.time()-t0:.0f}s] heatmaps {hm.shape} {hm.dtype} cache={fc._hm_path()} (onceden vardi: {cache_existed})")
check("hm_shape", hm.shape == (len(ft), 2, fc.K, fc.K) and hm.dtype == np.float32, f"{hm.shape} {hm.dtype}")
rebuild_equal = None
if cache_existed:           # onbellek onceden varsa taze bir inşa ile de karsilastir (gercek yeniden kurulum)
    fresh = fc._build_heatmaps()
    rebuild_equal = bool(np.array_equal(fresh, np.asarray(hm)))
    del fresh
    check("hm_fresh_rebuild_equals_cache", rebuild_equal)


def build_heatmap_ref(points, K=fc.K):
    """run_all_pt.build_heatmap (orijinal kare-basi dongu; faz_a/00_folds.py'deki kopya ile ayni)."""
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


rng = np.random.RandomState(0)                    # 00_folds.py ile ayni secim
chk = rng.choice(len(ft), 600, replace=False)
merged_idx = {}
maxdiff = 0.0
n_nonempty = 0
for r in chk:
    s, fid = ft.at[r, "session"], ft.at[r, "frame_id"]
    if s not in merged_idx:
        m = common.load_merged(s)
        merged_idx[s] = (m, m.groupby("frame_id").indices)
    m, ix = merged_idx[s]
    pts = m.iloc[ix[fid]] if fid in ix else m.iloc[0:0]
    ref = build_heatmap_ref(pts).transpose(2, 0, 1)
    n_nonempty += int(len(pts) > 0)
    maxdiff = max(maxdiff, float(np.abs(ref - np.asarray(hm[r])).max()))
check("hm_600_frames_maxdiff_0", maxdiff == 0.0, f"max|diff|={maxdiff:.3g} (600 kare, {n_nonempty} tanesi bos olmayan)")

# ------------------------------------------------------------------ 2) sayilar ve katlar
w = fc.windows()
sess = common.sessions()
check("n_frames_16272", len(ft) == 16272, str(len(ft)))
check("n_windows_16248", len(w) == 16248, str(len(w)))
n_te0 = int(w["s0_test"].sum())
check("s0_split_3250_12998", n_te0 == 3250 and len(w) - n_te0 == 12998, f"test {n_te0} / egitim {len(w) - n_te0}")
al = fc.anom_loops()
loops_local = w.drop_duplicates("gid").groupby("session").size()
fs_ref = pd.read_csv(fc.OUT / "folds_state.csv")
loops_ref = fs_ref.drop_duplicates("gid").groupby("session").size()
check("loop_counts_per_session_equal", loops_local.equals(loops_ref), "; ".join(f"{k.split('_')[1]}={v}" for k, v in loops_local.items()))
n_cyc = int((al["fold"] >= 0).sum())
P(f"s02 katlanan tam dongu: {n_cyc}; kullanilan toplam dongu (anomali dahil): {int(al['use'].isin(['normal_crossfit','anomaly_test_all_folds']).sum())}")

tmpd = fc.CACHE_DIR / "equiv"
tmpd.mkdir(parents=True, exist_ok=True)
fold_cols = ["win", "session", "session_idx", "frame_id", "t", "loop_id", "gid", "y_true", "y_raw", "s1", "s2", "s0_test"]
new = {
    "folds_state.csv": w[fold_cols].rename(columns={"s1": "s1_fold", "s2": "s2_fold"}),
    "folds_anom.csv": al,
    "split_s0.csv": w[["win", "session", "frame_id", "loop_id", "y_true", "s0_test"]],
}
fold_rows = []
for name, df in new.items():
    p = tmpd / name
    df.to_csv(p, index=False, lineterminator="\n")
    a = p.read_bytes()
    b = (fc.OUT / name).read_bytes()
    byte_eq = a == b
    lf_eq = a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")
    ra, rb = pd.read_csv(p), pd.read_csv(fc.OUT / name)
    val_eq = ra.shape == rb.shape and list(ra.columns) == list(rb.columns) and ra.equals(rb)
    # equals() NaN'lari esit sayar; float'lar tam esit olmali (tolerans yok)
    fold_rows.append({"dosya": name, "bayt_ayni": byte_eq, "bayt_ayni(CRLF->LF)": lf_eq, "deger_ayni": val_eq,
                      "satir": len(ra), "uzak_satir": len(rb)})
    check(f"fold_file_{name}", val_eq and lf_eq, f"bayt={byte_eq} bayt(LF-normal)={lf_eq} deger={val_eq} n={len(ra)}")

# ------------------------------------------------------------------ 3) B2 yeniden egitim (02_state_baselines.py ile ayni)
SEED = 20261104
FEATS = ["cx", "cy", "n_points", "snr_sum", "mean_abs_v", "std_x", "std_y"]
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}
R = fc.win_rows(w)
Y = w["y_true"].to_numpy()


def b2_features(rows):
    cols = []
    t = np.arange(fc.L, dtype=float)
    tc = t - t.mean()
    for f in FEATS:
        v = ft[f].to_numpy(float)
        v = np.where(np.isnan(v), FILL[f], v)[rows]
        cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
    return np.stack(cols, 1).astype(np.float32)


Xb2 = b2_features(R)
b2_rows = []
for f in (0, 1):
    te = w["s1"].to_numpy() == f
    tr_idx, te_idx = np.flatnonzero(~te), np.flatnonzero(te)
    t1 = time.time()
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=3)
    rf.fit(Xb2[tr_idx], Y[tr_idx])
    pr = np.zeros((len(te_idx), 6), np.float32)
    pr[:, rf.classes_.astype(int)] = rf.predict_proba(Xb2[te_idx])
    yp = pr.argmax(1)
    ref = pd.read_csv(fc.oof_path("b2", "S1", f))
    keys_ok = (ref["session"].to_numpy() == w["session"].to_numpy()[te_idx]).all() and \
              (ref["frame_id"].to_numpy() == w["frame_id"].to_numpy()[te_idx]).all() and len(ref) == len(te_idx)
    pref = ref[[f"p{c}" for c in range(6)]].to_numpy(np.float64)
    pmax = float(np.abs(pr.astype(np.float64) - pref).max())
    pmax32 = float(np.abs(pr - pref.astype(np.float32)).max())
    agree = float((yp == ref["y_pred"].to_numpy()).mean())
    nflip = int((yp != ref["y_pred"].to_numpy()).sum())
    m_loc = fc.state_metrics(Y[te_idx], yp)
    m_ref = fc.state_metrics(Y[te_idx], ref["y_pred"].to_numpy())
    P(f"B2 S1 f{f}: n_te={len(te_idx)} anahtar_ayni={keys_ok} y_pred uyumu={agree:.6f} (farkli {nflip}) max|dp|={pmax:.3g} "
      f"(float32-yuvarlanmis {pmax32:.3g}) acc yerel={m_loc['acc']:.6f} uzak={m_ref['acc']:.6f} macroF1 yerel={m_loc['macro_f1']:.6f} "
      f"uzak={m_ref['macro_f1']:.6f}  [{time.time()-t1:.0f}s]")
    b2_rows.append({"kat": f, "n_test": len(te_idx), "anahtarlar_ayni": bool(keys_ok), "y_pred_uyum": agree, "farkli_tahmin": nflip,
                    "max_abs_dp": pmax, "max_abs_dp_f32": pmax32, "acc_yerel": m_loc["acc"], "acc_uzak": m_ref["acc"],
                    "mF1_yerel": m_loc["macro_f1"], "mF1_uzak": m_ref["macro_f1"]})
    check(f"b2_f{f}_keys", keys_ok)
    check(f"b2_f{f}_agreement>=99.9%", agree >= 0.999, f"{agree:.6f} (farkli {nflip}/{len(te_idx)}); max|dp|={pmax:.3g}")

# ------------------------------------------------------------------ rapor
core = ["hm_600_frames_maxdiff_0", "n_frames_16272", "n_windows_16248", "s0_split_3250_12998", "loop_counts_per_session_equal",
        "fold_file_folds_state.csv", "fold_file_folds_anom.csv", "fold_file_split_s0.csv", "b2_f0_keys", "b2_f1_keys",
        "b2_f0_agreement>=99.9%", "b2_f1_agreement>=99.9%", "hm_shape"]
if rebuild_equal is not None:
    core.append("hm_fresh_rebuild_equals_cache")
ALL_OK = all(checks[k][0] for k in core)
identical = all(r["y_pred_uyum"] == 1.0 for r in b2_rows)

H("# A2.0 — Yerel = uzak boru hattı denkliği")
H("")
H(f"`faz_a2/00_equivalence.py` tarafından üretilir (DESIGN_A2.md A2.0; sapma yok). Tüm sayılar stdout'a da yazılır. Süre: {time.time()-t0:.0f} s.")
H("")
H(f"## SONUÇ: **{'PASS' if ALL_OK else 'FAIL'}**" + ("" if ALL_OK else " — A2 durur, kullanıcıya bildirilir."))
H("")
H(f"Ortam: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__} "
  f"(requirements.txt ile aynı sürümler). Yerel heatmap önbelleği: `{fc._hm_path().name}` (CACHE_DIR proje klasörü dışında).")
H("")
H("| denetim | sonuç | ayrıntı |")
H("|---|---|---|")
for k, (ok, d) in checks.items():
    H(f"| {k} | {'PASS' if ok else 'FAIL'} | {d} |")
H("")
H("## Heatmap")
H(f"- Yerel önbellek ({hm.shape[0]}, 2, 48, 48) float32; orijinal `build_heatmap` döngüsüyle 600 rastgele karede (RandomState(0), `faz_a/00_folds.py` ile aynı seçim) "
  f"**maks |fark| = {maxdiff:.3g}**."
  + (f" Önbellek zaten vardı; taze yeniden inşa ile birebir eşit: {rebuild_equal}." if cache_existed else " Önbellek bu çalıştırmada sıfırdan kuruldu."))
H("")
H("## Sayılar ve katlar")
H(f"- Kare **{len(ft)}**, pencere **{len(w)}** ({len(ft)} − {len(sess)}×4), S0 test **{n_te0}** / eğitim **{len(w)-n_te0}**; s02'de katlanan tam döngü: {n_cyc}.")
H("- Döngü (loop_id grubu) sayısı, oturum başına (Faz A `folds_state.csv` ile aynı): " + ", ".join(f"{k.split('_')[1]}={v}" for k, v in loops_local.items()) + ".")
H("")
H(common.md_table(pd.DataFrame(fold_rows)))
H("")
H("Not: yerel CSV'ler `lineterminator=\"\n\"` ile yazılır (Faz A uzak Linux'ta, LF); bayt eşitliği ham ve ayrıca CRLF→LF normalize edilerek raporlanır. "
  "Geçme ölçütü: normalize bayt eşitliği **ve** değer eşitliği (float'lar toleranssız).")
H("")
H("## B2 (RandomForest) yeniden eğitimi — S1 kat 0 ve 1")
H("`02_state_baselines.py` ile aynı: 7 kare istatistiği × (son, ortalama, eğim) = 21 özellik, RF(300, class_weight=balanced, random_state=20261104; n_jobs=3 — sonuç n_jobs'tan bağımsız), "
  "eğitim = S1 katı ≠ f tüm pencereler.")
H("")
H(common.md_table(pd.DataFrame(b2_rows), "{:.6g}"))
H("")
H(f"- y_pred özdeş mi (iki katta da %100): **{identical}**. Olasılıklar: `max |Δp|` yukarıda (OOF dosyası float32'yi metin olarak yazdığından 1e-7 mertebesi yuvarlama beklenir).")
H("")
H("Geçme ölçütü: heatmap farkı 0; sayılar ve katlar özdeş; B2 tahmin uyumu ≥ %99,9 (%100 değilse oran raporlanır).")
H("")
H(MARK)
H("")
txt = "\n".join(LINES) + "\n"
p = OUT2 / "00_equivalence.md"
p.write_text(txt, encoding="utf-8")
P(f"yazildi: {p}  SONUC={'PASS' if ALL_OK else 'FAIL'}  [{time.time()-t0:.0f}s]")
sys.exit(0 if ALL_OK else 1)

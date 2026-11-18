"""Faz A2 / A2.0b - a2_cycles.py oz-sinamasi: Faz A 04 sonuclarini Faz A rolleriyle birebir yeniden uretir.

- her (kaynak x segmentasyon x f x rol) dongusu icin D0/D1/D2 ve p0..p3: reports/faz_a/04_cycle_scores.csv.gz ile max |fark| <= 1e-9,
  ayni dongu anahtarlari ve sayilari; 04_cycle_counts.csv ile sayim tablosu
- AUROC / FAR tablolari: 04_anomaly_table.csv (D0..D3, tum hucreler) ve 04_anomaly_perfold.csv ile karsilastirma;
  B2xpred, B2xtof, ToFxtof, CNN-GRUxpred hucreleri ayrica yazdirilir
- yeni yardimcilarin birim sinamalari (smoothed p, cross-conformal)
Sonuc reports/faz_a2/00_equivalence.md'nin sonuna (SELFTEST isaretinden sonra) eklenir; tekrar kosulursa o bolum yenilenir.
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

import a2_cycles as ac
import fa_common as fc
from common import md_table

OUT2 = ROOT / "reports" / "faz_a2"
MARK = "<!-- SELFTEST-BEGIN -->"
TOL = 1e-9
t0 = time.time()
LINES = []
checks = {}


def P(*a):
    print(*a, flush=True)


def H(s=""):
    LINES.append(s)
    P(s)


def check(name, ok, detail=""):
    checks[name] = (bool(ok), detail)
    P(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ------------------------------------------------------------------ 1) dongu skorlari
S, CNT, NCAL = ac.score_all(verbose=True)
ref = pd.read_csv(fc.OUT / "04_cycle_scores.csv.gz")
KEY = ["state_source", "segmentation", "session", "cycle_idx", "start_frame_id", "f", "role"]
P(f"[{time.time()-t0:.0f}s] yerel {len(S)} satir, Faz A {len(ref)} satir")
check("rows_equal", len(S) == len(ref), f"yerel {len(S)} / Faz A {len(ref)}")
check("keys_unique", (not S.duplicated(KEY).any()) and (not ref.duplicated(KEY).any()))
ks = lambda d: d[KEY].sort_values(KEY).reset_index(drop=True)
check("keys_identical", ks(S).equals(ks(ref)))
cnt_cell = lambda d: d.groupby(["state_source", "segmentation", "role"]).size()
check("counts_per_cell_role_identical", cnt_cell(S).equals(cnt_cell(ref)))
M = S.merge(ref, on=KEY, how="outer", suffixes=("", "_ref"), validate="1:1", indicator=True)
check("outer_merge_all_matched", (M["_merge"] == "both").all(), f"{M['_merge'].value_counts().to_dict()}")
M = M[M["_merge"] == "both"]
maxd = {}
for c in ["D0", "D1", "D2", "p0", "p1", "p2", "p3", "dur_s", "d2_nref", "n_visits", "end_frame_id", "anchor_loop", "tof_fold"]:
    a, b = M[c].to_numpy(float), M[c + "_ref"].to_numpy(float)
    nan_ok = bool((np.isnan(a) == np.isnan(b)).all())
    d = np.nanmax(np.abs(a - b)) if (~np.isnan(a) & ~np.isnan(b)).any() else 0.0
    maxd[c] = (float(d), nan_ok)
for c in ["atype", "visit_seq", "natural_dev"]:
    fx = lambda x: x.astype(object).where(x.notna(), "").astype(str)      # bos dizi: CSV'de NaN okunur
    maxd[c] = (0.0 if (fx(M[c]) == fx(M[c + "_ref"])).all() else 1.0, True)
worst = max(v[0] for v in maxd.values())
check("scores_maxabsdiff<=1e-9", worst <= TOL and all(v[1] for v in maxd.values()),
      f"max |fark| = {worst:.3g}; NaN deseni ayni: {all(v[1] for v in maxd.values())}")
P("sutun bazinda max |fark|:", {k: f"{v[0]:.2g}" for k, v in maxd.items()})
nt = int((S["role"] == "test").sum())
P(f"test satiri {nt}; p dolu satir {int(S['p3'].notna().sum())} (Faz A {int(ref['p3'].notna().sum())})")

# sayim tablosu
refc = pd.read_csv(fc.OUT / "04_cycle_counts.csv")
ck = ["state_source", "segmentation", "session"]
cols = [c for c in refc.columns if c not in ck]
mc = CNT.merge(refc, on=ck, suffixes=("", "_ref"), validate="1:1")
cnt_ok = len(mc) == len(refc) == len(CNT) and all(
    np.allclose(mc[c].to_numpy(float), mc[c + "_ref"].to_numpy(float), equal_nan=True) if c != "atype" else (mc[c] == mc[c + "_ref"]).all()
    for c in cols)
check("cycle_counts_table_identical", cnt_ok, f"{len(CNT)} satir")

# ------------------------------------------------------------------ 2) AUROC / FAR tablolari
PF, SM = ac.evaluate(S)
TAB = ac.summarize(PF, SM)
reft = pd.read_csv(fc.OUT / "04_anomaly_table.csv")
refp = pd.read_csv(fc.OUT / "04_anomaly_perfold.csv")
reft = reft[reft["variant"].isin(["D0", "D1", "D2", "D3"])]
refp = refp[refp["variant"].isin(["D0", "D1", "D2", "D3"])]
tk = ["state_source", "segmentation", "variant", "atype"]
mt = TAB.merge(reft, on=tk, how="outer", suffixes=("", "_ref"), validate="1:1", indicator=True)
check("anomaly_table_rows_matched", (mt["_merge"] == "both").all() and len(mt) == len(reft) == len(TAB),
      f"yerel {len(TAB)} / Faz A(D0..D3) {len(reft)}")
shared = [c for c in TAB.columns if c not in tk and c + "_ref" in mt.columns]
tmax = 0.0
tbad = []
for c in shared:
    a, b = mt[c].to_numpy(), mt[c + "_ref"].to_numpy()
    if a.dtype == bool or b.dtype == bool or a.dtype == object:
        ok = bool((a.astype(bool) == b.astype(bool)).all())
        if not ok:
            tbad.append(c)
        continue
    a, b = a.astype(float), b.astype(float)
    if not (np.isnan(a) == np.isnan(b)).all():
        tbad.append(c + "(NaN)")
        continue
    d = float(np.nanmax(np.abs(a - b))) if (~np.isnan(a)).any() else 0.0
    tmax = max(tmax, d)
    if d > TOL:
        tbad.append(f"{c}:{d:.2g}")
check("anomaly_table_all_cells<=1e-9", not tbad and tmax <= TOL, f"{len(shared)} sutun, max |fark| = {tmax:.3g}; sorunlu: {tbad}")
pk = ["state_source", "segmentation", "variant", "atype", "f"]
mp = PF.merge(refp, on=pk, how="outer", suffixes=("", "_ref"), validate="1:1", indicator=True)
pmax = max(float(np.nanmax(np.abs(mp[c].to_numpy(float) - mp[c + "_ref"].to_numpy(float)))) for c in ["auroc", "auprc", "n_neg", "n_pos", "prevalence"])
nan_ok = all(bool((np.isnan(mp[c].to_numpy(float)) == np.isnan(mp[c + "_ref"].to_numpy(float))).all()) for c in ["auroc", "auprc"])
check("anomaly_perfold_all_rows<=1e-9", (mp["_merge"] == "both").all() and pmax <= TOL and nan_ok,
      f"{len(PF)} satir yerel / {len(refp)} Faz A; max |fark| = {pmax:.3g}")

# okunabilir ozet: 4 hucre
CELLS = [("b2", "pred"), ("b2", "tof"), ("tof", "tof"), ("cnngru", "pred")]
NM = {"b2": "B2", "tof": "ToF oracle", "cnngru": "CNN-GRU"}
rows = []
for sr, sg in CELLS:
    for vn in ("D3", "D1", "D2"):
        a = TAB[(TAB.state_source == sr) & (TAB.segmentation == sg) & (TAB.variant == vn)].set_index("atype")
        b = reft[(reft.state_source == sr) & (reft.segmentation == sg) & (reft.variant == vn)].set_index("atype")
        r = {"hücre": f"{NM[sr]} × {sg}", "skor": vn}
        for an in ("seq(135+145)", "tempo(fast+slow)", "tum anomali"):
            r[f"AUROC {an}"] = f"{a.at[an, 'auroc_mean']:.4f} (FA {b.at[an, 'auroc_mean']:.4f})"
        r["FAR@0.10"] = f"{a.at['tum anomali', 'far_a10']:.4f} (FA {b.at['tum anomali', 'far_a10']:.4f})"
        r["FAR@0.05"] = f"{a.at['tum anomali', 'far_a05']:.4f} (FA {b.at['tum anomali', 'far_a05']:.4f})"
        r["tespit@0.10 tüm"] = f"{a.at['tum anomali', 'det_a10']:.4f} (FA {b.at['tum anomali', 'det_a10']:.4f})"
        rows.append(r)
        assert abs(a.at["tum anomali", "far_a10"] - b.at["tum anomali", "far_a10"]) <= TOL
        assert abs(a.at["tum anomali", "auroc_mean"] - b.at["tum anomali", "auroc_mean"]) <= TOL
CELLTAB = pd.DataFrame(rows)
P(CELLTAB.to_string())

# ------------------------------------------------------------------ 3) yeni yardimcilar: birim sinamalar
rng = np.random.RandomState(1)
cal = rng.randint(0, 6, 30).astype(float)
s = rng.randint(0, 7, 40).astype(float)
check("smoothed_p_u1_equals_standard", np.allclose(ac.conf_p_smoothed(cal, s, 1.0), ac.conf_p(cal, s), atol=0, rtol=0))
dr = ac.smoothed_draws(cal, s, 200, seed=7)
dr2 = ac.smoothed_draws(cal, s, 200, seed=7)
std = ac.conf_p(cal, s)
check("smoothed_draws_deterministic_and_bounded", np.array_equal(dr, dr2) and (dr <= std + 1e-12).all() and (dr >= 1.0 / (len(cal) + 1) * 0).all()
      and dr.shape == (200, 40))
# smoothed p, degistirilebilir (exchangeable) verilerde cal kumesi uzerinden ortalamada nominal: 4000 bagimsiz (cal, test) cifti
rr_ = np.random.RandomState(3)
ps = []
for _ in range(4000):
    cal_u = rr_.randint(0, 4, 20).astype(float)
    ps.append(ac.smoothed_draws(cal_u, rr_.randint(0, 4, 1).astype(float), 1, seed=int(rr_.randint(1 << 30)))[0, 0])
ps = np.array(ps)
cov = [float((ps <= a).mean()) for a in (0.05, 0.10, 0.25)]
check("smoothed_p_nominal_coverage", all(abs(c - a) < 0.02 for c, a in zip(cov, (0.05, 0.10, 0.25))), f"{cov}")
check("d3_def", np.allclose(ac.d3([0.04, 0.3, 0.9], [0.2, 0.01, 0.8]), [0.08, 0.02, 1.0]))

# cross-conformal
LAB = ac._default_labels()
C, _ = ac.build_cycles("b2", "pred", LAB["b2"])
Ct, _ = ac.build_cycles("tof", "tof", LAB["tof"])
cf = ac.cross_scores(Ct, 0)
nn = Ct[(Ct.atype == "normal")].groupby("fold").size().to_dict()
exp_ncal = sum(nn[k] for k in range(1, 5))
check("cross_n_cal_equals_remaining_folds", cf["n_cal"] == exp_ncal and cf["folds_k"] == [1, 2, 3, 4],
      f"n_cal={cf['n_cal']} (beklenen {exp_ncal}); egitim/model={cf['n_train']}; kat sayilari={nn}")
nte = int(((Ct.atype == "normal") & (Ct.fold == 0)).sum() + (Ct.atype != "normal").sum())
check("cross_test_matrix_shape", cf["S"]["D1"].shape == (nte, 4) and len(cf["test"]) == nte, f"{cf['S']['D1'].shape}")
p1 = ac.cross_p(cf, "D1")
check("cross_p_range", (p1 >= 1.0 / (cf["n_cal"] + 1) - 1e-12).all() and (p1 <= 1.0 + 1e-12).all(), f"min p={p1.min():.4f}")
# her model k icin 3 katli egitim parametreleri == fit_params(ayni liste) ve kat-disi skor ayni (bagimsiz yeniden hesap)
k = cf["folds_k"][0]
tr = [c for c in Ct.to_dict("records") if c["atype"] == "normal" and c["fold"] in cf["folds_k"] and c["fold"] != k]
par = ac.fit_params(tr)
ck_ = [c for c in Ct.to_dict("records") if c["atype"] == "normal" and c["fold"] == k]
chk = ac.score_cycles(ck_, par)["D2"].to_numpy()
check("cross_fold_k_scores_reproducible", np.array_equal(chk, cf["cal"]["D2"][0]))
# u=1 standart; u=None standart
check("cross_p_smoothed_u1_equals_standard", np.allclose(ac.cross_p(cf, "D1", u=1.0), p1, atol=1e-12))
cd = ac.cross_p_draws(cf, "D1", 50, seed=5)
check("cross_p_draws_le_standard", (cd <= p1[None, :] + 1e-12).all() and cd.shape == (50, nte))
# keyfi liste ile Faz A rolleriyle ayni sonuc (fit_params -> score)
ro = fc.anom_roles(0)
cl = C.to_dict("records")
tr0 = [c for c in cl if c["atype"] == "normal" and c["fold"] in ro["train"]]
sc0 = ac.score_cycles(tr0, ac.fit_params(tr0))
rr = ref[(ref.state_source == "b2") & (ref.segmentation == "pred") & (ref.f == 0) & (ref.role == "train")]
mm = sc0.merge(rr, on=["session", "cycle_idx", "start_frame_id"], suffixes=("", "_r"), validate="1:1")
check("fit_params_arbitrary_list_reproduces_train_rows", len(mm) == len(rr) == len(sc0) and
      float(np.abs(mm["D1"] - mm["D1_r"]).max()) <= TOL and float(np.abs(mm["D2"] - mm["D2_r"]).max()) <= TOL)

# ------------------------------------------------------------------ rapor (00_equivalence.md'ye ekle)
ALL_OK = all(v[0] for v in checks.values())
H("## A2.0b — `a2_cycles.py` öz-sınaması (Faz A rolleriyle birebir yeniden üretim)")
H("")
H(f"`faz_a2/00b_cycles_selftest.py` tarafından üretilir. Süre: {time.time()-t0:.0f} s.")
H("")
H(f"### SONUÇ: **{'PASS' if ALL_OK else 'FAIL'}**")
H("")
H("| denetim | sonuç | ayrıntı |")
H("|---|---|---|")
for k_, (ok, d) in checks.items():
    H(f"| {k_} | {'PASS' if ok else 'FAIL'} | {d} |")
H("")
H(f"- Yerel yeniden hesap: **{len(S)}** (dönüş, f, rol) satırı = Faz A `04_cycle_scores.csv.gz` ({len(ref)}); D0/D1/D2/p0..p3 için **max |fark| = {worst:.3g}** (eşik 1e-9); "
  f"`04_anomaly_table.csv` (D0–D3, {len(shared)} sütun × {len(TAB)} satır) max |fark| = {tmax:.3g}; `04_anomaly_perfold.csv` max |fark| = {pmax:.3g}.")
H("- Sütun bazında max |fark|: " + ", ".join(f"{k_}={v[0]:.2g}" for k_, v in maxd.items()) + ".")
H("")
H("Hücre özeti (yerel; parantez içi Faz A değeri):")
H("")
H(md_table(CELLTAB))
H("")
H(f"Faz A rolleri: test = f; kalibrasyon = (f+1)%5, (f+2)%5; eğitim = (f+3)%5, (f+4)%5 (n_cal min/ort/maks = {int(NCAL['n_cal'].min())}/{NCAL['n_cal'].mean():.1f}/{int(NCAL['n_cal'].max())}).")
H("")
txt = "\n".join(LINES) + "\n"
p = OUT2 / "00_equivalence.md"
base = p.read_text(encoding="utf-8") if p.exists() else "# A2.0 — Yerel = uzak boru hattı denkliği\n\n(00_equivalence.py henüz koşulmadı)\n\n" + MARK + "\n\n"
if MARK not in base:
    base = base.rstrip("\n") + "\n\n" + MARK + "\n\n"
head = base.split(MARK)[0] + MARK + "\n\n"
p.write_text(head + txt, encoding="utf-8")
P(f"eklendi: {p}  SONUC={'PASS' if ALL_OK else 'FAIL'}  [{time.time()-t0:.0f}s]")
sys.exit(0 if ALL_OK else 1)

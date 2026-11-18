"""Faz C / C.1 - Kucuk ormanin C'ye aktarimi ve derlenmis gercek boyutu. DESIGN_C.md C.1'e birebir.
Modeller: 10 agac x None (B.1 secimi), 25 agac x None; B2 (300 x None) yalniz aritmetik. 5 kat + tam veri modeli.
Kodlamalar: (a) float32 olasilik/yumusak oy, (b) uint8 nicemli olasilik/yumusak oy, (c) sert oy argmax uint8.
Kapi: (a)+double toplama, C cikarimi == sklearn predict, tum S1 test pencereleri + tam veri modeli."""
import os, sys, time, json, subprocess, shutil, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe")); sys.path.insert(0, str(ROOT / "faz_a")); sys.path.insert(0, str(ROOT / "faz_a2"))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import sklearn
import fa_common as fc
import a2_cycles as ac

SEED = 20261104
NJOBS = 3
FEATS = ["cx", "cy", "n_points", "snr_sum", "mean_abs_v", "std_x", "std_y"]
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}
GCC = os.environ.get("GCC") or shutil.which("gcc") or "gcc"
SIZE = os.environ.get("SIZE_TOOL") or shutil.which("size") or "size"
CDIR = ROOT / "faz_c" / "c"
WORK = CDIR / "_work"
REP = ROOT / "reports" / "faz_c"
OOFD = ROOT / "faz_c" / "oof"
for d in (WORK, REP, OOFD):
    d.mkdir(parents=True, exist_ok=True)
KB192, KB256, KB512 = 192 * 1024, 256 * 1024, 512 * 1024
ENCS = ["a", "b", "c"]
ENC_NAME = {"a": "(a) float32 olasilik, yumusak oy", "b": "(b) uint8 nicemli olasilik, yumusak oy", "c": "(c) sert oy, yaprak argmax uint8"}
TCRIT = stats.t.ppf(0.975, 4)

print(f"sklearn {sklearn.__version__} numpy {np.__version__} pandas {pd.__version__}", flush=True)
ft = fc.frame_table()
W = fc.windows()
R = fc.win_rows(W)
Y = W["y_true"].to_numpy()
S1 = W["s1"].to_numpy()


def b2_features(rows):          # faz_b/01_tiny_forests.py ile birebir ayni
    names, cols = [], []
    t = np.arange(fc.L, dtype=float); tc = t - t.mean()
    for f in FEATS:
        v = ft[f].to_numpy(float)
        v = np.where(np.isnan(v), FILL[f], v)[rows]
        cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
        names += [f + "_last", f + "_mean", f + "_slope"]
    return np.stack(cols, 1).astype(np.float32), names


X, NAMES = b2_features(R)
assert X.dtype == np.float32 and X.shape[1] == 21 and len(X) == 16248
X = np.ascontiguousarray(X)
X.tofile(WORK / "X.bin")
N = len(X)


def fit_cfg(nt, tr_idx):
    rf = RandomForestClassifier(n_estimators=nt, max_depth=None, class_weight="balanced", random_state=SEED, n_jobs=NJOBS)
    rf.fit(X[tr_idx], Y[tr_idx])
    return rf


def tci(d):
    d = np.asarray(d, float)
    m, s = d.mean(), d.std(ddof=1)
    h = TCRIT * s / np.sqrt(len(d))
    return m, m - h, m + h


# ----------------------------------------------------------------------------- referanslar (Faz A B2, Faz B B.1)
b2_oof = fc.load_oof("b2", "S1")
mm = W[["session", "frame_id", "s1"]].merge(b2_oof[["session", "frame_id", "y_pred", "y_true"]], on=["session", "frame_id"], how="left")
B2_F1 = np.array([fc.state_metrics(mm.loc[mm.s1 == f, "y_true"].to_numpy(), mm.loc[mm.s1 == f, "y_pred"].to_numpy())["macro_f1"] for f in range(5)])
print(f"[ref] Faz A B2 macro-F1 per fold {np.round(B2_F1, 4)} mean={B2_F1.mean():.4f}", flush=True)
FB = pd.read_csv(ROOT / "reports" / "faz_b" / "01_tiny_forests.csv").set_index("config")


# ----------------------------------------------------------------------------- dışa aktarım
def export_forest(rf):
    """sklearn RF -> yapi-dizileri (struct-of-arrays). Dugumler agac basina on-sirada; cocuk indeksleri agac-yerel."""
    assert list(rf.classes_.astype(int)) == list(range(6)), rf.classes_
    feat, thr, left, right, leafp = [], [], [], [], []
    noff, loff = [], []
    nn = nl = 0
    max_value_dev = 0.0
    for e in rf.estimators_:
        t = e.tree_
        n = t.node_count
        cl, cr = t.children_left, t.children_right
        isleaf = cl == -1
        v = t.value[:, 0, :6].astype(np.float64)
        s = v.sum(1)
        max_value_dev = max(max_value_dev, float(np.abs(s - 1.0).max()))
        p = v / np.where(s == 0, 1.0, s)[:, None]         # sklearn predict_proba da agac basina normalize eder
        lid = np.cumsum(isleaf) - 1                        # agac-yerel yaprak indeksi
        th = t.threshold.astype(np.float64).copy()
        t32 = th.astype(np.float32)
        up = t32.astype(np.float64) > th
        t32[up] = np.nextafter(t32[up], np.float32(-np.inf))
        t32[isleaf] = 0.0
        assert not (t32.astype(np.float64)[~isleaf] > th[~isleaf]).any()
        f_ = np.where(isleaf, 255, t.feature).astype(np.int64)
        assert f_[~isleaf].max() < 21
        l_ = np.where(isleaf, lid, cl)
        r_ = np.where(isleaf, 0, cr)
        feat.append(f_.astype(np.uint8)); thr.append(t32); left.append(l_); right.append(r_)
        leafp.append(p[isleaf])
        noff.append(nn); loff.append(nl)
        nn += n; nl += int(isleaf.sum())
    A = dict(feat=np.concatenate(feat), thr=np.concatenate(thr), left=np.concatenate(left), right=np.concatenate(right),
             leafp=np.concatenate(leafp), noff=np.array(noff), loff=np.array(loff), T=len(rf.estimators_), nodes=nn, leaves=nl,
             max_tree_nodes=max(e.tree_.node_count for e in rf.estimators_), value_dev=max_value_dev)
    A["wchild"] = 2 if A["max_tree_nodes"] <= 32767 else 4
    A["offbytes"] = 2 if nn <= 65535 else 4
    A["leafq"] = np.floor(A["leafp"] * 255.0 + 0.5).astype(np.uint8)       # (b) yuvarlama: yari yukari
    A["leafc"] = A["leafp"].argmax(1).astype(np.uint8)                      # (c) esitlikte kucuk sinif
    A["n_impure"] = int((A["leafp"].max(1) < 1.0 - 1e-12).sum())
    return A


def arith_bytes(nodes, leaves, T, enc, wchild, offbytes, part=False):
    """Dizi baytlari toplami (dolgu yok): dugum basina 1 (feature) + 4 (thr) + 2*wchild; yaprak tablosu; agac ofsetleri."""
    leafb = {"a": 24, "b": 6, "c": 1}[enc]
    nodeb = nodes * (1 + 4 + 2 * wchild)
    return nodeb + leaves * leafb + T * 2 * offbytes


def emit_arr(name, ctype, vals, per_line=24, fmt=None):
    out = [f"static const {ctype} {name}[{len(vals)}] = {{"]
    toks = [fmt(v) for v in vals] if fmt else [str(int(v)) for v in vals]
    for i in range(0, len(toks), per_line):
        out.append(",".join(toks[i:i + per_line]) + ",")
    out.append("};")
    return "\n".join(out)


def f32lit(v):
    t = "%.9g" % float(v)
    if not any(ch in t for ch in ".eE"):
        t += ".0"
    return t + "f"


def gen_c(A, enc, path):
    wt = "int16_t" if A["wchild"] == 2 else "int32_t"
    ot = "uint16_t" if A["offbytes"] == 2 else "uint32_t"
    assert A["T"] <= 255
    s = ['#include <stdint.h>', '#include "forest_api.h"', f'#define NTREES {A["T"]}', '#define LEAF 255', ""]
    s.append(emit_arr("NOFF", ot, A["noff"]))
    s.append(emit_arr("LOFF", ot, A["loff"]))
    s.append(emit_arr("FEAT", "uint8_t", A["feat"], 40))
    s.append(emit_arr("THR", "float", A["thr"], 12, f32lit))
    s.append(emit_arr("LEFT", wt, A["left"], 20))
    s.append(emit_arr("RIGHT", wt, A["right"], 20))
    if enc == "a":
        s.append(emit_arr("LEAFV", "float", A["leafp"].astype(np.float32).ravel(), 12, f32lit))
    elif enc == "b":
        s.append(emit_arr("LEAFV", "uint8_t", A["leafq"].ravel(), 36))
    else:
        s.append(emit_arr("LEAFV", "uint8_t", A["leafc"], 40))
    s.append(f"""
static unsigned leaf_of(const float *x, unsigned t) {{
    unsigned base = NOFF[t], n = 0;
    for (;;) {{
        unsigned g = base + n;
        uint8_t f = FEAT[g];
        if (f == LEAF) return LOFF[t] + (unsigned)LEFT[g];
        n = (x[f] <= THR[g]) ? (unsigned)LEFT[g] : (unsigned)RIGHT[g];
    }}
}}
""")
    if enc == "a":
        s.append("""int forest_predict(const float *x) {
    float acc[6] = {0, 0, 0, 0, 0, 0};
    for (unsigned t = 0; t < NTREES; t++) { unsigned l = leaf_of(x, t) * 6; for (int c = 0; c < 6; c++) acc[c] += LEAFV[l + c]; }
    int b = 0; for (int c = 1; c < 6; c++) if (acc[c] > acc[b]) b = c;
    return b;
}
#ifdef WITH_SUM
void forest_sum(const float *x, double *acc) {
    for (int c = 0; c < 6; c++) acc[c] = 0.0;
    for (unsigned t = 0; t < NTREES; t++) { unsigned l = leaf_of(x, t) * 6; for (int c = 0; c < 6; c++) acc[c] += (double)LEAFV[l + c]; }
}
#endif
""")
    elif enc == "b":
        s.append("""int forest_predict(const float *x) {
    uint16_t acc[6] = {0, 0, 0, 0, 0, 0};
    for (unsigned t = 0; t < NTREES; t++) { unsigned l = leaf_of(x, t) * 6; for (int c = 0; c < 6; c++) acc[c] += LEAFV[l + c]; }
    int b = 0; for (int c = 1; c < 6; c++) if (acc[c] > acc[b]) b = c;
    return b;
}
#ifdef WITH_SUM
void forest_sum(const float *x, double *acc) {
    for (int c = 0; c < 6; c++) acc[c] = 0.0;
    for (unsigned t = 0; t < NTREES; t++) { unsigned l = leaf_of(x, t) * 6; for (int c = 0; c < 6; c++) acc[c] += (double)LEAFV[l + c]; }
}
#endif
""")
    else:
        s.append("""int forest_predict(const float *x) {
    uint8_t acc[6] = {0, 0, 0, 0, 0, 0};
    for (unsigned t = 0; t < NTREES; t++) acc[LEAFV[leaf_of(x, t)]]++;
    int b = 0; for (int c = 1; c < 6; c++) if (acc[c] > acc[b]) b = c;
    return b;
}
#ifdef WITH_SUM
void forest_sum(const float *x, double *acc) {
    for (int c = 0; c < 6; c++) acc[c] = 0.0;
    for (unsigned t = 0; t < NTREES; t++) acc[LEAFV[leaf_of(x, t)]] += 1.0;
}
#endif
""")
    Path(path).write_text("\n".join(s), encoding="ascii")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def sections(obj):
    out = run([SIZE, "-A", str(obj)])
    d = {}
    for ln in out.splitlines():
        m = re.match(r"^(\S+)\s+(\d+)\s+(\d+)\s*$", ln)
        if m:
            d[m.group(1)] = int(m.group(2))
    return d


def build_and_run(A, enc, tag, keep_src=False):
    """C kaynagi uret; (1) -Os -c nesnesi (WITH_SUM yok) ile boyut, (2) -Os -DWITH_SUM + surucu ile calistir. Donus: sizes, sums, preds."""
    src = (CDIR / f"forest_{tag}_{enc}.c") if keep_src else (WORK / f"forest_{tag}_{enc}.c")
    gen_c(A, enc, src)
    obj = WORK / f"{tag}_{enc}.o"
    run([GCC, "-Os", "-c", "-I", str(CDIR), str(src), "-o", str(obj)])
    sec = sections(obj)
    rd = sum(v for k, v in sec.items() if k == ".rdata")
    rd_other = sum(v for k, v in sec.items() if k.startswith(".rdata") and k != ".rdata")
    sz = dict(rdata=rd, rdata_other=rd_other, text=sec.get(".text", 0), data=sec.get(".data", 0), bss=sec.get(".bss", 0),
              all_sections=json.dumps(sec))
    exe = WORK / f"{tag}_{enc}.exe"
    run([GCC, "-Os", "-DWITH_SUM", "-I", str(CDIR), str(src), str(CDIR / "driver.c"), "-o", str(exe)])
    base = WORK / f"out_{tag}_{enc}"
    run([str(exe), str(WORK / "X.bin"), str(N), str(base)])
    pred = np.fromfile(str(base) + ".pred", dtype=np.uint8).reshape(N, 2)
    sums = np.fromfile(str(base) + ".sum", dtype=np.float64).reshape(N, 6)
    for suf in (".pred", ".sum"):
        os.remove(str(base) + suf)
    exe.unlink()
    if not keep_src:
        src.unlink()
    return sz, sums, pred


def np_forest(A, enc):
    """Bagimsiz numpy yeniden uygulamasi: ayni dizilerden toplamlar (double)."""
    acc = np.zeros((N, 6), np.float64)
    for t in range(A["T"]):
        b = A["noff"][t]; lo = A["loff"][t]
        n = np.zeros(N, np.int64)
        done = np.zeros(N, bool)
        leaf = np.zeros(N, np.int64)
        for _ in range(200):
            g = b + n
            f = A["feat"][g]
            isl = f == 255
            new = isl & ~done
            leaf[new] = lo + A["left"][g][new]
            done |= isl
            if done.all():
                break
            fa = np.where(isl, 0, f).astype(np.int64)
            go_left = X[np.arange(N), fa] <= A["thr"][g]
            nxt = np.where(go_left, A["left"][g], A["right"][g])
            n = np.where(done, n, nxt)
        assert done.all()
        if enc == "a":
            acc += A["leafp"].astype(np.float32).astype(np.float64)[leaf]
        elif enc == "b":
            acc += A["leafq"].astype(np.float64)[leaf]
        else:
            acc[np.arange(N), A["leafc"][leaf]] += 1.0
    return acc


def pipeline(tag, per_fold_pred, per_fold_proba):
    """Faz B 01_tiny_forests ile ayni: OOF -> pred segmentasyonu -> D3 standart conformal; 'tum anomali'."""
    parts = []
    for f in range(5):
        te = np.flatnonzero(S1 == f)
        wins = W.iloc[te]
        df = pd.DataFrame({"session": wins["session"].to_numpy(), "frame_id": wins["frame_id"].to_numpy(), "t": wins["t"].to_numpy(),
                           "loop_id": wins["loop_id"].to_numpy(), "fold": int(f), "y_true": wins["y_true"].to_numpy(),
                           "y_raw": wins["y_raw"].to_numpy(), "y_pred": per_fold_pred[f]})
        for c in range(6):
            df[f"p{c}"] = per_fold_proba[f][:, c]
        df = df[fc.OOF_COLS]
        df.to_csv(OOFD / f"{tag}__S1__f{f}.csv.gz", index=False, compression="gzip")
        parts.append(df)
    oof = pd.concat(parts, ignore_index=True)
    labs = ac.labels_from_oof(oof, strict=True)
    assert labs is not None, tag
    C, cnt = ac.build_cycles(tag, "pred", labs)
    S, _ = ac.score_group(C, verbose=False)
    PF, SM = ac.evaluate(S)
    TAB = ac.summarize(PF, SM)
    d3 = TAB[(TAB.variant == "D3")].set_index("atype")
    a = d3.loc["tum anomali"]
    return dict(auroc=float(a.auroc_mean), auroc_sd=float(a.auroc_sd), far10=float(a.far_a10), det10=float(a.det_a10),
                auroc_seq=float(d3.loc["seq(135+145)", "auroc_mean"]), auroc_tempo=float(d3.loc["tempo(fast+slow)", "auroc_mean"]),
                n_cycles=int(len(C)))


# ----------------------------------------------------------------------------- ana dongu
MODELS = {10: "tf_n10_dN", 25: "tf_n25_dN"}
size_rows, fold_rows, gate_rows, mism_rows, res_rows, ref_rows = [], [], [], [], [], []
VAL_DEV = [0.0]
PUR = []
t00 = time.time()
for nt, ref_cfg in MODELS.items():
    mtag = f"m{nt}"
    print(f"\n===== n_trees={nt} =====", flush=True)
    units = {}
    for f in range(5):
        rf = fit_cfg(nt, np.flatnonzero(S1 != f))
        units[f"f{f}"] = (rf, export_forest(rf))
    rf_full = fit_cfg(nt, np.arange(N))
    units["full"] = (rf_full, export_forest(rf_full))
    # --- kapi 0: Faz B B.1 yeniden uretimi
    nodes_f = [units[f"f{f}"][1]["nodes"] for f in range(5)]
    f1_sk = []
    sk_pred = {}
    sk_proba = {}
    for f in range(5):
        te = np.flatnonzero(S1 == f)
        rf = units[f"f{f}"][0]
        sk_proba[f] = rf.predict_proba(X[te])
        sk_pred[f] = rf.classes_[sk_proba[f].argmax(1)].astype(int)
        assert (sk_pred[f] == rf.predict(X[te])).all()
        f1_sk.append(fc.state_metrics(Y[te], sk_pred[f])["macro_f1"])
    f1_sk = np.array(f1_sk)
    sk_full_proba = rf_full.predict_proba(X)
    sk_full_pred = rf_full.predict(X).astype(int)
    r0 = FB.loc[ref_cfg]
    ref_rows.append(dict(model=ref_cfg, metric="macro_f1_mean", ours=f1_sk.mean(), faz_b=r0.f1_mean))
    ref_rows.append(dict(model=ref_cfg, metric="nodes_fold_mean", ours=float(np.mean(nodes_f)), faz_b=r0.nodes_mean))
    ref_rows.append(dict(model=ref_cfg, metric="nodes_full", ours=units["full"][1]["nodes"], faz_b=r0.nodes_full))
    print(f"[kapi0] {ref_cfg}: mF1 {f1_sk.mean():.6f} (Faz B {r0.f1_mean:.6f}); nodes fold-mean {np.mean(nodes_f):.1f} (Faz B {r0.nodes_mean:.1f}); "
          f"nodes full {units['full'][1]['nodes']} (Faz B {int(r0.nodes_full)})", flush=True)
    for u_, (_, A_) in units.items():
        print(f"[saflik] {mtag} {u_}: yaprak {A_['leaves']}, saf olmayan yaprak {A_['n_impure']} ({A_['n_impure']/A_['leaves']:.5f})", flush=True)
        PUR.append(dict(model=f"n{nt}_None", unit=u_, leaves=A_["leaves"], impure=A_["n_impure"], frac=A_["n_impure"] / A_["leaves"]))
    VAL_DEV[0] = max(VAL_DEV[0], max(u[1]['value_dev'] for u in units.values()))
    print(f"[value] tree_.value satir toplami max|sum-1| = {max(u[1]['value_dev'] for u in units.values()):.3e} (>0 ise normalize degil)", flush=True)
    dpb = pipeline(f"sk_{ref_cfg}", sk_pred, {f: sk_proba[f] for f in range(5)})
    ref_rows.append(dict(model=ref_cfg, metric="auroc_all", ours=dpb["auroc"], faz_b=r0.auroc_all))
    ref_rows.append(dict(model=ref_cfg, metric="far_a10", ours=dpb["far10"], faz_b=r0.far_a10))
    ref_rows.append(dict(model=ref_cfg, metric="det_a10_all", ours=dpb["det10"], faz_b=r0.det_a10_all))
    print(f"[kapi0] anomali: AUROC {dpb['auroc']:.6f} (Faz B {r0.auroc_all:.6f}) FAR10 {dpb['far10']:.6f} ({r0.far_a10:.6f}) det10 {dpb['det10']:.6f} ({r0.det_a10_all:.6f})", flush=True)
    sk_pipe = dpb

    for enc in ENCS:
        res = {}
        for u, (rf, A) in units.items():
            sz, sums, pred = build_and_run(A, enc, f"{mtag}_{u}", keep_src=(u == "full"))
            # bagimsiz numpy yeniden uygulamasi ile C toplamlari birebir ayni olmali
            ref_sum = np_forest(A, enc)
            c_np_equal = bool((ref_sum == sums).all())
            np_pred = ref_sum.argmax(1)
            c_np_pred_equal = bool((np_pred == pred[:, 0]).all())
            embedded_equal = bool((pred[:, 0] == pred[:, 1]).all())
            arith = arith_bytes(A["nodes"], A["leaves"], A["T"], enc, A["wchild"], A["offbytes"])
            size_rows.append(dict(model=f"n{nt}_None", enc=enc, unit=u, nodes=A["nodes"], leaves=A["leaves"], wchild=A["wchild"],
                                  arith_bytes=arith, bytes8_assumed=A["nodes"] * 8, **sz,
                                  c_vs_numpy_sums_equal=c_np_equal, c_vs_numpy_pred_equal=c_np_pred_equal, embedded_vs_double_pred_equal=embedded_equal))
            res[u] = (sums, pred, A)
            print(f"  [{mtag} {enc} {u}] nodes={A['nodes']} leaves={A['leaves']} arith={arith}B rdata={sz['rdata']}B text={sz['text']}B "
                  f"| C==numpy sums:{c_np_equal} preds:{c_np_pred_equal} | embedded==double:{embedded_equal}", flush=True)
        # --- tahmin uyusmasi, F1, kapi
        fold_preds = {}
        fold_proba = {}
        for f in range(5):
            te = np.flatnonzero(S1 == f)
            sums, pred, A = res[f"f{f}"]
            fold_preds[f] = pred[te, 0].astype(int)
            denom = {"a": A["T"], "b": 255.0 * A["T"], "c": A["T"]}[enc]
            fold_proba[f] = sums[te] / denom
        # kapi: kodlama (a), double toplama
        for u in list(f"f{f}" for f in range(5)) + ["full"]:
            sums, pred, A = res[u]
            if u == "full":
                idx = np.arange(N); skp = sk_full_pred; skpr = sk_full_proba
            else:
                f = int(u[1]); idx = np.flatnonzero(S1 == f); skp = sk_pred[f]; skpr = sk_proba[f]
            cp = pred[idx, 0].astype(int)
            agree_d = float((cp == skp).mean())
            n_mis = int((cp != skp).sum())
            ce = pred[idx, 1].astype(int)
            agree_e = float((ce == skp).mean())
            gate_rows.append(dict(model=f"n{nt}_None", enc=enc, unit=u, n_windows=len(idx), n_mismatch_double=n_mis, agree_double=agree_d,
                                  n_mismatch_embedded=int((ce != skp).sum()), agree_embedded=agree_e))
            if n_mis:
                for j in np.flatnonzero(cp != skp):
                    mean_c = sums[idx][j] / {"a": A["T"], "b": 255.0 * A["T"], "c": A["T"]}[enc]
                    top2 = np.sort(skpr[j])[::-1][:2]
                    mism_rows.append(dict(model=f"n{nt}_None", enc=enc, unit=u, win=int(idx[j]), c_pred=int(cp[j]), sk_pred=int(skp[j]),
                                          max_abs_proba_diff=float(np.abs(mean_c - skpr[j]).max()), sk_top2_gap=float(top2[0] - top2[1]),
                                          sk_tie_within_1e12=bool(top2[0] - top2[1] < 1e-12)))
        # fold F1
        f1c = np.array([fc.state_metrics(Y[S1 == f], fold_preds[f])["macro_f1"] for f in range(5)])
        agree_folds = np.array([(fold_preds[f] == sk_pred[f]).mean() for f in range(5)])
        agree_all = float(np.concatenate([fold_preds[f] == sk_pred[f] for f in range(5)]).mean())
        for f in range(5):
            fold_rows.append(dict(model=f"n{nt}_None", enc=enc, fold=f, f1_c=f1c[f], f1_sklearn=f1_sk[f], f1_b2=B2_F1[f],
                                  agree=agree_folds[f], n_test=int((S1 == f).sum())))
        d_b2 = tci(f1c - B2_F1)
        d_fl = tci(f1c - f1_sk)
        pp = pipeline(f"c_{mtag}_{enc}", fold_preds, fold_proba)
        print(f"  [{mtag} {enc}] agree_vs_sklearn={agree_all:.6f} mF1={f1c.mean():.4f}±{f1c.std(ddof=1):.4f} dB2={d_b2[0]:+.4f} [{d_b2[1]:+.4f},{d_b2[2]:+.4f}] "
              f"dFloat={d_fl[0]:+.4f} [{d_fl[1]:+.4f},{d_fl[2]:+.4f}] | AUROC={pp['auroc']:.4f} FAR10={pp['far10']:.4f} det10={pp['det10']:.4f} "
              f"seq={pp['auroc_seq']:.4f} tempo={pp['auroc_tempo']:.4f}", flush=True)
        res_rows.append(dict(model=f"n{nt}_None", enc=enc, agree_all=agree_all, f1_mean=f1c.mean(), f1_sd=f1c.std(ddof=1),
                             d_f1_vs_b2=d_b2[0], d_b2_lo=d_b2[1], d_b2_hi=d_b2[2], d_f1_vs_float=d_fl[0], d_float_lo=d_fl[1], d_float_hi=d_fl[2],
                             auroc_all=pp["auroc"], auroc_sd=pp["auroc_sd"], far_a10=pp["far10"], det_a10=pp["det10"],
                             auroc_seq=pp["auroc_seq"], auroc_tempo=pp["auroc_tempo"], n_pred_cycles=pp["n_cycles"],
                             f1_sklearn_mean=f1_sk.mean(), sk_auroc=sk_pipe["auroc"], sk_far10=sk_pipe["far10"], sk_det10=sk_pipe["det10"]))
print(f"\nmodels done in {time.time()-t00:.0f}s", flush=True)

SZ = pd.DataFrame(size_rows)
GT = pd.DataFrame(gate_rows)
FD = pd.DataFrame(fold_rows)
RS = pd.DataFrame(res_rows)
MS = pd.DataFrame(mism_rows) if mism_rows else pd.DataFrame(columns=["model", "enc", "unit", "win", "c_pred", "sk_pred", "max_abs_proba_diff", "sk_top2_gap", "sk_tie_within_1e12"])
RF_ = pd.DataFrame(ref_rows)
RF_["abs_diff"] = (RF_.ours - RF_.faz_b).abs()
SZ.to_csv(REP / "01_export_c_sizes.csv", index=False)
GT.to_csv(REP / "01_export_c_gate.csv", index=False)
FD.to_csv(REP / "01_export_c_folds.csv", index=False)
MS.to_csv(REP / "01_export_c_mismatch.csv", index=False)
RF_.to_csv(REP / "01_export_c_ref_gate.csv", index=False)
PU = pd.DataFrame(PUR)
PU.to_csv(REP / "01_export_c_leaf_purity.csv", index=False)

# ----------------------------------------------------------------------------- boyut ozeti (fold-ortalama / tam veri / B2 aritmetik)
sumrows = []
for model in ["n10_None", "n25_None"]:
    for enc in ENCS:
        g = SZ[(SZ.model == model) & (SZ.enc == enc)]
        fm = g[g.unit != "full"]; fu = g[g.unit == "full"].iloc[0]
        sumrows.append(dict(model=model, enc=enc, nodes_fold_mean=fm.nodes.mean(), nodes_full=int(fu.nodes), leaves_full=int(fu.leaves),
                            arith_fold_mean=fm.arith_bytes.mean(), arith_full=int(fu.arith_bytes),
                            rdata_fold_mean=fm.rdata.mean(), rdata_full=int(fu.rdata), rdata_other_full=int(fu.rdata_other),
                            text_fold_mean=fm.text.mean(), text_full=int(fu.text),
                            bytes8_fold_mean=fm.nodes.mean() * 8, bytes8_full=int(fu.nodes) * 8,
                            ratio_rdata_over_8B_fold=fm.rdata.mean() / (fm.nodes.mean() * 8), ratio_rdata_over_8B_full=fu.rdata / (int(fu.nodes) * 8),
                            wchild=int(fu.wchild)))
# B2 (300 x None): yalniz aritmetik; yaprak sayisi = (dugum + agac sayisi)/2 (ikili tam agac)
b2r = FB.loc["tf_n300_dN"]
for enc in ENCS:
    T = 300
    nf, nu = float(b2r.nodes_mean), int(b2r.nodes_full)
    lf, lu = (nf + T) / 2, (nu + T) / 2
    sumrows.append(dict(model="n300_None(B2)", enc=enc, nodes_fold_mean=nf, nodes_full=nu, leaves_full=int(lu),
                        arith_fold_mean=arith_bytes(nf, lf, T, enc, 2, 4), arith_full=arith_bytes(nu, lu, T, enc, 2, 4),
                        rdata_fold_mean=np.nan, rdata_full=np.nan, rdata_other_full=np.nan, text_fold_mean=np.nan, text_full=np.nan,
                        bytes8_fold_mean=nf * 8, bytes8_full=nu * 8, ratio_rdata_over_8B_fold=np.nan, ratio_rdata_over_8B_full=np.nan, wchild=2))
SM_ = pd.DataFrame(sumrows)

# master csv: boyut ozeti + sonuclar
RS_ = RS.copy()
MASTER = SM_.merge(RS_, on=["model", "enc"], how="left")
MASTER.to_csv(REP / "01_export_c.csv", index=False)
SM_.to_csv(REP / "01_export_c_sizes_summary.csv", index=False)
RS.to_csv(REP / "01_export_c_results.csv", index=False)

print("\n[boyut ozeti]")
print(SM_.drop(columns=["wchild"]).to_string(), flush=True)
print("\n[sonuclar]")
print(RS.to_string(), flush=True)
print("\n[kapi]")
print(GT.to_string(), flush=True)
print("\n[uyusmayan pencere sayisi]", len(MS), flush=True)
if len(MS):
    print(MS.to_string(), flush=True)

# ----------------------------------------------------------------------------- calisma belleği
MEM = dict(window=5 * 7 * 4, feat=21 * 4, acc=6 * 4)
MEM["total"] = sum(MEM.values())
print(f"[calisma bellegi] pencere tamponu {MEM['window']} B + ozellik vektoru {MEM['feat']} B + toplayici {MEM['acc']} B = {MEM['total']} B (+ tek satir sonuc)", flush=True)

# ----------------------------------------------------------------------------- karar kurali (harfiyen)
sel = "n10_None"
dec = []
for enc in ENCS:
    s_ = SM_[(SM_.model == sel) & (SM_.enc == enc)].iloc[0]
    r_ = RS[(RS.model == sel) & (RS.enc == enc)].iloc[0]
    dec.append(dict(enc=enc, rdata_full=int(s_.rdata_full), kb=s_.rdata_full / 1024, d_b2=r_.d_f1_vs_b2, d_float=r_.d_f1_vs_float,
                    fit192=bool(s_.rdata_full <= KB192), fit256=bool(s_.rdata_full <= KB256), fit512=bool(s_.rdata_full <= KB512),
                    f1_ok_b2=bool(r_.d_f1_vs_b2 >= -0.02), f1_ok_float=bool(r_.d_f1_vs_float >= -0.02)))
DEC = pd.DataFrame(dec)
DEC.to_csv(REP / "01_export_c_decision.csv", index=False)
print("\n[karar tablosu]\n" + DEC.to_string(), flush=True)
cand = {}
for lim_name, lim, col in (("192 KB MSS veri RAM", KB192, "fit192"), ("256 KB DSP L2", KB256, "fit256"), ("512 KB MSS program RAM", KB512, "fit512")):
    ok = DEC[(DEC[col]) & (DEC.f1_ok_b2)]
    cand[lim_name] = ok
    print(f"[karar] {lim_name}: kodlamalar (sabit veri <= limit VE dF1(vs B2) >= -0.02): {list(ok.enc)}", flush=True)
verdict = None
for lim_name in cand:
    if len(cand[lim_name]):
        best = cand[lim_name].sort_values("rdata_full").iloc[0]
        verdict = (lim_name, best)
        break
if verdict:
    print(f"[karar] SONUC: {verdict[0]} icine sigar; en kucuk kodlama ({verdict[1].enc}) {verdict[1].kb:.1f} KB, dF1(vs B2)={verdict[1].d_b2:+.4f}", flush=True)
else:
    print("[karar] SONUC: sigmaz", flush=True)

# ----------------------------------------------------------------------------- önceden yazilmis okuma kontrolleri
a_rows = SM_[(SM_.enc == "a") & (SM_.model == sel)].iloc[0]
c_rows = SM_[(SM_.enc == "c") & (SM_.model == sel)].iloc[0]
b_rows = SM_[(SM_.enc == "b") & (SM_.model == sel)].iloc[0]
read_a_gt8 = bool(a_rows.rdata_fold_mean > a_rows.bytes8_fold_mean and a_rows.rdata_full > a_rows.bytes8_full)
read_a_ratio = (a_rows.ratio_rdata_over_8B_fold, a_rows.ratio_rdata_over_8B_full)
read_a_range = bool(1.5 <= read_a_ratio[0] <= 3 and 1.5 <= read_a_ratio[1] <= 3)
read_c_smallest = bool(c_rows.rdata_full < b_rows.rdata_full < a_rows.rdata_full and c_rows.rdata_fold_mean < b_rows.rdata_fold_mean < a_rows.rdata_fold_mean)
print(f"[okuma] (a) olculen > 8B/dugum: {read_a_gt8}; oran fold-ort {read_a_ratio[0]:.3f}x, tam veri {read_a_ratio[1]:.3f}x; 1.5-3x araliginda: {read_a_range}; (c) en kucuk (c<b<a): {read_c_smallest}", flush=True)

# ----------------------------------------------------------------------------- rapor (md)
def kb(x):
    return f"{x/1024:.1f}"


gate_sel = GT[(GT.model == sel) & (GT.enc == "a")]
gate_all = GT[GT.enc == "a"]
gate_pass = bool((gate_all.n_mismatch_double == 0).all())
gate_sel_pass = bool((gate_sel.n_mismatch_double == 0).all())
ref_pass = bool((RF_.abs_diff < 1e-9).all())
chk_np = bool(SZ.c_vs_numpy_sums_equal.all() and SZ.c_vs_numpy_pred_equal.all())
emb_rows = GT[GT.enc == "a"]
L = []
L.append("# Faz C / C.1 - Kucuk ormanin C'ye aktarimi ve derlenmis gercek boyutu\n")
L.append("Brief: `faz_c/DESIGN_C.md` C.1 (calistirmadan once yazildi). Betik: `faz_c/01_export_c.py`. Uretilen C: `faz_c/c/` (tam veri modelleri `forest_m10_full_{a,b,c}.c`, `forest_m25_full_{a,b,c}.c`, `driver.c`, `forest_api.h`). "
         "Tum sayilar stdout'ta ve `01_export_c*.csv` dosyalarinda. Cipte CALISTIRILMADI; x86-64 mingw gcc -Os ile derlendi.\n")
L.append("## Ozet\n")
L.append(f"- **Kapi 0 (Faz B yeniden uretimi):** {'GECTI' if ref_pass else 'GECMEDI'} (tum satirlarda |fark| < 1e-9; ayrinti asagida).")
L.append(f"- **Esdegerlik kapisi (kodlama a, double toplama, C == sklearn predict):** {'GECTI (%100)' if gate_pass else 'GECMEDI'} - secilen n=10 modeli: {'gecti' if gate_sel_pass else 'gecmedi'}; "
         f"toplam uyusmayan pencere (n10+n25, a): {int(gate_all.n_mismatch_double.sum())}.")
L.append(f"- C cikarimi ile bagimsiz numpy yeniden uygulamasi (ayni dizilerden) tum model/kodlama/birimlerde {'birebir ayni toplam ve tahmin' if chk_np else 'AYNI DEGIL'}.")
if verdict:
    v = verdict[1]
    L.append(f"- **Karar kurali (harfiyen):** secilen n=10/None modeli, kodlama ({v.enc}) tam veri modelinde **{kb(v.rdata_full)} KB** sabit veri (olculen `.rdata`), dF1(B2'ye gore) = {v.d_b2:+.4f} >= -0.02 -> **{verdict[0]}** icine sigar."
             if gate_sel_pass else f"- Karar kurali uygulanabilir degil: kapi gecmedi (boyut iddiasi kurulmaz).")
else:
    L.append("- **Karar kurali:** hicbir kodlama 512 KB'a kadar sigmadi -> **sigmaz**.")
L.append("")
L.append("## 1. Kapi 0 - Faz B yeniden uretimi\n")
L.append("| model | olcu | bu betik | Faz B / Faz A | |fark| |\n|---|---|---|---|---|")
for r in RF_.itertuples():
    L.append(f"| {r.model} | {r.metric} | {r.ours:.6f} | {r.faz_b:.6f} | {r.abs_diff:.2e} |")
L.append(f"\nFaz A B2 macro-F1 (5 kat ortalamasi) = {B2_F1.mean():.4f} (referans 0.8564).\n")
L.append(f"`tree_.value` sklearn 1.9.1'de ihrac sirasinda kontrol edildi (varsayilmadi): tum agaclarda max |satir toplami-1| = {VAL_DEV[0]:.3e} "
         f"({'dugum basina normalize edilmis oranlar' if VAL_DEV[0] < 1e-9 else 'NORMALIZE DEGIL'}). Ihracta yine sklearn predict_proba gibi agac basina normalize edildi (value / value.sum).\n")
pu10 = PU[(PU.model == "n10_None") & (PU.unit == "full")].iloc[0]
L.append(f"Yaprak saflik (ek bilgi, `_leaf_purity.csv`): n=10 tam veri modelinde {int(pu10.leaves)} yapraktan {int(pu10.impure)} tanesi saf degil ({pu10.frac:.5f}); "
         "agaclar sinirsiz derinlikte oldugundan yapraklar neredeyse saftir; (b) nicemleme ve (c) sert oy bu nedenle (a) ile ayni tahmini verir.\n")
L.append("## 2. Uygulama\n")
L.append("- Struct-of-arrays, `static const`: `FEAT` uint8 (yaprak = 255), `THR` float32, `LEFT`/`RIGHT` int16 (agac-yerel indeks; agac basina dugum <= 32767), "
         "yaprak dugumunde `LEFT` = agac-yerel yaprak indeksi; `NOFF`/`LOFF` agac basina dugum/yaprak taban ofseti (uint16, toplam dugum <= 65535). "
         "Yaprak tablosu ayri: (a) 6 x float32, (b) 6 x uint8, (c) 1 x uint8. Yaprak dugumunde `THR`/`RIGHT` kullanilmaz (yer harcanir; optimize edilmedi).")
L.append("- Esik: float64 esigi asmayan en buyuk float32 (`nextafter` ile asagi yuvarlama); ihracta tum bolunmelerde dogrulandi.")
L.append("- (b) nicemleme: floor(p*255 + 0.5); toplam uint16; (c) yaprak argmax (esitlikte kucuk sinif), oylar uint8; esitlikte kucuk sinif.")
L.append("- Kapi: sklearn'in gordugu float32 matris ham ikili dosyadan C surucusune; ozellikler C'de yeniden hesaplanmadi; toplama double.\n")
L.append("## 3. Esdegerlik kapisi (kodlama a)\n")
L.append("| model | birim | pencere | uyusmayan (double) | uyusma (double) | uyusmayan (gomulu float32 toplama) | uyusma (float32) |\n|---|---|---|---|---|---|---|")
for r in gate_all.itertuples():
    L.append(f"| {r.model} | {r.unit} | {r.n_windows} | {r.n_mismatch_double} | {r.agree_double:.6f} | {r.n_mismatch_embedded} | {r.agree_embedded:.6f} |")
if len(MS):
    L.append(f"\nUyusmayan pencereler (`01_export_c_mismatch.csv`): {len(MS)} satir (tum kodlamalar, sklearn'e gore). Ilk satirlar:\n")
    L.append(MS.head(10).to_string(index=False))
else:
    L.append("\nUyusmayan pencere yok (tum kodlama/birimler).")
gb = GT[GT.enc != "a"].groupby(["model", "enc"]).agg(n=("n_windows", "sum"), mis=("n_mismatch_double", "sum")).reset_index()
L.append("\n(b)/(c) icin sklearn predict ile toplam uyusmayan pencere (5 kat test + tam veri modeli toplami):\n")
L.append("| model | kodlama | pencere | uyusmayan |\n|---|---|---|---|")
for r in gb.itertuples():
    L.append(f"| {r.model} | {r.enc} | {r.n} | {r.mis} |")
L.append("\n## 4. Kodlama sonuclari (S1, 5 kat)\n")
L.append("dF1 = C kodlamasinin kat macro-F1'i eksi referans (B2 = Faz A 300 agac; ya da ayni modelin sklearn float cikarimi), eslesmis 5 kat, %95 t-GA (df=4).\n")
L.append("| model | kodlama | uyusma (sklearn) | macro-F1 (ort ± sd) | dF1 vs B2 [GA] | dF1 vs float [GA] | AUROC (tum anomali, D3) | FAR@0.10 | tespit@0.10 | AUROC seq | AUROC tempo |\n|---|---|---|---|---|---|---|---|---|---|---|")
for r in RS.itertuples():
    L.append(f"| {r.model} | {r.enc} | {r.agree_all:.6f} | {r.f1_mean:.4f} ± {r.f1_sd:.4f} | {r.d_f1_vs_b2:+.4f} [{r.d_b2_lo:+.4f}, {r.d_b2_hi:+.4f}] | "
             f"{r.d_f1_vs_float:+.4f} [{r.d_float_lo:+.4f}, {r.d_float_hi:+.4f}] | {r.auroc_all:.4f} ± {r.auroc_sd:.4f} | {r.far_a10:.4f} | {r.det_a10:.4f} | {r.auroc_seq:.4f} | {r.auroc_tempo:.4f} |")
L.append("\nReferans (sklearn float cikarimi, ayni model): " + "; ".join(
    f"{m}: macro-F1 {RS[RS.model==m].f1_sklearn_mean.iloc[0]:.4f}, AUROC {RS[RS.model==m].sk_auroc.iloc[0]:.4f}, FAR@0.10 {RS[RS.model==m].sk_far10.iloc[0]:.4f}, tespit@0.10 {RS[RS.model==m].sk_det10.iloc[0]:.4f}"
    for m in ["n10_None", "n25_None"]) + ". Faz A B2 x pred D3: bkz. Faz B CSV (300 x None: AUROC 0.9237, FAR 0.0784, tespit 0.6335).\n")
L.append("## 5. Gercek derlenmis boyut (gcc -Os -c, mingw64 x86-64)\n")
L.append("`.rdata` = model sabitleri (dolgu dahil), `.text` = yalniz `forest_predict` (test fonksiyonu `forest_sum` derlenmedi). Aritmetik = dizi baytlari toplami (dugum x (1+4+2*2) + yaprak x yaprak-baytlari + agac ofsetleri). "
         "'8 B/dugum' B.1 varsayimidir. x86 `.text` ARM Cortex-R4F'yi temsil ETMEZ; yalniz sabit veri mimariden bagimsizdir.\n")
L.append("| model | kodlama | dugum (fold-ort / tam) | yaprak (tam) | aritmetik KB (fold-ort / tam) | olculen .rdata KB (fold-ort / tam) | .rdata - aritmetik B (tam) | olculen / 8B-varsayim (fold-ort / tam) | .text B (fold-ort / tam) |\n|---|---|---|---|---|---|---|---|---|")
for r in SM_.itertuples():
    if r.model.startswith("n300"):
        L.append(f"| {r.model} | {r.enc} | {r.nodes_fold_mean:.1f} / {r.nodes_full} | {r.leaves_full} | {kb(r.arith_fold_mean)} / {kb(r.arith_full)} | derlenmedi | - | - | - |")
    else:
        L.append(f"| {r.model} | {r.enc} | {r.nodes_fold_mean:.1f} / {r.nodes_full} | {r.leaves_full} | {kb(r.arith_fold_mean)} / {kb(r.arith_full)} | {kb(r.rdata_fold_mean)} / {kb(r.rdata_full)} | "
                 f"{int(r.rdata_full - r.arith_full)} | {r.ratio_rdata_over_8B_fold:.2f}x / {r.ratio_rdata_over_8B_full:.2f}x | {r.text_fold_mean:.0f} / {r.text_full} |")
L.append("\nB.1 varsayimi (8 B/dugum): n=10 fold-ort " + kb(a_rows.bytes8_fold_mean) + " KB, tam veri " + kb(a_rows.bytes8_full) + " KB.\n")
L.append("## 6. Calisma bellegi (ikincil)\n")
L.append(f"Pencere tamponu 5x7x4 = {MEM['window']} B; ozellik vektoru 21x4 = {MEM['feat']} B; olasilik toplayici 6x4 = {MEM['acc']} B; toplam {MEM['total']} B (+ tek satir sonuc). "
         "Gerceklesen C kodunda toplayicilar (b) uint16, (c) uint8 oldugundan fiili bellek bundan kucuktur; ozellik hesaplama kodu ve yigit dahil degildir.\n")
L.append("## 7. Onceden yazilmis okuma ve karar kurali (harfiyen)\n")
L.append(f"- **Okuma (a):** olculen sabit veri 8 B/dugum hesabindan buyuk: **{'evet' if read_a_gt8 else 'hayir'}** (oran fold-ort {read_a_ratio[0]:.2f}x, tam veri {read_a_ratio[1]:.2f}x). "
         f"Tahmin ~1.5-3x: **{'tutuyor' if read_a_range else 'TUTMUYOR'}**.")
L.append(f"- **Okuma (c) en kucuk:** **{'tutuyor' if read_c_smallest else 'TUTMUYOR'}** (tam veri .rdata: a {kb(a_rows.rdata_full)} KB, b {kb(b_rows.rdata_full)} KB, c {kb(c_rows.rdata_full)} KB).")
L.append("\nKarar tablosu (secilen n=10/None, tam veri modeli; esikler 192 KB = 196608 B, 256 KB, 512 KB; dF1 tolerans -0.02):\n")
L.append("| kodlama | .rdata B | KB | dF1 vs B2 | dF1 vs float | <=192 KB | <=256 KB | <=512 KB | dF1(B2) >= -0.02 | dF1(float) >= -0.02 |\n|---|---|---|---|---|---|---|---|---|---|")
for r in DEC.itertuples():
    L.append(f"| {r.enc} | {r.rdata_full} | {r.kb:.1f} | {r.d_b2:+.4f} | {r.d_float:+.4f} | {r.fit192} | {r.fit256} | {r.fit512} | {r.f1_ok_b2} | {r.f1_ok_float} |")
L.append("\nKural: herhangi bir kodlama <= 192 KB ve dF1 >= -0.02 ise -> \"X KB'lik derlenmis model, MSS veri RAM'inin (192 KB) icine sigar\"; yoksa 256 KB DSP L2 / 512 KB MSS program RAM'i; yoksa \"sigmaz\". "
         "dF1 olarak B.1'deki referansa uygun sekilde B2'ye gore dF1 kullanildi; float modele gore dF1 de ayni sonucu verip vermedigi son iki sutunda.")
if not gate_sel_pass:
    L.append("\n**Kapi gecmedi: boyut iddiasi kurulmaz.**")
elif verdict:
    v = verdict[1]
    if verdict[0].startswith("192"):
        sent = f"\"{v.kb:.1f} KB'lik derlenmis model (kodlama {v.enc}, model sabitleri), MSS veri RAM'inin (192 KB) icine sigar\""
    else:
        sent = f"\"{v.kb:.1f} KB'lik derlenmis model (kodlama {v.enc}), {verdict[0]} icine sigar; 192 KB MSS veri RAM'ine sigmaz\""
    L.append(f"\n**Sonuc:** {verdict[0]} icin gecerli. Makale icin onerilen cumle: {sent} [`.rdata` olculen {int(v.rdata_full)} B; dF1(B2) {v.d_b2:+.4f}]. "
             f"Pay: 196608 - {int(v.rdata_full)} = {196608 - int(v.rdata_full)} B (%{(196608 - v.rdata_full) / 196608 * 100:.1f}); sigma payi dar ve yalniz model sabitleri icin gecerli "
             f"(calisma bellegi {MEM['total']} B ayri; yazilimin kendisi/yigit/tamponlar sayilmadi). "
             "192 KB'a sigmayan kodlamalar: " + ", ".join(f"({r.enc}) {r.kb:.1f} KB" for r in DEC.itertuples() if not r.fit192) + ".")
else:
    L.append("\n**Sonuc: sigmaz** (512 KB'a kadar hicbir kodlama kosulu saglamadi).")
L.append("\nHer durumda: **cipte calistirilmadi; bellek aygit yazilimiyla paylasilir** (veri RAM'inin yazilim, yigit, radar ham verisi tamponlari ile paylasildigi dikkate alinmali; olculen boyut yalniz model sabitleridir ve x86-64 .rdata'dir).")
L.append("\n## 8. Dosyalar\n`faz_c/01_export_c.py`, `faz_c/c/*` (forest_m{10,25}_full_{a,b,c}.c, driver.c, forest_api.h), `faz_c/oof/` (C kodlamalarinin OOF'u), "
         "`reports/faz_c/01_export_c.csv` (boyut ozeti + sonuclar), `_sizes.csv` (her birim), `_sizes_summary.csv`, `_results.csv`, `_gate.csv`, `_folds.csv`, `_mismatch.csv`, `_ref_gate.csv`, `_decision.csv`.")
(REP / "01_export_c.md").write_text("\n".join(L), encoding="utf-8")
print("\n" + "\n".join(L), flush=True)
shutil.rmtree(WORK, ignore_errors=True)
print("done", flush=True)

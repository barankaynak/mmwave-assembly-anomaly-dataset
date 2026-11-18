"""Faz B / B.3 - Kare hizi: her k'inci kare tutulursa (k in {2,3,5}) ne kalir? DESIGN_B.md B.3'e birebir.
Her seviye: oturum basina kareler k'da bir tutulur (faz kaydirmasi off, ana kosu off=0) -> L_k = ceil(5/k) pencere -> B2 ozellikleri
(7 istatistik x son/ortalama/egim; L=1'de egim 0) -> RF 300 (balanced, seed 20261104, n_jobs=3) S1 5 kat (kat = pencerenin son karesinin
dongusunun Faz A kati) -> OOF -> debounce (sure olarak sabit: MIN_BOX 0,2 s, MAX_GAP 0,1 s -> kareye yuvarlanir, en az 1 / 0) -> 10 Hz'e
sifirinci-derece tutma (ZOH) -> ac.build_cycles('pred') -> D3 AUROC/FAR (Faz A hatti). Ayrica D-ham dokunus tespiti (A3.2 yontemi).
k=1 satiri kapi: Faz A B2'yi (kat F1, kat AUROC, D-ham sayilari) birebir uretmeli; uretemezse durur.
Ek (brief disi, ikincil, kurala girmez): k=2 icin MAX_GAP=1 (yari-yukari yuvarlama), faz kaydirmasi off>=1, RF tohumu 1-4.
Her sayi stdout'a da basilir."""
import sys, time, json, importlib.util
from math import ceil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe")); sys.path.insert(0, str(ROOT / "faz_a")); sys.path.insert(0, str(ROOT / "faz_a2")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import common as pc
import fa_common as fc
import a2_cycles as ac
import b_frames as bf

SEED = 20261104
NJOBS = 3
FEATS = bf.STAT_COLS
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}
REP = ROOT / "reports" / "faz_b"
OOFD = REP / "oof"
OOFD.mkdir(parents=True, exist_ok=True)
MIN_BOX_S, MAX_GAP_S = 0.2, 0.1
THR_MM = 140.0
WIN_S = 0.1          # ToF ornegi -> 10 Hz kare atamasi (A3.2 ile ayni; kare kumesi tam hizda belirlenir, sonra tutulan karelerle kesisir)

# --- A3.2'den fit_logit / wilson / bin tanimlari (modul degistirilmeden, importlib ile)
_spec = importlib.util.spec_from_file_location("env32", ROOT / "faz_a3" / "02_envelope.py")
env = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(env)
BINLAB, EDGES = env.BINLAB, env.EDGES

ft = fc.frame_table()
W0 = fc.windows()
SESS6 = pc.sessions()
SHORT = {s: "s" + s.split("_")[1] for s in SESS6}
ROWS_OF = {s: np.flatnonzero((ft["session"] == s).to_numpy()) for s in SESS6}
assert all((ft["row"].to_numpy()[ROWS_OF[s]] == ROWS_OF[s]).all() for s in SESS6)
_g = W0.groupby("gid")["s1"].agg(["min", "max"])
assert (_g["min"] == _g["max"]).all(), "bir grup birden fazla kata dagilmis"
GID2FOLD = _g["min"].to_dict()
GID = ft["gid"].to_numpy()
CAT = {f: np.where(np.isnan(ft[f].to_numpy(float)), FILL[f], ft[f].to_numpy(float)) for f in FEATS}
POS = ft["frame_idx_in_session"].to_numpy()

# ------------------------------------------------------------------ pencereler
def deb_frames(k, mg_variant="banker"):
    """Sure olarak sabit debounce -> kare. 'banker' (ana): Python round (k=2'de MAX_GAP 0,5 -> 0); 'halfup': 0,5 -> 1;
    'mg1': MAX_GAP = 1 kare (her k); 'fixed21': kare sayisi sabit (MIN_BOX 2, MAX_GAP 1; 10 Hz'deki gibi, saniye karsiligi k ile buyur)."""
    mb = max(1, int(round(MIN_BOX_S / 0.1 / k)))
    mg = max(0, int(round(MAX_GAP_S / 0.1 / k)))
    if mg_variant == "halfup":
        mg = max(0, int(np.floor(MAX_GAP_S / 0.1 / k + 0.5)))
    elif mg_variant == "mg1":
        mg = 1
    elif mg_variant == "fixed21":
        mb, mg = 2, 1
    return mb, mg


def build_windows(k, off):
    Lk = ceil(5 / k)
    parts = []
    for s in SESS6:
        r = ROWS_OF[s]
        kept = r[off::k]
        n = len(kept)
        m = np.arange(Lk - 1, n)
        idx = m[:, None] - (Lk - 1) + np.arange(Lk)[None, :]
        parts.append(pd.DataFrame({"session": s, "m": m, "last": kept[m]}).join(
            pd.DataFrame(kept[idx], columns=[f"r{i}" for i in range(Lk)])))
    w = pd.concat(parts, ignore_index=True)
    last = w["last"].to_numpy()
    w["frame_id"] = ft["frame_id"].to_numpy()[last]; w["t"] = ft["t"].to_numpy()[last]
    w["loop_id"] = ft["loop_id"].to_numpy()[last]; w["y_true"] = ft["y_true"].to_numpy()[last]
    w["y_raw"] = ft["y_raw"].to_numpy()[last]; w["gid"] = GID[last]
    w["s1"] = [GID2FOLD[g] for g in w["gid"]]
    R = w[[f"r{i}" for i in range(Lk)]].to_numpy()
    return w, R, Lk


def features(R, Lk):
    tc = np.arange(Lk, dtype=float); tc = tc - tc.mean()
    cols = []
    for f in FEATS:
        v = CAT[f][R]
        slope = (v * tc).sum(1) / (tc ** 2).sum() if Lk > 1 else np.zeros(len(v))
        cols += [v[:, -1], v.mean(1), slope]
    return np.stack(cols, 1).astype(np.float32)


def write_oof(tag, f, wins, proba):
    df = pd.DataFrame({"session": wins["session"].to_numpy(), "frame_id": wins["frame_id"].to_numpy(), "t": wins["t"].to_numpy(),
                       "loop_id": wins["loop_id"].to_numpy(), "fold": int(f), "y_true": wins["y_true"].to_numpy(),
                       "y_raw": wins["y_raw"].to_numpy(), "y_pred": proba.argmax(1)})
    for c in range(fc.N_CLASSES):
        df[f"p{c}"] = proba[:, c]
    df[fc.OOF_COLS].to_csv(OOFD / f"fr_{tag}__S1__f{f}.csv.gz", index=False, compression="gzip")


# ------------------------------------------------------------------ anomali hatti
def anomaly(labs, name):
    C, cnt = ac.build_cycles(name, "pred", labs)
    S, _ = ac.score_group(C, verbose=False)
    PF, SM = ac.evaluate(S)
    TAB = ac.summarize(PF, SM)
    d3 = TAB[TAB.variant == "D3"].set_index("atype")
    pf = PF[PF.variant == "D3"][["atype", "f", "auroc"]]
    return d3, pf, len(C)


def upsample(pred_kept, off, k, n_full):
    """kept akis (debounce'lu) -> tam hizda ZOH: kept m'inci kare icin label, [off+k*m, off+k*(m+1)) karelerine tutulur; off oncesi none."""
    up = np.zeros(n_full, dtype=int)
    j = np.arange(n_full)
    ok = j >= off
    up[ok] = pred_kept[np.minimum((j[ok] - off) // k, len(pred_kept) - 1)]
    return up


# ------------------------------------------------------------------ dokunuslar (A3.2 mantigi, kopya; 02_envelope.py degistirilmedi)
def extract_touches():
    rows = []
    for s in SESS6:
        d = pc.load_tof_raw(s)
        tt = d["t"].to_numpy()
        dist = d[pc.DIST_COLS].to_numpy(float)
        act = np.where(np.isnan(dist), False, dist < THR_MM)
        dtm = float(np.median(np.diff(tt)))
        fs = ft[ft["session"] == s]
        tf = fs["t"].to_numpy()
        pos = np.searchsorted(tf, tt, side="left")
        posc = np.minimum(pos, len(tf) - 1)
        inwin = (pos < len(tf)) & ((tf[posc] - tt) < WIN_S - 1e-12)
        fof = np.where(inwin, posc, -1)
        for b in range(5):
            a = act[:, b].astype(np.int8)
            ch = np.flatnonzero(np.diff(np.r_[0, a, 0]))
            for i0, i1e in zip(ch[0::2], ch[1::2]):
                i1 = i1e - 1
                dur = float(tt[i1] - tt[i0] + dtm)
                fr = np.unique(fof[i0:i1 + 1]); fr = fr[fr >= 0]
                rows.append(dict(session=s, box=b + 1, dur_s=dur, bin=BINLAB[env.bin_of(dur)], fr=fr, n_frames=len(fr)))
    return pd.DataFrame(rows)


TOUCH = extract_touches()
print(f"dokunus sayisi: {len(TOUCH)} (kare kumesi bos olan, tam hizda tanimsiz: {int((TOUCH.n_frames == 0).sum())})", flush=True)


def touch_detect(pred_by_session, k, off, Lk):
    """pred_by_session[s] = kept m basina ham y_pred (penceresiz ilk Lk-1 kept kare -> 0). Donus: dokunus basina (n_kept_frames, hit)."""
    nk, hit = [], []
    for r in TOUCH.itertuples(index=False):
        fr = r.fr
        if len(fr) == 0:
            nk.append(0); hit.append(False); continue
        kp = fr[(fr >= off) & ((fr - off) % k == 0)]
        nk.append(len(kp))
        if len(kp) == 0:
            hit.append(False); continue
        hit.append(bool((pred_by_session[r.session][(kp - off) // k] == r.box).any()))
    return np.array(nk), np.array(hit)


def detection_tables(tag, nk, hit):
    """Ana tanim: payda = tam hizda degerlendirilebilir dokunuslar (n_frames>0); tutulan karesi olmayan dokunus = kacirma. Kosullu: yalniz >=1 tutulan karesi olanlar."""
    T = TOUCH.assign(nk=nk, hit=hit)
    det, fits = [], []
    for s in SESS6:
        g = T[(T.session == s) & (T.n_frames > 0)]
        for bl in BINLAB:
            gg = g[g["bin"] == bl]
            n, kk, lost = len(gg), int(gg.hit.sum()), int((gg.nk == 0).sum())
            p, lo, hi = env.wilson(kk, n)
            nc = n - lost
            pc_, loc, hic = env.wilson(kk, nc)
            det.append(dict(tag=tag, session=SHORT[s], bin=bl, n_eval=n, k_detect=kk, rate=p, wilson_lo=lo, wilson_hi=hi,
                            n_lost_sampling=lost, n_cond=nc, rate_cond=pc_, wcond_lo=loc, wcond_hi=hic))
        for mode, gg in (("all", g), ("cond", g[g.nk > 0])):
            f = env.fit_logit(gg["dur_s"].to_numpy(), gg["hit"].astype(int).to_numpy())
            f.update(tag=tag, session=SHORT[s], mode=mode)
            fits.append(f)
    return pd.DataFrame(det), pd.DataFrame(fits)


# ------------------------------------------------------------------ referans: Faz A B2
b2 = fc.load_oof("b2", "S1"); assert len(b2) == 16248
mm0 = W0[["session", "frame_id", "s1", "row_last"]].merge(b2[["session", "frame_id", "y_pred"]], on=["session", "frame_id"], how="left", validate="1:1")
Y0, S10 = W0["y_true"].to_numpy(), W0["s1"].to_numpy()
REF_F1 = np.array([fc.state_metrics(Y0[S10 == f], mm0.loc[mm0.s1 == f, "y_pred"].to_numpy())["macro_f1"] for f in range(5)])
labs_ref = ac.labels_from_oof(b2, strict=True)
d3_ref, pf_ref, ncyc_ref = anomaly(labs_ref, "b2ref")
REF_AUC = pf_ref[pf_ref.atype == "tum anomali"].sort_values("f")["auroc"].to_numpy()
rr = d3_ref.loc["tum anomali"]
print(f"[ref] Faz A B2 kat macro-F1 {np.round(REF_F1,4)} ort={REF_F1.mean():.6f}; kat AUROC {np.round(REF_AUC,4)} ort={REF_AUC.mean():.6f}; FAR={rr.far_a10:.4f} tespit={rr.det_a10:.4f} dongu={ncyc_ref}", flush=True)
assert abs(REF_AUC.mean() - 0.923697) < 1e-5
V2_B_FRAME = bf.frame_bytes(dict(tau_db=15.0, vel="full", pts="merged", group=False))["v2_B_frame"]
print(f"V2 bayt/kare (taban, tum kareler) = {V2_B_FRAME:.1f}", flush=True)
POSL0 = POS[W0["row_last"].to_numpy()]

# Faz A B2 D-ham sayilari (A3.2 csv, B2 / D-ham)
A32 = pd.read_csv(ROOT / "reports" / "faz_a3" / "02_envelope_detection.csv")
A32 = A32[(A32.model == "B2") & (A32.tanim == "D-ham")].set_index(["session", "bin"])
A32F = pd.read_csv(ROOT / "reports" / "faz_a3" / "02_envelope_fits.csv")
A32F = A32F[(A32F.model == "B2") & (A32F.tanim == "D-ham")].set_index("session")

ROWS, FOLDROWS, DETS, FITS = [], [], [], []


def run_level(tag, kind, k, off=0, seed=SEED, mg_variant="banker"):
    t0 = time.time()
    w, R, Lk = build_windows(k, off)
    X = features(R, Lk)
    Y, S1 = w["y_true"].to_numpy(), w["s1"].to_numpy()
    mb, mg = deb_frames(k, mg_variant)
    preds, pr_all = {}, np.zeros((len(w), 6), np.float32)
    f1, accs = [], []
    for f in range(5):
        te, tr = np.flatnonzero(S1 == f), np.flatnonzero(S1 != f)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=NJOBS)
        rf.fit(X[tr], Y[tr])
        pr = np.zeros((len(te), 6), np.float32)
        pr[:, rf.classes_.astype(int)] = rf.predict_proba(X[te])
        write_oof(tag, f, w.iloc[te], pr)
        pr_all[te] = pr
        m_ = fc.state_metrics(Y[te], pr.argmax(1))
        f1.append(m_["macro_f1"]); accs.append(m_["acc"])
    f1 = np.array(f1)
    ypred = pr_all.argmax(1)
    pooled = fc.state_metrics(Y, ypred)
    # kisitli referans: Faz A B2 ayni tutulan karelerde (pencere son karesi tutulan kare)
    keep0 = ((POSL0 - off) % k == 0) & (POSL0 >= off)
    ref_r = np.array([fc.state_metrics(Y0[(S10 == f) & keep0], mm0.loc[(mm0.s1 == f).to_numpy() & keep0, "y_pred"].to_numpy())["macro_f1"] for f in range(5)])
    # kept akislari
    pred_kept, labs, noop_viol = {}, {}, 0
    for s in SESS6:
        r = ROWS_OF[s]; nkept = len(r[off::k])
        pk = np.zeros(nkept, dtype=int)
        sel = (w["session"] == s).to_numpy()
        pk[w["m"].to_numpy()[sel]] = ypred[sel]
        pred_kept[s] = pk
        if s in ac.ctx().SESS:
            pdeb = ac.debounce_labels(pk, min_box=mb, max_gap=mg)
            up = upsample(pdeb, off, k, len(r))
            chk = ac.debounce_labels(up)                      # build_cycles'in ic debounce'u (2,1)
            diff = np.flatnonzero(chk != up)
            noop_viol += int((diff < len(up) - k).sum())     # oturum sonu kismi tutma haric
            labs[s] = up
    assert noop_viol == 0, f"ic debounce no-op degil: {noop_viol}"
    d3, pf, ncyc = anomaly(labs, f"fr_{tag}")
    auc = pf[pf.atype == "tum anomali"].sort_values("f")["auroc"].to_numpy()
    assert len(auc) == 5 and not np.isnan(auc).any(), (tag, auc)
    c1, c2 = bf.paired_ci(f1, REF_F1), bf.paired_ci(auc, REF_AUC)
    kp = bf.knee_pass(c1, c2)
    cr = bf.paired_ci(f1, ref_r)
    a, sq, tp = d3.loc["tum anomali"], d3.loc["seq(135+145)"], d3.loc["tempo(fast+slow)"]
    nk, hit = touch_detect(pred_kept, k, off, Lk)
    det, fits = detection_tables(tag, nk, hit)
    DETS.append(det); FITS.append(fits)
    ev = TOUCH.assign(nk=nk, hit=hit); ev = ev[ev.n_frames > 0]
    vs = ev[ev.session.map(SHORT).isin(["s02", "s03", "s04"]) & (ev.dur_s >= 0.5)]
    assert vs.session.nunique() == 3 and len(vs) == 347, (tag, vs.session.unique(), len(vs))   # s02+s03+s04 (>=0.5 s), 2026-11-05 fix
    wv = env.wilson(int(vs.hit.sum()), len(vs))
    row = dict(tag=tag, kind=kind, k=k, off=off, seed=seed, hz=10.0 / k, L=Lk, min_box_fr=mb, max_gap_fr=mg, dt_pred_s=0.1 * k,
               window_span_s=0.1 * (k * (Lk - 1) + 1), n_windows=len(w), n_pred_cycles=ncyc,
               f1_mean=f1.mean(), f1_sd=f1.std(ddof=1), acc_mean=float(np.mean(accs)),
               **{f"f1_pooled_{c}": pooled[f"f1_{c}"] for c in range(6)},
               df1_mean=c1["mean"], df1_sd=c1["sd"], df1_lo=c1["lo"], df1_hi=c1["hi"],
               ref_restricted_f1_mean=ref_r.mean(), df1_restricted_mean=cr["mean"], df1_restricted_lo=cr["lo"], df1_restricted_hi=cr["hi"],
               auroc=auc.mean(), auroc_sd=auc.std(ddof=1), dauc_mean=c2["mean"], dauc_sd=c2["sd"], dauc_lo=c2["lo"], dauc_hi=c2["hi"],
               auroc_seq=sq.auroc_mean, auroc_tempo=tp.auroc_mean, det_a10=a.det_a10, det_seq=sq.det_a10, det_tempo=tp.det_a10,
               far_a10=a.far_a10, far_se_a10=a.far_se_a10,
               pass_f1_point=kp["f1_point"], pass_f1_ci=kp["f1_ci"], pass_auc_point=kp["auc_point"], pass_auc_ci=kp["auc_ci"],
               pass_literal=kp["literal"], pass_point_only=kp["point_only"],
               touch_n_eval=len(ev), touch_lost_sampling=int((ev.nk == 0).sum()), touch_detect_all=float(ev.hit.mean()),
               touch_detect_cond=float(ev[ev.nk > 0].hit.mean()), vs_n=len(vs), vs_hit=int(vs.hit.sum()), vs_rate=wv[0], vs_lo=wv[1], vs_hi=wv[2],
               v2_B_s=V2_B_FRAME * 10.0 / k, adc_ratio=57344.0 / V2_B_FRAME,
               sec=time.time() - t0)
    ROWS.append(row)
    for f in range(5):
        FOLDROWS.append(dict(tag=tag, fold=f, macro_f1=f1[f], ref_macro_f1=REF_F1[f], ref_restricted_f1=ref_r[f], auroc_d3_all=auc[f], ref_auroc=REF_AUC[f]))
    print(f"[{tag}] k={k} off={off} seed={seed} {10.0/k:.2f} Hz L={Lk} mb/mg={mb}/{mg} n_win={len(w)} | mF1={f1.mean():.4f}±{f1.std(ddof=1):.4f} "
          f"dF1={c1['mean']:+.4f} CI[{c1['lo']:+.4f},{c1['hi']:+.4f}] (kisitli ref {ref_r.mean():.4f}, d={cr['mean']:+.4f}) | AUROC={auc.mean():.4f}±{auc.std(ddof=1):.4f} "
          f"dAUC={c2['mean']:+.4f} CI[{c2['lo']:+.4f},{c2['hi']:+.4f}] seq={sq.auroc_mean:.4f} tempo={tp.auroc_mean:.4f} det10={a.det_a10:.3f} FAR10={a.far_a10:.4f} cycles={ncyc} "
          f"| pass F1(pt,ci)=({kp['f1_point']},{kp['f1_ci']}) AUC(pt,ci)=({kp['auc_point']},{kp['auc_ci']}) literal={kp['literal']} "
          f"| touch detect all={row['touch_detect_all']:.3f} cond={row['touch_detect_cond']:.3f} lost={row['touch_lost_sampling']}/{len(ev)}; "
          f"s02-04 >=0.5s: {int(vs.hit.sum())}/{len(vs)}={wv[0]:.3f} [{wv[1]:.3f},{wv[2]:.3f}] {row['sec']:.0f}s", flush=True)
    pd.DataFrame(ROWS).to_csv(REP / "03_frame_rate.csv", index=False)
    pd.DataFrame(FOLDROWS).to_csv(REP / "03_frame_rate_folds.csv", index=False)
    pd.concat(DETS, ignore_index=True).to_csv(REP / "03_frame_rate_detection.csv", index=False)
    pd.concat(FITS, ignore_index=True).to_csv(REP / "03_frame_rate_fits.csv", index=False)
    return row, det, fits, f1, auc


# ------------------------------------------------------------------ KAPI (k=1 == Faz A B2)
row1, det1, fits1, f11, auc1 = run_level("k1", "main", 1)
fmax = np.abs(f11 - REF_F1).max(); amax = np.abs(auc1 - REF_AUC).max()
print(f"[kapi1] k=1: max|dF1 kat|={fmax:.2e}; max|dAUROC kat|={amax:.2e}; FAR {row1['far_a10']:.4f} vs {rr.far_a10:.4f}; tespit {row1['det_a10']:.4f} vs {rr.det_a10:.4f}", flush=True)
bad = 0
for r in det1.itertuples(index=False):
    a = A32.loc[(r.session, r.bin)]
    if int(a.k_detect) != r.k_detect or int(a.n_eval) != r.n_eval:
        bad += 1
        print(f"[kapi2] FARK {r.session} {r.bin}: A3.2 {int(a.k_detect)}/{int(a.n_eval)} vs {r.k_detect}/{r.n_eval}", flush=True)
f_all = fits1[fits1["mode"] == "all"].set_index("session")
fbad = 0
for s, a in A32F.iterrows():
    b = f_all.loc[s]
    for c in ("d50", "d90", "b0", "b1"):
        x, y = a[c], b[c]
        if not ((np.isnan(x) and np.isnan(y)) or abs(x - y) < 1e-9):
            fbad += 1
            print(f"[kapi2] fit FARK {s} {c}: {x} vs {y}", flush=True)
print(f"[kapi2] k=1 D-ham sayim farki (6 oturum x 5 kutu): {bad}; lojistik uydurma farki: {fbad}", flush=True)
if fmax > 1e-6 or amax > 1e-6 or bad or fbad:
    print("[kapi] DURDU: k=1 yolu Faz A B2 / A3.2 ile ozdes degil", flush=True)
    sys.exit(3)
print("[kapi] gecti: k=1 Faz A B2 ve A3.2 D-ham ile birebir", flush=True)

# ------------------------------------------------------------------ ana seviyeler
for k in (2, 3, 5):
    run_level(f"k{k}", "main", k)

# ------------------------------------------------------------------ ikincil (brief disi)
run_level("k2_mg1", "deb_alt", 2, mg_variant="halfup")
for k in (3, 5):
    run_level(f"k{k}_mg1", "deb_alt", k, mg_variant="mg1")
for k in (2, 3, 5):
    run_level(f"k{k}_deb21", "deb_alt", k, mg_variant="fixed21")
for k in (2, 3, 5):
    for off in range(1, k):
        run_level(f"k{k}_o{off}", "phase", k, off=off)
for k in (1, 2, 3, 5):
    for sd in (1, 2, 3, 4):
        run_level(f"k{k}_s{sd}", "seed", k, seed=sd)
print("done", flush=True)

"""Faz A2 / A2.5 - CPU latency (single thread). DESIGN_A2.md A2.5.
Modes:  python 05_latency.py            -> driver: trains RF (S1 fold 0), spawns 3 worker processes, writes the report
        python 05_latency.py --worker i -> one timing run (50 warm-up + 1000 timed windows per method)
        python 05_latency.py --report   -> rebuild report from the 3 run JSONs
CNN-GRU weights are RANDOM-INIT (exact architecture of faz_a/01_state_cnn.py): latency does not depend on weight values."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time, json, gc, platform, subprocess, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a")); sys.path.insert(0, str(ROOT / "probe"))
import numpy as np, pandas as pd
import fa_common as fc
import common
REP = ROOT / "reports" / "faz_a2"
CACHE = fc.CACHE_DIR
SEED = 20261104
N_WARM, N_TIME = 50, 1000
FEATS = ["cx", "cy", "n_points", "snr_sum", "mean_abs_v", "std_x", "std_y"]
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}
MODEL_PKL = CACHE / "a2_latency_models.joblib"
CNN_STREAM = "CNN-GRU streaming (1 new CNN frame + GRU over 5 cached)"
CNN_FULL = "CNN-GRU (5 frames, full window)"
HM = "Heatmap build per frame (2x48x48)"
FS = "Frame stats from points (7 feats, approx.)"
ROWS = ["B1", "B2", "B2_features_only", CNN_FULL, CNN_STREAM, HM, FS]


def stats(ms):
    a = np.asarray(ms, float)
    q1, med, q3, p95 = np.percentile(a, [25, 50, 75, 95])
    return {"median": med, "q1": q1, "q3": q3, "iqr": q3 - q1, "p95": p95, "mean": a.mean(), "n": len(a)}


def load_cnn_module():
    spec = importlib.util.spec_from_file_location("state_cnn", ROOT / "faz_a" / "01_state_cnn.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def b2_features(ft, rows):
    cols = []
    t = np.arange(fc.L, dtype=float); tc = t - t.mean()
    for f in FEATS:
        v = ft[f].to_numpy(float)
        v = np.where(np.isnan(v), FILL[f], v)[rows]
        cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
    return np.stack(cols, 1).astype(np.float32)


def centroids(Y, LX, LY, mask):
    c = np.full((5, 2), np.nan)
    for k in range(1, 6):
        m = mask & (Y == k) & ~np.isnan(LX)
        if m.sum():
            c[k - 1] = [LX[m].mean(), LY[m].mean()]
    return c


def driver():
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    ft = fc.frame_table(); W = fc.windows(); R = fc.win_rows(W)
    Y = W["y_true"].to_numpy()
    tr = W["s1"].to_numpy() != 0; te = np.flatnonzero(W["s1"].to_numpy() == 0)
    X = b2_features(ft, R)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=3)
    t0 = time.time(); rf.fit(X[tr], Y[tr])
    print(f"RF trained on S1 fold-0 train windows n={tr.sum()} in {time.time()-t0:.0f}s", flush=True)
    rf.set_params(n_jobs=1)
    cent = centroids(Y, ft["cx"].to_numpy(float)[R[:, -1]], ft["cy"].to_numpy(float)[R[:, -1]], tr)
    tau = float(pd.read_csv(fc.OUT / "02_baseline_folds.csv").query("split=='S1' and fold==0")["tau"].iloc[0])
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump({"rf": rf, "cent": cent, "tau": tau, "test_idx": te}, MODEL_PKL)
    REP.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    for i in range(3):
        print(f"--- run {i} ---", flush=True)
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(i)], env=env)
        assert r.returncode == 0
        time.sleep(3)
    write_report()


def worker(i):
    import joblib, torch
    torch.set_num_threads(1)
    d = joblib.load(MODEL_PKL); rf, cent, tau, te = d["rf"], d["cent"], d["tau"], d["test_idx"]
    rf.set_params(n_jobs=1)
    ft = fc.frame_table(); W = fc.windows(); R = fc.win_rows(W)
    hm = fc.heatmaps(mmap=False)
    rng = np.random.RandomState(0)
    sel = rng.choice(te, N_WARM + N_TIME, replace=False)
    CX = ft["cx"].to_numpy(float); CY = ft["cy"].to_numpy(float)
    out = {"run": i, "torch_threads": torch.get_num_threads(), "rf_n_jobs": rf.n_jobs}

    last_rows = R[sel, -1]
    sess_arr = ft["session"].to_numpy(); fid_arr = ft["frame_id"].to_numpy()
    pts = {}
    for s in sorted(set(sess_arr[last_rows])):
        m = common.load_merged(s)
        fids = set(fid_arr[last_rows[sess_arr[last_rows] == s]])
        mm = m[m["frame_id"].isin(fids)]
        for fid, g in mm.groupby("frame_id", sort=False):
            pts[(s, fid)] = (g["x"].to_numpy(np.float32), g["y"].to_numpy(np.float32),
                             g["snr"].to_numpy(np.float32), g["velocity"].to_numpy(np.float32))
        del m
    pts_list = [pts[(sess_arr[r], fid_arr[r])] for r in last_rows]
    K = fc.K

    def build_hm(xs, ys, snr, vel):
        ch0 = np.zeros(K * K, np.float32); ch1 = np.zeros(K * K, np.float32)
        xi = np.floor((xs - fc.X_MIN) / (fc.X_MAX - fc.X_MIN) * K).astype(np.int64)
        yi = np.floor((ys - fc.Y_MIN) / (fc.Y_MAX - fc.Y_MIN) * K).astype(np.int64)
        ok = (xi >= 0) & (xi < K) & (yi >= 0) & (yi < K)
        flat = yi[ok] * K + xi[ok]
        np.maximum.at(ch0, flat, snr[ok]); np.add.at(ch1, flat, vel[ok])
        ch1 = np.where(ch0 > 0, ch1, 0.0).astype(np.float32)
        return np.stack([ch0.reshape(K, K), ch1.reshape(K, K)])

    def frame_stats(xs, ys, snr, vel):
        w = np.clip(snr, 1e-9, None); sw = w.sum()
        return (float((xs * w).sum() / sw), float((ys * w).sum() / sw), len(xs), float(sw),
                float(np.abs(vel).mean()), float(xs.std()), float(ys.std()))

    out["heatmap_max_abs_diff_vs_cache"] = max(
        float(np.abs(build_hm(*pts_list[k]) - hm[last_rows[k]]).max()) for k in range(0, len(sel), 10))

    def timeit(fn):
        idx = list(range(len(sel)))
        for k in idx[:N_WARM]:
            fn(k)
        gc.collect()
        ms = []
        for k in idx[N_WARM:]:
            t = time.perf_counter_ns(); fn(k); ms.append((time.perf_counter_ns() - t) / 1e6)
        return stats(ms)

    lx, ly = CX[R[sel, -1]], CY[R[sel, -1]]

    def b1(k):
        dd = np.sqrt((lx[k] - cent[:, 0]) ** 2 + (ly[k] - cent[:, 1]) ** 2)
        dd = np.where(np.isnan(dd), np.inf, dd); j = dd.argmin()
        return 0 if dd[j] > tau else j + 1
    out["B1"] = timeit(b1)
    out["B2"] = timeit(lambda k: rf.predict(b2_features(ft, R[sel[k]:sel[k] + 1])))
    out["B2_features_only"] = timeit(lambda k: b2_features(ft, R[sel[k]:sel[k] + 1]))

    sc = load_cnn_module()
    torch.manual_seed(0)
    model = sc.CNNGRU().eval()
    out["cnngru_params"] = sc.n_params(model)
    xs_t = [torch.from_numpy(np.ascontiguousarray(hm[R[j]])).unsqueeze(0) for j in sel]   # (1,5,2,48,48)

    def cnn(k):
        with torch.no_grad():
            return model(xs_t[k]).argmax(1).item()
    out[CNN_FULL] = timeit(cnn)

    with torch.no_grad():
        cached = model.cnn(xs_t[0][:, 0]).unsqueeze(1).repeat(1, 4, 1)            # (1,4,feat) stand-in for cached features

    def stream(k):
        with torch.no_grad():
            f_new = model.cnn(xs_t[k][:, -1]).unsqueeze(1)
            z = torch.cat([cached, f_new], 1)
            _, h = model.gru(z)
            h = model.drop_gru(h.squeeze(0))
            return model.fc2(torch.relu(model.fc1(model.drop(h)))).argmax(1).item()
    out[CNN_STREAM] = timeit(stream)

    out[HM] = timeit(lambda k: build_hm(*pts_list[k]))
    out[FS] = timeit(lambda k: frame_stats(*pts_list[k]))
    out["mean_points_per_frame"] = float(np.mean([len(p[0]) for p in pts_list]))
    (REP / f"05_latency_run{i}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"run {i}: " + "; ".join(f"{r} med={out[r]['median']:.4f} iqr={out[r]['iqr']:.4f} p95={out[r]['p95']:.4f}" for r in ROWS), flush=True)
    print(f"run {i}: heatmap max|diff| vs cache = {out['heatmap_max_abs_diff_vs_cache']}, params={out['cnngru_params']}, pts/frame={out['mean_points_per_frame']:.1f}", flush=True)


def write_report():
    import torch, sklearn
    runs = [json.loads((REP / f"05_latency_run{i}.json").read_text(encoding="utf-8")) for i in range(3)]
    meta_p = REP / "05_latency_meta.txt"
    note = meta_p.read_text(encoding="utf-8").strip() if meta_p.exists() else "bkz. ajan notu"
    L = ["# A2.5 - CPU gecikmesi (tek is parcacigi)", "",
         "On-kayit: `faz_a2/DESIGN_A2.md` A2.5. Betik: `faz_a2/05_latency.py`. Her sayi stdout'a da basilmistir.", "",
         "## Ortam", "",
         f"- CPU: AMD Ryzen 7 3700X (8 cekirdek / 16 is parcacigi), yerel Windows 11; Python {platform.python_version()}, torch {torch.__version__}, sklearn {sklearn.__version__}, numpy {np.__version__}.",
         f"- Tek is parcacigi: OMP/MKL/OPENBLAS_NUM_THREADS=1, torch.set_num_threads(1) (olculen torch is parcacigi = {runs[0]['torch_threads']}), RF n_jobs={runs[0]['rf_n_jobs']}.",
         f"- Olcum: yontem basina {N_WARM} isinma + {N_TIME} olculen pencere (S1 kat 0 test pencerelerinden sabit tohumlu rastgele secim, 3 kosuda ayni pencereler); `time.perf_counter_ns`; tek cagri = tek pencere (batch 1). Her kosu ayri bir surec.",
         "- B1/B2 S1 kat 0 egitim verisiyle (B2: RF 300 agac, class_weight=balanced, tohum 20261104, 21 ozellik; B1: sinif centroidleri + Faz A tau). "
         f"CNN-GRU **rastgele baslatilmis agirliklarla** (faz_a/01_state_cnn.py'deki birebir mimari, {runs[0]['cnngru_params']:,} parametre, eval modu, no_grad); gecikme agirlik degerlerine bagli degildir, egitim yapilmadi.",
         f"- Makine yuku (olcum oncesi): {note}", "",
         "## Sonuc: her kosu icin medyan [IQR] (p95), ms", "",
         "| Yontem | Kosu 1 | Kosu 2 | Kosu 3 | Kosu medyanlari araligi |", "|---|---|---|---|---|"]
    for r in ROWS:
        cells = [f"{x[r]['median']:.4f} [{x[r]['iqr']:.4f}] ({x[r]['p95']:.4f})" for x in runs]
        meds = [x[r]["median"] for x in runs]
        L.append(f"| {r} | " + " | ".join(cells) + f" | {min(meds):.4f} - {max(meds):.4f} (yayilim %{100*(max(meds)-min(meds))/np.mean(meds):.1f}) |")
    L += ["", "## Ozet (3 kosunun ortalamasi)", "",
          "| Yontem | Medyan ms | IQR ms | p95 ms | 100 ms kare araligina orani (medyan) |", "|---|---|---|---|---|"]
    mean = {}
    for r in ROWS:
        m = np.mean([x[r]["median"] for x in runs]); q = np.mean([x[r]["iqr"] for x in runs]); p = np.mean([x[r]["p95"] for x in runs])
        mean[r] = m
        L.append(f"| {r} | {m:.4f} | {q:.4f} | {p:.4f} | %{m:.3f} |")
    L += ["", "## Notlar", "",
          f"- Tek-kare heatmap yapimi, onbellekteki tam boru hatti heatmap'ine karsi dogrulandi (olculen karelerin her 10'uncusu): max |fark| = {max(x['heatmap_max_abs_diff_vs_cache'] for x in runs)}. Karede ortalama {runs[0]['mean_points_per_frame']:.1f} nokta.",
          "- 'B2' = 21 ozelligin 5 karelik, onceden hesaplanmis kare istatistiklerinden cikarimi + RF.predict (300 agac). Kare istatistiklerinin ham noktalardan hesabi ayri satirdadir (yaklasik: SNR-agirlikli centroid, std, |v| ortalamasi; Faz A kare tablosunun v!=0 filtresini birebir yapmaz, yalniz maliyet mertebesi icindir).",
          f"- '{CNN_FULL}': 5 karenin hepsi CNN'den gecer (Faz A degerlendirmesi gibi; heatmap girdisi hazir, zamanlamaya dahil degil). '{CNN_STREAM}' ek, betimleyici bir senaryodur: akista yalniz yeni karenin CNN ozelligi hesaplanir, onceki 4 ozellik onbellekten gelir (burada yer tutucu ozelliklerle taklit edildi; sayi yalniz gecikme icindir).",
          f"- Uctan uca pencere karari (yeni kare icin heatmap + model): CNN-GRU tam pencere + heatmap = {mean[CNN_FULL] + mean[HM]:.3f} ms; CNN-GRU streaming + heatmap = {mean[CNN_STREAM] + mean[HM]:.3f} ms; B2 + kare istatistigi = {mean['B2'] + mean[FS]:.3f} ms (medyanlarin toplami).",
          "- **Eski makale karsilastirmasi:** eski makaledeki 12,6 ms CPU gecikmesi **Keras** ile ve **Ryzen 7 5800X** uzerinde olculmustu (farkli makine, farkli cerceve). Buradaki degerler PyTorch + Ryzen 7 3700X + tek is parcacigi ile olculmustur; dogrudan es-degerlik iddiasi yoktur.",
          f"- Faz A'daki eski B2 olcumu (`reports/faz_a/baseline_runs.json`: 13.32 ms, 300 ornegin ortalamasi) ile bu raporun B2 medyani ({mean['B2']:.2f} ms) karsilastirilabilir.",
          "", "## Desteklenen cumle (ikincil)", "",
          f"\"Masaustu CPU'da (Ryzen 7 3700X, tek is parcacigi) pencere basina B2 ~{mean['B2']:.1f} ms, CNN-GRU ~{mean[CNN_FULL]:.1f} ms; her ikisi de 100 ms kare araliginin {'altinda' if max(mean['B2'], mean[CNN_FULL]) < 100 else 'USTUNDE'} (CNN-GRU araligin %{mean[CNN_FULL]:.1f}'i).\""]
    (REP / "05_latency.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--report":
        write_report()
    else:
        driver()

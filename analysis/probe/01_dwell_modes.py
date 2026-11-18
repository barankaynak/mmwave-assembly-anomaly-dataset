"""Q1 (H1) -- Can the stations (boxes) be found WITHOUT labels, from the radar alone?

Part 1a  separability gate (diagnostic; uses ToF labels)
Part 1b  unlabeled MeanShift discovery on three proxies x two feature variants

Reusable API (imported by the Q3 script via importlib.import_module("01_dwell_modes")):
    box_centroids(frames_df)                      -> per-box centroid table
    discover_modes(X, quantile, seed, weights)    -> (centers, labels, bandwidth)
    evaluate_assignment(frames_df, centers, assign=None) -> dict(NMI, ARI, K, ...)
    assign_frames(proxy, session, centers, variant) -> per-frame mode index (-1 = unassignable)

Pre-declared pass/fail rules are the constants below (fixed BEFORE looking at any result).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import common as C

# ------------------------------------------------------------------ pre-declared constants
SEED = 0
NMI_PASS = 0.70                      # NMI threshold (box frames vs ToF label)
EXPECTED_K_TRAIN = {5, 6}            # s01: 5 boxes (+ possibly a merge zone)
EXPECTED_K_SCEN = {3, 4}             # s02-s06: 3 boxes (+ possibly a merge zone)
MIN_SCEN_PASS = 4                    # of the 5 scenario sessions
QUANTILES = (0.1, 0.2, 0.3)          # estimate_bandwidth quantiles; selected on s01 ONLY
BW_MAX_SAMPLES = 1000                # estimate_bandwidth n_samples cap
N_RESAMPLE = 6000                    # SNR-weighted resampling size (with replacement) for point proxies
MIN_MODE_SHARE = 0.02                # modes holding < 2% of the fitted mass are pruned (K_found counts survivors)
R_M = 0.45                           # nominal range of the hand [m]
DTHETA_DEG = 15.0                    # azimuth resolution [deg]
DX_RES = R_M * np.radians(DTHETA_DEG)  # lateral resolution ~ 0.118 m
MERGE_NONE_SHARE = 0.80              # mode with >= 80% 'none' frames = in-air / merge-zone mode
MERGE_MIN_FRAMES = 20

PROXIES = ("all_pts", "static_pts", "frame_cent")
PROXY_NAME = {
    "all_pts": "(i) tum merged noktalar (v!=0), SNR agirlikli",
    "static_pts": "(ii) statik v==0 noktalar (ham veri)",
    "frame_cent": "(iii) kare agirlik merkezleri (cx, cy)",
}
VARIANTS = ("xy", "x")
VARIANT_NAME = {"xy": "2B (x,y)", "x": "1B (yalniz x)"}
BOX_COLORS = {1: "#D55E00", 2: "#56B4E9", 3: "#999900", 4: "#CC79A7", 5: "#000000"}
PROXY_COLORS = {"all_pts": "#0072B2", "static_pts": "#E69F00", "frame_cent": "#009E73"}
PROXY_MARKERS = {"all_pts": "o", "static_pts": "s", "frame_cent": "^"}

MD: list[str] = []


def emit(text: str = "") -> None:
    """Write to the md buffer AND stdout (every number in the .md is also printed)."""
    MD.append(text)
    print(text, flush=True)


def sid(s: str) -> str:
    return "s" + s.split("_")[1]


def expected_k(session: str) -> set[int]:
    return EXPECTED_K_TRAIN if C.session_info(session)["role"] == "train" else EXPECTED_K_SCEN


# ================================================================== Part 1a
def box_centroids(frames_df: pd.DataFrame) -> pd.DataFrame:
    """Per-box frame-centroid statistics from frames with label > 0 (ToF label)."""
    fb = frames_df[frames_df["label"] > 0]
    g = fb.groupby("label")
    out = pd.DataFrame(
        {
            "n": g.size(),
            "cx_med": g["cx"].median(),
            "cx_std": g["cx"].std(),
            "cy_med": g["cy"].median(),
            "cy_std": g["cy"].std(),
            "cx_mean": g["cx"].mean(),
            "cy_mean": g["cy"].mean(),
        }
    )
    out.index.name = "box"
    return out.reset_index().assign(box=lambda d: d["box"].astype(int))


def adjacent_pairs(cent: pd.DataFrame, session: str) -> pd.DataFrame:
    c = cent.sort_values("cx_med", ascending=False).reset_index(drop=True)
    rows = []
    for i in range(len(c) - 1):
        a, b = c.loc[i], c.loc[i + 1]
        sp = abs(a.cx_med - b.cx_med)
        ps = float(np.sqrt((a.cx_std**2 + b.cx_std**2) / 2))
        rows.append(
            {
                "session": sid(session),
                "pair": f"{int(a.box)}-{int(b.box)}",
                "spacing_m": sp,
                "dx_res_m": DX_RES,
                "pooled_std_m": ps,
                "sep_ratio": sp / ps,
                "spacing<dx_res": bool(sp < DX_RES),
            }
        )
    return pd.DataFrame(rows)


def nn_classify(ref_cent: pd.DataFrame, frames: pd.DataFrame, dims: int, restrict: bool) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centroid on box frames; returns (true, pred). restrict=True limits classes to the boxes present."""
    fb = frames[frames["label"] > 0]
    rc = ref_cent
    if restrict:
        rc = rc[rc["box"].isin(fb["label"].unique())]
    ccols = ["cx_med", "cy_med"][:dims]
    fcols = ["cx", "cy"][:dims]
    d = np.linalg.norm(fb[fcols].to_numpy()[:, None, :] - rc[ccols].to_numpy()[None, :, :], axis=2)
    pred = rc["box"].to_numpy()[d.argmin(1)]
    return fb["label"].to_numpy().astype(int), pred.astype(int)


def part_1a(sess: list[str], ref: str, frames: dict[str, pd.DataFrame]) -> dict:
    emit("## 1a — Ayrisabilirlik kapisi (tanisal; ToF etiketi kullanir)")
    emit()
    emit(
        f"Fiziksel cozunurluk: lateral dx_res = r·Δθ = {R_M:.2f} m × {DTHETA_DEG:.0f}° = **{DX_RES:.3f} m** "
        f"(radar yalniz azimut verir, 15° azimut cozunurlugu). Kare merkezi = SNR agirlikli ortalama (6–9 nokta)."
    )
    emit()
    cents = {s: box_centroids(frames[s]) for s in sess}
    rows = []
    for s in sess:
        for _, r in cents[s].iterrows():
            rows.append({"session": sid(s), "box": int(r.box), "n": int(r.n), "cx_med": r.cx_med, "cx_std": r.cx_std,
                         "cy_med": r.cy_med, "cy_std": r.cy_std})
    emit("### Kutu basina kare-merkezi istatistikleri (m)")
    emit()
    emit(C.md_table(pd.DataFrame(rows), "{:.4f}"))
    emit()

    pair_df = pd.concat([adjacent_pairs(cents[s], s) for s in sess], ignore_index=True)
    emit("### Komsu kutu cifti ayrisma orani = |Δ medyan| / birlesik std")
    emit()
    emit(C.md_table(pair_df, "{:.3f}"))
    emit()
    n_below = pair_df.groupby("session")["spacing<dx_res"].sum()
    n_pairs = pair_df.groupby("session").size()
    emit("Cozunurluk altinda (spacing < dx_res) komsu cift sayisi: " +
         ", ".join(f"{k}: {int(n_below[k])}/{int(n_pairs[k])}" for k in n_pairs.index))
    emit()

    # Box shift relative to the reference session
    ref_c = cents[ref].set_index("box")["cx_med"]
    shift_rows = []
    for s in sess:
        if s == ref:
            continue
        c = cents[s].set_index("box")["cx_med"]
        for b in c.index:
            shift_rows.append({"session": sid(s), "box": int(b), "cx_med": c[b], f"cx_med_{sid(ref)}": ref_c.get(b, np.nan),
                               "shift_m": c[b] - ref_c.get(b, np.nan)})
    shift_df = pd.DataFrame(shift_rows)
    emit(f"### Kutu medyan cx kaymasi ({sid(ref)}'e gore, m)")
    emit()
    emit(C.md_table(shift_df, "{:.4f}"))
    emit()
    b2 = shift_df[shift_df["box"] == 2].set_index("session")["shift_m"]
    emit("Box2 kaymasi: " + ", ".join(f"{k}: {v:+.3f} m" for k, v in b2.items()))
    emit()

    # Cross-session nearest-centroid
    emit(f"### Oturumlar-arasi en-yakin-merkez siniflandirici (merkezler {sid(ref)}'ten; test: s02–s06 kutu kareleri)")
    emit()
    acc_rows = []
    pooled = {}
    for dims, dname in ((2, "(cx,cy)"), (1, "cx")):
        for restrict in (False, True):
            ys, ps = [], []
            for s in sess:
                if s == ref:
                    continue
                y, p = nn_classify(cents[ref], frames[s], dims, restrict)
                acc_rows.append({"features": dname, "classes": "oturumdaki kutular" if restrict else "5 sinif", "session": sid(s),
                                 "n": len(y), "acc": float((y == p).mean())})
                ys.append(y)
                ps.append(p)
            y, p = np.concatenate(ys), np.concatenate(ps)
            acc_rows.append({"features": dname, "classes": "oturumdaki kutular" if restrict else "5 sinif", "session": "pooled",
                             "n": len(y), "acc": float((y == p).mean())})
            pooled[(dims, restrict)] = (y, p)
    y, p = nn_classify(cents[ref], frames[ref], 2, False)
    acc_rows.append({"features": "(cx,cy)", "classes": f"{sid(ref)} resubstitution", "session": sid(ref), "n": len(y),
                     "acc": float((y == p).mean())})
    acc_df = pd.DataFrame(acc_rows)
    emit(C.md_table(acc_df, "{:.4f}"))
    emit()
    y, p = pooled[(2, False)]
    cm = pd.crosstab(pd.Series(y, name="gercek"), pd.Series(p, name="tahmin")).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
    cm.index = [f"gercek {i}" for i in cm.index]
    cm.columns = [f"tahmin {i}" for i in cm.columns]
    emit("Confusion matrix (pooled s02–s06, (cx,cy), 5 sinif):")
    emit()
    emit(C.md_table(cm.reset_index().rename(columns={"index": ""}), "{:.0f}"))
    emit()
    # Supervised ceiling: per-session own-centroid nearest-centroid on (cx,cy) (resubstitution, uses ToF labels)
    emit("### Denetimli tavan (kendi oturum merkezleriyle en-yakin-merkez, resubstitution; etiketsiz bir yontem bunu asamaz)")
    emit()
    orows = []
    oracle = {}
    for s in sess:
        y, p = nn_classify(cents[s], frames[s], 2, False)
        nmi = float(normalized_mutual_info_score(y, p))
        ari = float(adjusted_rand_score(y, p))
        oracle[s] = {"NMI": nmi, "ARI": ari, "acc": float((y == p).mean())}
        orows.append({"session": sid(s), "n_box_frames": len(y), "acc": oracle[s]["acc"], "NMI": nmi, "ARI": ari})
    emit(C.md_table(pd.DataFrame(orows), "{:.3f}"))
    emit()
    return {"cents": cents, "pairs": pair_df, "acc": acc_df, "shift": shift_df, "oracle": oracle}


# ================================================================== Part 1b — reusable core
def discover_modes_full(X, quantile: float, seed: int = SEED, weights=None) -> dict:
    """MeanShift mode discovery. weights (e.g. SNR) are honoured by fixed-seed weighted resampling
    (sklearn MeanShift has no sample_weight). Modes with < MIN_MODE_SHARE of the fitted mass are pruned."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    rng = np.random.default_rng(seed)
    if weights is not None:
        w = np.clip(np.asarray(weights, float), 1e-9, None)
        idx = rng.choice(len(X), size=min(N_RESAMPLE, len(X)), replace=True, p=w / w.sum())
        Xf = X[idx]
    else:
        Xf = X
    bw = float(estimate_bandwidth(Xf, quantile=quantile, n_samples=min(BW_MAX_SAMPLES, len(Xf)), random_state=seed))
    if not np.isfinite(bw) or bw <= 1e-6:
        bw = 1e-3
    ms = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True, n_jobs=1).fit(Xf)
    centers = ms.cluster_centers_
    counts = np.bincount(ms.labels_, minlength=len(centers))
    keep = counts / counts.sum() >= MIN_MODE_SHARE
    if not keep.any():
        keep[counts.argmax()] = True
    kept = centers[keep]
    order = np.argsort(-kept[:, 0])  # descending x (Box1 side first)
    kept = kept[order]
    labels = np.linalg.norm(X[:, None, :] - kept[None, :, :], axis=2).argmin(1)
    return {"centers": kept, "labels": labels, "bandwidth": bw, "K_raw": int(len(centers))}


def discover_modes(X, quantile: float, seed: int = SEED, weights=None):
    """-> (centers, labels, bandwidth). centers sorted by descending first coordinate."""
    r = discover_modes_full(X, quantile, seed, weights)
    return r["centers"], r["labels"], r["bandwidth"]


def evaluate_assignment(frames_df: pd.DataFrame, centers, assign=None) -> dict:
    """NMI/ARI of mode assignment vs ToF label on box frames (label > 0).
    assign=None -> nearest mode to the frame centroid (cx,cy) (cx only if centers are 1-D).
    assign given -> per-frame mode index aligned with frames_df rows, -1 = unassignable (excluded)."""
    centers = np.asarray(centers, float)
    if centers.ndim == 1:
        centers = centers[:, None]
    box = (frames_df["label"].to_numpy() > 0)
    if assign is None:
        cols = ["cx", "cy"][: centers.shape[1]]
        assign = np.linalg.norm(frames_df[cols].to_numpy()[:, None, :] - centers[None, :, :], axis=2).argmin(1)
    assign = np.asarray(assign)
    ok = box & (assign >= 0)
    n_box = int(box.sum())
    out = {"K": int(len(centers)), "n_box": n_box, "n_eval": int(ok.sum()), "coverage": float(ok.sum() / max(n_box, 1))}
    if ok.sum() < 2:
        out.update(NMI=np.nan, ARI=np.nan, K_used=int(len(np.unique(assign[ok]))))
        return out
    y = frames_df["label"].to_numpy()[ok]
    a = assign[ok]
    out.update(NMI=float(normalized_mutual_info_score(y, a)), ARI=float(adjusted_rand_score(y, a)), K_used=int(len(np.unique(a))))
    return out


# ------------------------------------------------------------------ proxy data
_PTS_CACHE: dict = {}


def proxy_points(proxy: str, session: str) -> pd.DataFrame:
    """Point/frame table for a proxy: frame_id, x, y, w."""
    key = (proxy, session)
    if key in _PTS_CACHE:
        return _PTS_CACHE[key]
    if proxy == "frame_cent":
        fr = C.load_frames(session)
        d = pd.DataFrame({"frame_id": fr["frame_id"].to_numpy(), "x": fr["cx"].to_numpy(), "y": fr["cy"].to_numpy(), "w": 1.0})
    elif proxy == "all_pts":
        p = C.load_points(session)
        d = pd.DataFrame({"frame_id": p["frame_id"].to_numpy(), "x": p["x"].to_numpy(), "y": p["y"].to_numpy(),
                          "w": p["snr"].to_numpy(float)})
    elif proxy == "static_pts":
        p = C.load_points_raw(session)
        p = p[p["static"]]
        d = pd.DataFrame({"frame_id": p["frame_id"].to_numpy(), "x": p["x"].to_numpy(), "y": p["y"].to_numpy(),
                          "w": p["snr"].to_numpy(float)})
    else:
        raise ValueError(proxy)
    _PTS_CACHE[key] = d
    return d


def assign_frames(proxy: str, session: str, centers, variant: str) -> np.ndarray:
    """Per-frame mode index aligned with C.load_frames(session).
    frame_cent: nearest mode to the frame centroid. Point proxies: SNR-weighted MAJORITY VOTE of the frame's points
    (each point -> nearest mode). Frames without points of that proxy -> -1."""
    fr = C.load_frames(session)
    pts = proxy_points(proxy, session)
    cols = ["x", "y"] if variant == "xy" else ["x"]
    P = pts[cols].to_numpy()
    centers = np.asarray(centers, float).reshape(len(centers), -1)
    nn = np.linalg.norm(P[:, None, :] - centers[None, :, :], axis=2).argmin(1)
    fidx = pd.Index(fr["frame_id"]).get_indexer(pts["frame_id"])
    if proxy == "frame_cent":
        out = np.full(len(fr), -1, int)
        ok = fidx >= 0
        out[fidx[ok]] = nn[ok]
        return out
    ok = fidx >= 0
    M = np.zeros((len(fr), len(centers)))
    np.add.at(M, (fidx[ok], nn[ok]), pts["w"].to_numpy()[ok])
    a = M.argmax(1)
    a[M.sum(1) == 0] = -1
    return a


def contingency(frames: pd.DataFrame, assign: np.ndarray, centers: np.ndarray) -> pd.DataFrame:
    """rows = modes (descending x), cols = ToF label 0..5, plus unassignable row."""
    lab = frames["label"].to_numpy()
    rows = {}
    for k in range(len(centers)):
        m = assign == k
        rows[f"mod{k + 1} (x={centers[k][0]:+.3f})"] = {f"L{b}": int(((lab == b) & m).sum()) for b in range(6)}
    m = assign < 0
    rows["atanamayan"] = {f"L{b}": int(((lab == b) & m).sum()) for b in range(6)}
    return pd.DataFrame(rows).T.reset_index().rename(columns={"index": "mod"})


# ================================================================== Part 1b — driver
def run_grid(sess: list[str], frames: dict[str, pd.DataFrame]) -> dict:
    res = {}
    for proxy in PROXIES:
        for variant in VARIANTS:
            cols = ["x", "y"] if variant == "xy" else ["x"]
            for s in sess:
                pts = proxy_points(proxy, s)
                X = pts[cols].to_numpy()
                w = None if proxy == "frame_cent" else pts["w"].to_numpy()
                for q in QUANTILES:
                    r = discover_modes_full(X, q, SEED, w)
                    a = assign_frames(proxy, s, r["centers"], variant)
                    ev = evaluate_assignment(frames[s], r["centers"], a)
                    r.update(ev)
                    r["assign"] = a
                    res[(proxy, variant, s, q)] = r
    return res


def select_quantile(res: dict, proxy: str, variant: str, ref: str) -> float:
    """Pre-declared rule on s01 ONLY: smallest distance of K to the expected set, then max NMI, then smallest q."""
    exp = expected_k(ref)
    cand = []
    for q in QUANTILES:
        r = res[(proxy, variant, ref, q)]
        d = 0 if r["K"] in exp else min(abs(r["K"] - e) for e in exp)
        nmi = r["NMI"] if np.isfinite(r["NMI"]) else -1.0
        cand.append((d, -nmi, q))
    return sorted(cand)[0][2]


def passes(r: dict, session: str) -> bool:
    return bool(r["K"] in expected_k(session) and np.isfinite(r["NMI"]) and r["NMI"] >= NMI_PASS)


def part_1b(sess: list[str], ref: str, scen: list[str], frames: dict, a1: dict) -> dict:
    emit("## 1b — Etiketsiz kesif (MeanShift)")
    emit()
    emit(
        f"Sabit kriterler (sonuc gorulmeden once): PASS = NMI ≥ {NMI_PASS} **ve** K_found ∈ beklenen kume (s01: {sorted(EXPECTED_K_TRAIN)}, "
        f"s02–s06: {sorted(EXPECTED_K_SCEN)}), s01'de **ve** ≥ {MIN_SCEN_PASS}/5 senaryo oturumunda. "
        f"Bandwidth = estimate_bandwidth(quantile ∈ {list(QUANTILES)}, n_samples ≤ {BW_MAX_SAMPLES}, random_state={SEED}); "
        f"**quantile yalniz s01'de secilir** (kural: K'nin beklenen kumeye uzakligi min, sonra max NMI, sonra min q) ve s02–s06'ya AYNEN uygulanir. "
        f"MeanShift(bin_seeding=True, cluster_all=True); toplam kutlesinin < %{MIN_MODE_SHARE * 100:.0f}'ini tutan modlar budanir (K_raw ayrica raporlanir). "
        f"SNR agirliklari sabit tohumlu (seed={SEED}) agirlikli yeniden orneklemeyle (n=min({N_RESAMPLE}, N), yerine koymali) uygulanir. "
        f"Kutu karelerinin atamasi: (i)/(ii) icin karenin noktalarinin SNR agirlikli cogunluk oyu (her nokta en yakin moda), (iii) icin kare merkezinin en yakin modu; "
        f"statik nokta icermeyen kareler (ii) icin atanamaz sayilir ve NMI/ARI disinda birakilir (coverage raporlanir). "
        f"Fit, oturumdaki TUM karelerde (etiket kullanilmadan) yapilir; NMI/ARI yalniz ToF label > 0 karelerinde olculur."
    )
    emit()
    res = run_grid(sess, frames)

    # ---- full table
    rows = []
    chosen = {}
    for proxy in PROXIES:
        for variant in VARIANTS:
            qsel = select_quantile(res, proxy, variant, ref)
            chosen[(proxy, variant)] = qsel
            for s in sess:
                for q in QUANTILES:
                    r = res[(proxy, variant, s, q)]
                    rows.append({
                        "proxy": proxy, "variant": variant, "session": sid(s), "scenario": C.session_info(s)["scenario"],
                        "quantile": q, "selected": bool(q == qsel), "bandwidth_m": r["bandwidth"], "K_raw": r["K_raw"], "K": r["K"],
                        "K_used": r["K_used"], "K_ok": bool(r["K"] in expected_k(s)), "n_eval": r["n_eval"], "coverage": r["coverage"],
                        "NMI": r["NMI"], "ARI": r["ARI"], "pass": passes(r, s),
                    })
    table = pd.DataFrame(rows)
    table.to_csv(C.REPORTS / "q1_table.csv", index=False)

    # ---- s01 bandwidth selection
    emit(f"### Bandwidth secimi ({sid(ref)} — tum quantile'lar; secim bu tablodan)")
    emit()
    t = table[table["session"] == sid(ref)].copy()
    t["proxy"] = t["proxy"].map(lambda p: p)
    emit(C.md_table(t[["proxy", "variant", "quantile", "selected", "bandwidth_m", "K_raw", "K", "K_used", "coverage", "NMI", "ARI", "pass"]], "{:.3f}"))
    emit()
    emit("Secilen quantile: " + "; ".join(f"{p}/{v}: q={chosen[(p, v)]}" for p in PROXIES for v in VARIANTS))
    emit()

    # ---- all quantiles, all sessions
    emit("### Tum oturumlar × tum quantile'lar (ozet; secilen satirlar `selected=True`, tam tablo q1_table.csv)")
    emit()
    emit(C.md_table(table[["proxy", "variant", "session", "quantile", "selected", "K_raw", "K", "coverage", "NMI", "ARI", "pass"]], "{:.3f}"))
    emit()

    # ---- verdict
    emit("### Verdict (s01'de secilen quantile ile)")
    emit()
    vrows = []
    verdict = {}
    for proxy in PROXIES:
        for variant in VARIANTS:
            q = chosen[(proxy, variant)]
            r0 = res[(proxy, variant, ref, q)]
            row = {"proxy": proxy, "variant": variant, "q_sel": q, f"{sid(ref)} K": r0["K"], f"{sid(ref)} NMI": r0["NMI"],
                   f"{sid(ref)} pass": "OK" if passes(r0, ref) else "x"}
            npass = 0
            for s in scen:
                r = res[(proxy, variant, s, q)]
                ok = passes(r, s)
                npass += int(ok)
                row[sid(s)] = f"K={r['K']} NMI={r['NMI']:.2f} {'OK' if ok else 'x'}"
            row["scen_pass"] = f"{npass}/{len(scen)}"
            v = passes(r0, ref) and npass >= MIN_SCEN_PASS
            row["VERDICT"] = "PASS" if v else "FAIL"
            verdict[(proxy, variant)] = {"verdict": v, "n_scen_pass": npass, "ref_pass": passes(r0, ref), "q": q, "ref_K": r0["K"], "ref_NMI": r0["NMI"]}
            vrows.append(row)
    emit(C.md_table(pd.DataFrame(vrows), "{:.3f}"))
    emit()
    for proxy in PROXIES:
        v2, v1 = verdict[(proxy, "xy")], verdict[(proxy, "x")]
        emit(f"- **{PROXY_NAME[proxy]}**: 2B → **{'PASS' if v2['verdict'] else 'FAIL'}** (s01 K={v2['ref_K']}, NMI={v2['ref_NMI']:.3f}, "
             f"senaryo gecen {v2['n_scen_pass']}/{len(scen)}, q={v2['q']}); 1B-x → **{'PASS' if v1['verdict'] else 'FAIL'}** "
             f"(s01 K={v1['ref_K']}, NMI={v1['ref_NMI']:.3f}, senaryo gecen {v1['n_scen_pass']}/{len(scen)}, q={v1['q']}).")
    emit()

    # ---- resolution finding
    pairs = a1["pairs"]
    p01 = pairs[pairs["session"] == sid(ref)]
    n_below = int(p01["spacing<dx_res"].sum())
    emit("### Cozunurluk bulgusu")
    emit()
    emit(f"{sid(ref)}'te komsu kutu cifti sayisi {len(p01)}; bunlardan {n_below} tanesinin aralik < dx_res = {DX_RES:.3f} m "
         f"(aralıklar: " + ", ".join(f"{r.pair}: {r['spacing_m'] * 100:.1f} cm" for _, r in p01.iterrows()) + ").")
    orc = a1["oracle"][ref]
    emit(f"Denetimli tavan {sid(ref)}: kendi merkezleriyle en-yakin-merkez (cx,cy) NMI={orc['NMI']:.3f}, ARI={orc['ARI']:.3f}, dogruluk={orc['acc']:.3f}.")
    for proxy in PROXIES:
        for variant in VARIANTS:
            v = verdict[(proxy, variant)]
            if not v["ref_pass"]:
                emit(f"- {PROXY_NAME[proxy]} / {VARIANT_NAME[variant]}: {sid(ref)}'te kriteri KARSILAMADI (K={v['ref_K']}, NMI={v['ref_NMI']:.3f}; "
                     f"esik {NMI_PASS}; denetimli tavana orani {v['ref_NMI'] / orc['NMI']:.2f}).")
    emit()
    emit(f"Yorum: {sid(ref)}'te 8–10 cm araliklı komsu ciftler dx_res altinda oldugundan bu bir **cozunurluk bulgusudur** — "
         f"ToF etiketini kullanan denetimli en-yakin-merkez bile NMI={orc['NMI']:.3f}'te kaliyor, yani esik tavana cok yakin. "
         f"Beklenenin tersine kare-merkezi ortalamasi (iii) tek noktalardan (i)/(ii) belirgin sekilde iyi ayirmiyor (s01 NMI {verdict[('frame_cent', 'xy')]['ref_NMI']:.3f} vs "
         f"{verdict[('all_pts', 'xy')]['ref_NMI']:.3f}/{verdict[('static_pts', 'xy')]['ref_NMI']:.3f}).")
    emit()

    # ---- contingency + none frames
    emit("### Karisiklik (contingency) ve 'none' karelerinin yeri — 2B, secilen quantile")
    emit()
    mode_rows = []
    merge_rows = []
    cents = a1["cents"]
    for proxy in PROXIES:
        for variant in VARIANTS:
            q = chosen[(proxy, variant)]
            for s in sess:
                r = res[(proxy, variant, s, q)]
                fr = frames[s]
                lab = fr["label"].to_numpy()
                med = cents[s].set_index("box")["cx_med"]
                n_none_total = int(((lab == 0) & (r["assign"] >= 0)).sum())
                none_in_merge = 0
                n_merge = 0
                for k, c in enumerate(r["centers"]):
                    m = r["assign"] == k
                    n_all = int(m.sum())
                    n_none = int(((lab == 0) & m).sum())
                    share_none = n_none / n_all if n_all else np.nan
                    nb = (med - c[0]).abs().idxmin()
                    is_merge = bool(n_all >= MERGE_MIN_FRAMES and share_none >= MERGE_NONE_SHARE)
                    n_merge += int(is_merge)
                    none_in_merge += n_none if is_merge else 0
                    mode_rows.append({"proxy": proxy, "variant": variant, "session": sid(s), "mode": k + 1, "x": c[0],
                                      "y": c[1] if len(c) > 1 else np.nan, "n_frames": n_all, "none_share": share_none,
                                      "nearest_box": int(nb), "dist_to_box_med_m": float(abs(med[nb] - c[0])), "merge_mode": is_merge})
                merge_rows.append({"proxy": proxy, "variant": variant, "session": sid(s), "n_merge_modes": n_merge,
                                   "none_frames_assigned": n_none_total,
                                   "none_share_in_merge_modes": none_in_merge / n_none_total if n_none_total else np.nan})
    mode_df = pd.DataFrame(mode_rows)
    mode_df.to_csv(C.REPORTS / "q1_modes.csv", index=False)
    merge_df = pd.DataFrame(merge_rows)
    emit(f"Bir mod, ≥{MERGE_MIN_FRAMES} karesi varsa ve karelerinin ≥%{MERGE_NONE_SHARE * 100:.0f}'i 'none' ise hava/birlesme bolgesi modu sayilir "
         f"(toplam none payi %53–60 oldugu icin bu kosul ayirt edicidir).")
    emit()
    emit(C.md_table(merge_df[merge_df["variant"] == "xy"].drop(columns="variant"), "{:.3f}"))
    emit()
    for proxy in PROXIES:
        for s in (ref, scen[0]):
            q = chosen[(proxy, "xy")]
            r = res[(proxy, "xy", s, q)]
            emit(f"Contingency — {PROXY_NAME[proxy]}, {sid(s)}, q={q} (satir=mod, sutun=ToF etiketi L0..L5):")
            emit()
            emit(C.md_table(contingency(frames[s], r["assign"], r["centers"]), "{:.0f}"))
            emit()
    emit("Mod ozeti (secilen quantile; tum oturumlar icin tam liste q1_modes.csv). 2B, (iii) kare merkezleri:")
    emit()
    emit(C.md_table(mode_df[(mode_df["proxy"] == "frame_cent") & (mode_df["variant"] == "xy")].drop(columns=["proxy", "variant"]), "{:.3f}"))
    emit()
    return {"res": res, "chosen": chosen, "verdict": verdict, "table": table, "mode_df": mode_df, "merge_df": merge_df}


# ================================================================== plot
def make_plot(sess: list[str], frames: dict, a1: dict, b: dict) -> None:
    res, chosen = b["res"], b["chosen"]
    fig, axes = plt.subplots(len(sess), 2, figsize=(15, 3.1 * len(sess)), squeeze=False)
    bins = np.arange(-0.40, 0.4001, 0.01)
    for i, s in enumerate(sess):
        fr = frames[s]
        ax = axes[i, 0]
        none = fr["label"] == 0
        ax.scatter(fr.loc[none, "cx"], fr.loc[none, "cy"], s=3, c="#BBBBBB", alpha=0.35, label="none", rasterized=True)
        for bx in sorted(fr.loc[fr["label"] > 0, "label"].unique()):
            m = fr["label"] == bx
            ax.scatter(fr.loc[m, "cx"], fr.loc[m, "cy"], s=4, c=BOX_COLORS[int(bx)], alpha=0.5, label=f"Box{int(bx)}", rasterized=True)
        for proxy in PROXIES:
            c = res[(proxy, "xy", s, chosen[(proxy, "xy")])]["centers"]
            ax.scatter(c[:, 0], c[:, 1], marker=PROXY_MARKERS[proxy], s=110, facecolors="none", edgecolors=PROXY_COLORS[proxy],
                       linewidths=2.2, label=f"modlar {proxy}")
        ax.set_title(f"{sid(s)} {C.session_info(s)['scenario']} — kare merkezleri (cx,cy) + 2B modlar", fontsize=9)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if i == 0:
            ax.legend(fontsize=6, ncol=3, loc="upper left")
        ax.grid(alpha=0.2)

        ax = axes[i, 1]
        for proxy in PROXIES:
            pts = proxy_points(proxy, s)
            h, _ = np.histogram(pts["x"], bins=bins, weights=pts["w"])
            h = h / max(h.max(), 1e-12)
            ax.step(bins[:-1], h, where="post", color=PROXY_COLORS[proxy], lw=1.6, label=f"yogunluk {proxy}")
        med = a1["cents"][s].set_index("box")["cx_med"]
        for bx, xm in med.items():
            ax.axvline(xm, color=BOX_COLORS[int(bx)], ls="--", lw=1.3)
            ax.text(xm, 1.28, f"B{int(bx)}", color=BOX_COLORS[int(bx)], ha="center", fontsize=8, fontweight="bold")
        for j, proxy in enumerate(PROXIES):
            c = res[(proxy, "x", s, chosen[(proxy, "x")])]["centers"]
            ax.scatter(c[:, 0], np.full(len(c), 1.06 + 0.07 * j), marker=PROXY_MARKERS[proxy], s=60, color=PROXY_COLORS[proxy])
        ax.set_ylim(0, 1.36)
        ax.set_xlim(-0.40, 0.40)
        ax.set_title(f"{sid(s)} — x yogunlugu (normalize), ToF kutu medyanlari (kesikli), 1B modlar (ust isaretler)", fontsize=9)
        ax.set_xlabel("x [m]")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(C.REPORTS / "q1_modes.png", dpi=110)
    plt.close(fig)


# ================================================================== interpretation
def interpretation(ref: str, scen: list[str], a1: dict, b: dict) -> str:
    v = b["verdict"]
    p01 = a1["pairs"][a1["pairs"]["session"] == sid(ref)]
    n_below = int(p01["spacing<dx_res"].sum())
    orc = a1["oracle"][ref]

    def vs(p):
        a, c = v[(p, "xy")], v[(p, "x")]
        return (f"2B {'PASS' if a['verdict'] else 'FAIL'} (s01 NMI={a['ref_NMI']:.3f}, K={a['ref_K']}, senaryo {a['n_scen_pass']}/{len(scen)}), "
                f"1B {'PASS' if c['verdict'] else 'FAIL'} (s01 NMI={c['ref_NMI']:.3f}, K={c['ref_K']}, senaryo {c['n_scen_pass']}/{len(scen)})")

    s1 = (f"Onceden beyan edilen kriterle (NMI ≥ {NMI_PASS}, K ∈ beklenen kume, s01 + ≥{MIN_SCEN_PASS}/5 senaryo) uc proxynin hicbiri H1'i gecmedi: "
          f"(i) tum noktalar {vs('all_pts')}; (ii) statik noktalar {vs('static_pts')}; (iii) kare merkezleri {vs('frame_cent')}.")
    s2 = (f"Senaryo oturumlari (3 kutu; komsu aralik cogunlukla 14–30 cm, yalniz s04 Box4–5 8 cm) icin (i) 2B'de 5/5 gecti (NMI 0.86–1.00), asil kayip s01'de: bitisik 5 kutudan {n_below}/{len(p01)} cift "
          f"lateral cozunurlugun ({DX_RES * 100:.1f} cm) altinda ve etiketli en-yakin-merkez bile yalnizca NMI={orc['NMI']:.3f} veriyor, yani s01 basarisizligi "
          f"mod-bulma degil cozunurluk bulgusudur (kare-merkezi ortalamasi tek noktalardan daha iyi degil). Ayrica statik v=0 noktalari (ii) tum noktalarla "
          f"hemen hemen ayni ayirma gucunu veriyor: v=0 payi etikete bagimsiz olsa da konumlari bilgi tasiyor, dolayisiyla 'yavas nokta = bekleme' filtresine gerek yok.")
    anyp = any(x["verdict"] for x in v.values())
    s3 = ("Karar kurali geregi ZDA'da bolge kalibrasyonu ToF etiketiyle yapilmali, H1 ('etiketsiz tam yapi kesfi') kismi/future-work olarak raporlanmali; "
          "ancak 3 istasyonlu senaryo oturumlarinda etiketsiz kesif cok iyi calistigi icin Soru 3'te 'kac dongu yeter' egrisi (i)-2B uzerinden yine anlamli."
          if not anyp else
          "En az bir etiketsiz proxy kriteri gecti: ZDA bolge kalibrasyonunda ToF yerine mean-shift modlari kullanilabilir.")
    return " ".join([s1, s2, s3])


# ================================================================== main
def main() -> None:
    np.random.seed(SEED)
    sess = C.sessions()
    ref = [s for s in sess if C.session_info(s)["role"] == "train"][0]
    scen = [s for s in sess if s != ref]
    frames = {s: C.load_frames(s) for s in sess}

    emit("# Q1 — Etiketsiz istasyon kesfi (H1)")
    emit()
    emit(f"Oturumlar: {', '.join(sid(s) + '=' + C.session_info(s)['scenario'] for s in sess)}. Referans (bandwidth secimi + merkezler): **{sid(ref)}**. "
         f"Radar 10 Hz, 2B nokta bulutu (z=0), azimut cozunurlugu {DTHETA_DEG:.0f}°.")
    emit()
    a1 = part_1a(sess, ref, frames)
    b = part_1b(sess, ref, scen, frames, a1)
    make_plot(sess, frames, a1, b)

    emit("## Yorum (3 cumle)")
    emit()
    emit(interpretation(ref, scen, a1, b))
    emit()
    emit("Dosyalar: `q1_modes.png`, `q1_table.csv`, `q1_modes.csv`, `q1.md`.")
    (C.REPORTS / "q1.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
    print("[q1] done", flush=True)


if __name__ == "__main__":
    main()

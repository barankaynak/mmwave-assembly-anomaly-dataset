"""Faz B / B.2 yardimci modul: ham mmwave_data.csv'den kare istatistiklerini bir oykunme altinda yeniden hesaplar.

Kare kumesi = fa_common.frame_table() kareleri (etiketler merged_df'ten geldigi icin sabit); noktalar ham dosyadan.
Istatistik tanimlari probe/common.load_frames + fa_common._frame_table ile BIREBIR AYNI pandas islemleri:
  w = snr.clip(lower=1e-9); cx = sum(x*w)/sum(w); cy; snr_sum = sum(w); n_points = size; mean_abs_v = mean(|v|);
  std_x/std_y = std(ddof=0) (NaN/0 islemi pandas groupby'unkiyle ayni).
Oykunme duzeyi = dict(tau_db, vel in {full,sign,removed}, pts in {merged,raw,static}, group in {False,True}).
Islem sirasi: nokta kumesi (c) -> SNR esigi (a) -> tepe gruplama (d) -> hiz (b).
"""
from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe")); sys.path.insert(0, str(ROOT / "faz_a"))
import numpy as np
import pandas as pd
import common
import fa_common as fc

SNR_UNIT = 10.0            # kayittaki SNR 0,1 dB biriminde: taban 150 = 15,0 dB (min(snr) = 150 tum oturumlarda)
ADC_BYTES_PER_FRAME = 57344
HZ = 10.0
STAT_COLS = ["cx", "cy", "n_points", "snr_sum", "mean_abs_v", "std_x", "std_y"]
FRAME_DT = 0.1


@lru_cache(maxsize=1)
def raw_points() -> pd.DataFrame:
    """Tum oturumlarin ham noktalari (v=0 dahil), yalniz frame_table karelerinde; dosya sirasi korunur.
    Sutunlar: session, frame_id, x, y, velocity, snr (float), range, azimuth, cell (menzil, aci hucre kimligi)."""
    ft = fc.frame_table()
    parts = []
    for s in common.sessions():
        r = pd.read_csv(common.DATA_PATH / s / "mmwave_data.csv")
        fids = ft.loc[ft["session"] == s, "frame_id"].to_numpy()
        r = r[r["frame_id"].isin(fids)].copy()
        r["session"] = s
        parts.append(r[["session", "frame_id", "x", "y", "velocity", "snr", "range", "azimuth"]])
    p = pd.concat(parts, ignore_index=True)
    p["snr"] = p["snr"].astype(float)
    # menzil / aci kutulari: degerler kesikli (+ ~1e-7 float gurultusu) -> sirali benzersiz degerlerde bosluk > esik = yeni kutu
    def cluster(v, gap):
        u = np.sort(np.unique(v.to_numpy()))
        cid = np.r_[0, np.cumsum(np.diff(u) > gap)]
        return np.searchsorted(u, v.to_numpy()), cid
    ru, rc = cluster(p["range"], 1e-4)
    au, ac_ = cluster(p["azimuth"], 1e-3)
    p["rbin"] = rc[ru]
    p["abin"] = ac_[au]
    return p


def bin_info() -> dict:
    p = raw_points()
    return {"n_range_bins": int(p["rbin"].nunique()), "n_az_bins": int(p["abin"].nunique())}


def select_points(level: dict) -> pd.DataFrame:
    p = raw_points()
    pts = level.get("pts", "merged")
    if pts == "merged":
        p = p[p["velocity"] != 0]
    elif pts == "static":
        p = p[p["velocity"] == 0]
    elif pts != "raw":
        raise ValueError(pts)
    tau = level.get("tau_db", 15.0)
    p = p[p["snr"] >= tau * SNR_UNIT]
    if level.get("group", False):
        # her (kare, menzil kutusu, aci kutusu) hucresinde yalniz en yuksek SNR'li nokta (esitlikte dosyada ilk)
        o = p.sort_values("snr", ascending=False, kind="mergesort")
        o = o.drop_duplicates(["session", "frame_id", "rbin", "abin"], keep="first")
        p = o.sort_index()
    return p


def frame_stats(level: dict) -> pd.DataFrame:
    """frame_table ile ayni satir sirasinda (row) istatistik tablosu; nokta kalmayan karede n_points=0, snr_sum=0, digerleri NaN."""
    ft = fc.frame_table()
    p = select_points(level)
    vel = level.get("vel", "full")
    v = p["velocity"].to_numpy()
    if vel == "full":
        absv = np.abs(v)
    elif vel == "sign":
        absv = (v != 0).astype(float)           # yalniz isaret biti: |sign(v)|; merged kumede sabit 1
    elif vel == "removed":
        absv = np.zeros(len(p))
    else:
        raise ValueError(vel)
    w = p["snr"].clip(lower=1e-9)
    q = p.assign(wx=p["x"] * w, wy=p["y"] * w, w=w, absv=absv, sgn=np.sign(v))
    out = []
    for s in common.sessions():
        g = q[q["session"] == s].groupby("frame_id", sort=True)
        st = pd.DataFrame({
            "n_points": g.size(),
            "cx": g["wx"].sum() / g["w"].sum(),
            "cy": g["wy"].sum() / g["w"].sum(),
            "snr_sum": g["w"].sum(),
            "mean_abs_v": g["absv"].mean(),
            "mean_sign": g["sgn"].mean(),
            "std_x": g["x"].std(ddof=0),
            "std_y": g["y"].std(ddof=0),
        })
        f = ft.loc[ft["session"] == s, ["row", "frame_id"]]
        st = f.merge(st.reset_index(), on="frame_id", how="left")
        out.append(st)
    st = pd.concat(out, ignore_index=True).sort_values("row").reset_index(drop=True)
    st["n_points"] = st["n_points"].fillna(0).astype(int)
    st["snr_sum"] = st["snr_sum"].fillna(0.0)
    if vel == "sign":
        # yalniz isaret biti akisi |v| vermez; merged kumede mean(|sign|) = 1 sabit olurdu (= 'removed' ile ayni ozellik).
        # Isaret akisinin destekledigi tek hiz istatistigi mean(sign v): mean_abs_v ozellik yuvasina konur (21 ozellik korunur).
        st["mean_abs_v"] = st["mean_sign"]
    if vel == "removed":
        st["mean_abs_v"] = 0.0                  # sabit tutulur (brief)
        st["mean_sign"] = 0.0
    assert (st["row"].to_numpy() == ft["row"].to_numpy()).all()
    return st


def frame_bytes(level: dict, st: pd.DataFrame | None = None) -> dict:
    """Bayt/kare. V2 = yalniz nokta bulutu TLV'leri (05_gaps.md 'V2'): TLV1 (8 B baslik + n x [x,y,z float32 = 12 B + hiz]) +
    TLV7 (8 B baslik + n x 4 B [SNR, noise uint16]). Hiz alani: full 4 B (float32), sign 1 B (bayt-hizali, ozel format), removed 0 B.
    Taban (n=9, full): 16 + 20*9 = 196 B (05_gaps V2). V3 = hattin okudugu yuk: x,y float32 (8) + hiz + SNR uint16 (2), baslik yok.
    Nokta kalmayan karede hicbir sey gonderilmez (0 B). bayt/s = 10 Hz x bayt/kare (05_gaps ile ayni gelenek: tum kayitli kareler)."""
    st = frame_stats(level) if st is None else st
    n = st["n_points"].to_numpy(float)
    vb = {"full": 4, "sign": 1, "removed": 0}[level.get("vel", "full")]
    v2 = np.where(n > 0, 16 + n * (12 + vb + 4), 0.0)
    v3 = n * (8 + vb + 2)
    return {"n_points_mean": float(n.mean()), "n_points_median": float(np.median(n)),
            "v2_B_frame": float(v2.mean()), "v2_B_frame_median": float(np.median(v2)), "v2_B_s": float(v2.mean() * HZ),
            "v3_B_frame": float(v3.mean()), "v3_B_s": float(v3.mean() * HZ),
            "adc_ratio_v2": ADC_BYTES_PER_FRAME / float(v2.mean()) if v2.mean() > 0 else float("nan"),
            "empty_frames": int((n == 0).sum())}


# ----------------------------------------------------------------------------- eslesik fark ve diz kurali
T975_DF4 = 2.7764451051977987


def paired_ci(x, ref):
    """5 katli eslesik fark d = x - ref: ortalama, sd(ddof=1), %95 t-GA (df=4), min/max kat farki."""
    d = np.asarray(x, float) - np.asarray(ref, float)
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    h = T975_DF4 * sd / np.sqrt(n) if n == 5 else float("nan")
    return {"mean": m, "sd": sd, "lo": m - h, "hi": m + h, "min": float(d.min()), "max": float(d.max())}


def knee_pass(f1_ci, auc_ci, tol_f1=0.02, tol_auc=0.03):
    """Brief: dusus <= tol (nokta tahmin) VE kat GA'si daha buyuk dususu disalamali (alt sinir >= -tol). Her olcut ayri."""
    r = {"f1_point": bool(f1_ci["mean"] >= -tol_f1), "f1_ci": bool(f1_ci["lo"] >= -tol_f1),
         "auc_point": bool(auc_ci["mean"] >= -tol_auc), "auc_ci": bool(auc_ci["lo"] >= -tol_auc)}
    r["literal"] = bool(r["f1_point"] and r["f1_ci"] and r["auc_point"] and r["auc_ci"])
    r["point_only"] = bool(r["f1_point"] and r["auc_point"])
    return r

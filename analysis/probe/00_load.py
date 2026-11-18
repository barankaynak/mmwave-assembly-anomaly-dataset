"""00 — schema + sanity checks. Writes reports/probe/00_schema.md and reports/probe/00_loops.csv.

Checks (known anchors): total frames 16,272; loop count vs paper's 314 complete cycles;
z/elevation == 0 (2-D cloud); velocity quantisation; recorded *_active vs ToF<140 mm rule;
frame-level label/loop consistency; what event a loop_id boundary coincides with;
dwell durations from 30 Hz ToF vs 10 Hz frames.
"""
import numpy as np
import pandas as pd

from common import (
    ACTIVE_COLS, DATA_PATH, DIST_COLS, REPORTS, TOF_ACTIVE_MM, label_from_active, load_frames, load_merged,
    load_points_raw, load_tof_raw, loops, md_table, session_info, sessions, tof_segments,
)

out = []
def emit(s=""):
    print(s)
    out.append(s)

S = sessions()
emit("# 00 — Veri şeması ve doğrulama")
emit(f"DATA_PATH = `{DATA_PATH}`  ")
emit(f"Bulunan oturumlar ({len(S)}): " + ", ".join(S))
emit()

# ---- columns / dtypes (first session)
m0 = load_merged(S[0])
emit("## merged_df.csv kolonları")
emit(", ".join(f"`{c}`:{m0[c].dtype}" for c in m0.columns if c not in ("ts_mmwave", "ts_dist")))
emit()

rows, loop_tabs, seg_rows, bound_rows = [], [], [], []
for s in S:
    info = session_info(s)
    m = load_merged(s)
    raw = pd.read_csv(DATA_PATH / s / "mmwave_data.csv", usecols=["frame_id"])
    f = load_frames(s)
    tof = load_tof_raw(s)
    dt_frames = np.median(np.diff(f["t"]))
    dt_tof = np.median(np.diff(tof["t"]))
    # merged labels vs rule on merged distances
    rule_m = label_from_active((m[DIST_COLS].to_numpy() < TOF_ACTIVE_MM).astype(int))
    rec_m = label_from_active(m[ACTIVE_COLS].to_numpy())
    vel = m["velocity"].round(4)
    lp = loops(s, source="tof")
    loop_tabs.append(lp)
    rows.append({
        "session": s, "scenario": info["scenario"],
        "points_merged": len(m), "points_raw": len(raw),
        "frames_merged": m["frame_id"].nunique(), "frames_raw": raw["frame_id"].nunique(),
        "frame_dt_med_s": dt_frames, "tof_dt_med_s": dt_tof,
        "z_eq_0": float((m["z"] == 0).mean()), "elev_eq_0": float((m["elevation"] == 0).mean()),
        "pts_per_frame_med": float(f["n_points"].median()),
        "frames_label_mixed": int((f["label_nunique"] > 1).sum()),
        "frames_loop_mixed": int((f["loop_nunique"] > 1).sum()),
        "rec_vs_rule_merged": float((rule_m == rec_m).mean()),
        "rec_vs_rule_tof": float((tof["label"] == tof["label_rule"]).mean()),
        "multi_active_tof": int((tof["label"] == -1).sum()),
        "n_loops": int(lp["loop_id"].nunique()), "loops_complete": int(lp["complete"].sum()),
        "loop_min": int(lp["loop_id"].min()), "loop_max": int(lp["loop_id"].max()),
        "vel_unique": int(vel.nunique()),
        "absv_lt_0.05": float((m["velocity"].abs() < 0.05).mean()),
    })
    # label shares per frame
    shares = f["label"].value_counts(normalize=True).sort_index()
    rows[-1].update({f"share_lab{int(k)}": float(v) for k, v in shares.items()})
    # segment durations: ToF 30 Hz vs frames 10 Hz
    for src in ("tof", "frames"):
        seg = tof_segments(s, source=src)
        for b, g in seg[seg["label"] > 0].groupby("label"):
            seg_rows.append({"session": s, "source": src, "box": int(b), "n_visits": len(g),
                             "dwell_med_s": float(g["duration_s"].median()),
                             "dwell_iqr_s": float(g["duration_s"].quantile(.75) - g["duration_s"].quantile(.25))})
    # loop boundary events: label of the frame before/after each loop_id change, and seconds since last box5 exit
    fl = f.sort_values("t")
    ch = np.flatnonzero(fl["loop_id"].to_numpy()[1:] != fl["loop_id"].to_numpy()[:-1]) + 1
    seg = tof_segments(s, source="tof")
    b5 = seg[seg["label"] == 5]
    b1 = seg[seg["label"] == 1]
    for i in ch:
        tb = fl["t"].iloc[i]
        prev5 = b5[b5["end"] <= tb + 1e-9]
        next1 = b1[b1["start"] >= tb - 1e-9]
        bound_rows.append({
            "session": s,
            "lab_before": int(fl["label"].iloc[i - 1]), "lab_after": int(fl["label"].iloc[i]),
            "since_box5_end_s": float(tb - prev5["end"].max()) if len(prev5) else np.nan,
            "until_box1_start_s": float(next1["start"].min() - tb) if len(next1) else np.nan,
        })

T = pd.DataFrame(rows)
emit("## Oturum başına özet")
core = ["session", "scenario", "frames_merged", "frames_raw", "points_merged", "points_raw", "frame_dt_med_s",
        "tof_dt_med_s", "pts_per_frame_med", "n_loops", "loops_complete", "loop_min", "loop_max"]
emit(md_table(T[core]))
emit()
emit(f"**Toplam kare (merged): {int(T['frames_merged'].sum())}** (beklenen 16.272); "
     f"toplam kare (mmwave_data): {int(T['frames_raw'].sum())}")
emit("Not: frames_raw − frames_merged farkı (varsa) merged_df'e girmeyen birkaç ham karedir; analiz kare etiketini merged_df'ten alır.  ")
emit("Not: 314 = 320 loop − 6 (her oturumun loop 0'ı, ilk Box1 girişinden önceki parça). `loop_id` her Box1 girişinde artar.  ")
emit(f"**Toplam loop: {int(T['n_loops'].sum())}; tam (beklenen sıra/kurala göre): {int(T['loops_complete'].sum())}** (makale: 314)")
emit()
emit("## Geometri / hız / etiket kontrolleri")
chk = ["session", "z_eq_0", "elev_eq_0", "vel_unique", "absv_lt_0.05", "frames_label_mixed", "frames_loop_mixed",
       "rec_vs_rule_merged", "rec_vs_rule_tof", "multi_active_tof"]
emit(md_table(T[chk]))
emit()
emit("## Kare etiket payları (0 = none)")
emit(md_table(T[["session"] + sorted(c for c in T.columns if c.startswith("share_lab"))].fillna(0.0)))
emit()

v = load_merged(S[1])["velocity"].round(3).value_counts().sort_index()
emit(f"## Hız kuantizasyonu ({S[1]}): en sık 12 değer")
emit(", ".join(f"{k:+.3f}:{n}" for k, n in v.sort_values(ascending=False).head(12).items()))
emit()

emit("## Ham nokta bulutu vs merged_df: düşürülen noktalar")
emit("merged_df.csv, mmwave_data.csv'den **tam olarak v == 0 noktalarını** düşürmüş mü? Ve v == 0 noktaları hangi kare etiketinde yoğun?")
rr = []
for s in S:
    r = load_points_raw(s)
    m = load_merged(s)
    lab_share = r[r["static"]].groupby("label").size() / r.groupby("label").size()
    rr.append({"session": s, "raw_pts": len(r), "static_pts": int(r["static"].sum()),
               "raw_minus_merged": len(r) - len(m), "static_share": float(r["static"].mean()),
               **{f"static_share_lab{int(k)}": float(v) for k, v in lab_share.items()}})
emit(md_table(pd.DataFrame(rr).fillna(0.0)))
emit()

L = pd.concat(loop_tabs, ignore_index=True)
L.to_csv(REPORTS / "00_loops.csv", index=False)
emit("## Loop yapısı (ToF segmentlerinden, debounce'lu)")
for s, g in L.groupby("session"):
    top = g["box_order"].value_counts().head(5)
    inc = g[~g["complete"]]
    emit(f"- **{s}**: süre medyan {g['duration_s'].median():.2f} s (IQR {g['duration_s'].quantile(.25):.2f}–"
         f"{g['duration_s'].quantile(.75):.2f}); en sık sıralar: " + "; ".join(f"`{k}`×{n}" for k, n in top.items())
         + f"; tam olmayan loop'lar: {', '.join(f'{r.loop_id}(`{r.box_order}`, {r.duration_s:.1f}s)' for r in inc.itertuples()) or '—'}")
emit()

B = pd.DataFrame(bound_rows)
emit("## loop_id sınırı hangi olaya denk geliyor?")
for s, g in B.groupby("session"):
    emit(f"- {s}: n={len(g)}; önceki/sonraki kare etiketi en sık: "
         + ", ".join(f"{a}→{b}×{n}" for (a, b), n in g.groupby(['lab_before', 'lab_after']).size().sort_values(ascending=False).head(4).items())
         + f"; son Box5 çıkışından beri medyan {g['since_box5_end_s'].median():.2f} s; sonraki Box1 girişine medyan {g['until_box1_start_s'].median():.2f} s")
emit()

SG = pd.DataFrame(seg_rows)
emit("## Kalış süreleri: 30 Hz ToF vs 10 Hz kare etiketi (medyan s)")
pv = SG.pivot_table(index=["session", "box"], columns="source", values=["dwell_med_s", "n_visits"]).reset_index()
pv.columns = ["_".join(c).strip("_") for c in pv.columns]
emit(md_table(pv))
emit()

(REPORTS / "00_schema.md").write_text("\n".join(out), encoding="utf-8")
print(f"\n[wrote] {REPORTS / '00_schema.md'}")

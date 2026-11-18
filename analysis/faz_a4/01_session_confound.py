"""Faz A4.1 (M1) - Anomali turu oturumla karisik mi?  DESIGN_A4.md A4.1 birebir.
(a) B2 x pred baslangictan once verilen alarmlar (A3.3'te 17/71 seq135, 17/114 seq145): tetikleyen olay + alarm anindaki B2 tahmini.
(b) Karsi kanal negatif kontrol (Faz A standart p, B2 x pred, alpha 0.10/0.05): Wilson + dongu-kumeli bootstrap.
(c) Oturum basina adim sureleri (30 Hz ToF segmentleri), medyan + IQR.
YASAK: s02 dogal sapma dongulerinin alarm durumuna bakmak -> s02 normal bir butun olarak (Faz A FAR) kullanilir, natural_dev ayrilmaz.
"""
from __future__ import annotations
import importlib.util, math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
import a2_cycles as ac  # noqa: E402
fc = ac.fc
sys.path.insert(0, str(ROOT / "probe"))
import common as pc  # noqa: E402  (probe/common.py)

OUT = ROOT / "reports" / "faz_a4"
OUT.mkdir(parents=True, exist_ok=True)
MD: list[str] = []
LINK: list = []
LINKC: list = []
ALPHA = 0.10
B_BOOT = 10000
SEED = 20261105


def emit(s=""):
    print(s)
    MD.append(s)


def md_table(df, ff="{:.3f}"):
    return pc.md_table(df, ff)


spec = importlib.util.spec_from_file_location("online03", ROOT / "faz_a3" / "03_online.py")
online = importlib.util.module_from_spec(spec)
spec.loader.exec_module(online)


# ------------------------------------------------------------------ yardimcilar
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ================================================================== (a)
def part_a():
    emit("## A4.1(a) Baslangictan ONCE verilen alarmlar (B2 x pred, A3.3 literal tanim)")
    emit("")
    ctx_ = ac.ctx()
    LAB = ac.source_labels(sources=("tof", "b2"))
    active = {}
    for s in ctx_.SESS:
        m = fc.common.load_merged(s)
        pf = m.groupby("frame_id", sort=True)[["box3_active", "box4_active"]].max()
        fid = ctx_.SE[s]["frame_id"].to_numpy()
        active[s] = {3: pf["box3_active"].reindex(fid).fillna(0).to_numpy().astype(bool),
                     4: pf["box4_active"].reindex(fid).fillna(0).to_numpy().astype(bool)}
    C, _ = ac.build_cycles("b2", "pred", LAB["b2"])
    cl = C.to_dict("records")
    LINKC.append(C)
    rows = []
    n_amb = 0
    for f in range(ac.N_FOLDS):
        ro = fc.anom_roles(f)
        tr, ca, te = ac._split_roles(cl, ro)
        par = ac.fit_params(tr)
        calS = ac.score_cycles(ca, par)
        cal1, cal2 = calS["D1"].to_numpy(float), calS["D2"].to_numpy(float)
        for c in te:
            if c["atype"] not in ("seq135", "seq145"):
                continue
            s = c["session"]
            lab_deb_all = ac.debounce_labels(LAB["b2"][s])
            vs, sel, tt = online.visit_list(c, ctx_, lab_deb_all)
            ev, fb, end_fr = online.build_events(c, par, vs, sel, tt, LAB["b2"][s], causal=False)
            al, fr, kind, _d1, _d2 = online.run_online(ev, cal1, cal2)
            m = len(sel)
            n = len(tt)
            cap = lambda i: int(min(i, n - 1))
            # --- olay -> ziyaret etiketleme
            tags = []
            for (efr, order, ekind, a, b) in ev:
                if ekind == "trans_start":
                    tg = (0, ("S", sel[0][0]))
                elif ekind == "trans_end":
                    tg = (m - 1, (sel[-1][0], "E")) if m else (None, ("S", "E"))
                elif ekind == "dwell":
                    cand = [j for j in range(m) if (efr == (end_fr if j == m - 1 else cap(sel[j][2])))]
                    if len(cand) != 1:
                        n_amb += 1
                    j = cand[-1] if cand else None
                    tg = (j, (sel[j][0],) if j is not None else (None,))
                else:  # trans_mid / gap
                    cand = [j for j in range(1, m) if efr == cap(sel[j][1] + ac.MIN_BOX - 1)]
                    if len(cand) != 1:
                        n_amb += 1
                    j = cand[0] if cand else None
                    tg = (j, (sel[j - 1][0], sel[j][0]) if j is not None else (None, None))
                tags.append(tg)
            # --- tetikleyen olay (ilk alarm eden olay) - run_online'in kopyasi + indeks
            d1 = d2 = 0.0
            first = None
            for k, (efr, _o, ekind, a, b) in enumerate(ev):
                if a is not None:
                    d1 = max(d1, a)
                if b is not None:
                    d2 = max(d2, b)
                if first is None:
                    p1 = float(ac.conf_p(cal1, [d1])[0])
                    p2 = float(ac.conf_p(cal2, [d2])[0])
                    if min(1.0, 2.0 * min(p1, p2)) <= ALPHA + 1e-9:
                        first = (efr, ekind, k, p1, p2)
            assert (first is not None) == al and (first is None or (first[0] == fr and first[1] == kind)), "run_online uyumsuz"
            oi = online.onset_frame(c, active)
            rec = dict(session=s, cycle_idx=c["cycle_idx"], f=f, atype=c["atype"], n_visits=m,
                       seq=",".join(map(str, c["seq"])), alarm=al, onset_fr=-1 if oi is None else oi)
            if al:
                k = first[2]
                tg = tags[k]
                rec.update(alarm_fr=fr, kind=kind, p1=first[3], p2=first[4], trig_visit=tg[0],
                           trig_boxes="->".join(map(str, tg[1])) if tg[1][0] is not None else "?",
                           trig_a=(tg[1][0] if len(tg[1]) == 2 else None),
                           trig_b=(tg[1][-1]),
                           pred_raw=int(LAB["b2"][s][fr]), pred_deb=int(lab_deb_all[fr]), tof_state=int(LAB["tof"][s][fr]),
                           pred_raw_prev=int(LAB["b2"][s][max(fr - 1, 0)]))
                rec["sec_from_onset"] = float(tt[fr] - tt[oi]) if oi is not None else np.nan
                rec["frames_before_onset"] = (oi - fr) if oi is not None else np.nan
            rows.append(rec)
    R = pd.DataFrame(rows)
    R["wbox"] = R.atype.map({"seq135": 3, "seq145": 4})
    # ---- yeniden uretim kontrolu
    emit("### 0. Yeniden uretim kontrolu (A3.3: baslangic anina esit/oncesi alarm 17/71 ve 17/114)")
    emit("")
    chk = []
    for at, exp in (("seq135", (17, 71)), ("seq145", (17, 114))):
        d = R[(R.atype == at) & R.alarm & (R.onset_fr >= 0)]
        e = d[d.sec_from_onset <= 1e-9]
        got = (len(e), len(d))
        chk.append(dict(atype=at, pre_onset=len(e), alarm_with_onset=len(d), expected=f"{exp[0]}/{exp[1]}", ok=got == exp,
                        n_pairs=int((R.atype == at).sum()), n_alarm_all=int(R[(R.atype == at)].alarm.sum()),
                        trig_kinds=str(e.kind.value_counts().to_dict())))
    CK = pd.DataFrame(chk)
    emit(md_table(CK))
    emit("")
    emit(f"- belirsiz olay->ziyaret eslemesi sayisi: {n_amb}")
    ok_all = bool(CK.ok.all())
    emit(f"- **Yeniden uretim: {'GECTI' if ok_all else 'BASARISIZ'}**")
    if not ok_all:
        (OUT / "01_session_confound.md").write_text("\n".join(MD), encoding="utf-8")
        print("REPRO=FAIL")
        sys.exit(1)
    print("REPRO=PASS")
    emit("")
    A = R[R.alarm & (R.onset_fr >= 0)].copy()
    A["pre"] = A.sec_from_onset <= 1e-9
    A["strict_pre"] = A.frames_before_onset > 0
    A["pred_wrong_raw"] = A.pred_raw == A.wbox
    A["pred_wrong_any34"] = A.pred_raw.isin([3, 4])
    for _c in ("alarm_fr", "pred_raw", "pred_deb", "tof_state", "trig_visit"):
        A[_c] = A[_c].astype(int)
    A["pred_wrong_deb"] = A.pred_deb == A.wbox

    def cat_with(flag):
        def _c(r):
            if r[flag]:
                return "(i) radar yanlis kutuyu gordu (tahmin = anomali kutusu)"
            if (r.kind == "dwell" and r.trig_b == 1) or (r.kind in ("trans_mid", "gap") and r.trig_a == 1):
                return "(ii) oturuma ozgu zamanlama (Box1 dwell / Box1->X)"
            return "(iii) diger"
        return _c
    A["cat_raw"] = A.apply(cat_with("pred_wrong_raw"), axis=1)
    A["cat"] = A.apply(cat_with("pred_wrong_deb"), axis=1)      # BIRINCIL: alarm mantiginin gordugu (debounce'lu) durum
    A.to_csv(OUT / "01_session_confound_a_alarms.csv", index=False)
    P = A[A.pre]

    emit("### 1. Ozet: alarm veren (dongu,f) ciftleri; baslangic = Box3/Box4'e ilk ToF girisi")
    emit("")
    s1 = []
    for at in ("seq135", "seq145"):
        d = A[A.atype == at]
        p = d[d.pre]
        s1.append(dict(tur=at, n_cift=int((R.atype == at).sum()), n_alarm=len(d), n_baslangic_oncesi_veya_esit=len(p),
                       oran_alarm_icinde=len(p) / len(d), n_kesin_once=int(p.strict_pre.sum()), n_esit=int((~p.strict_pre).sum()),
                       n_tekil_dongu=p[["session", "cycle_idx"]].drop_duplicates().shape[0],
                       n_tekil_dongu_alarmli=d[["session", "cycle_idx"]].drop_duplicates().shape[0]))
    emit(md_table(pd.DataFrame(s1)))
    emit("")
    emit("### 2. Baslangic oncesi/esit alarmlar: tetikleyen olay x alarm anindaki B2 tahmini (ham y_pred)")
    emit("")
    for at in ("seq135", "seq145"):
        p = P[P.atype == at]
        emit(f"**{at}** (n={len(p)})")
        emit("")
        ct = pd.crosstab([p.kind, p.trig_boxes], p.pred_raw)
        ct.columns = [f"pred={c}" for c in ct.columns]
        ct = ct.reset_index()
        emit(md_table(ct))
        emit("")
    emit("### 3. Iki aciklamanin ayrimi (kural: (i) alarm anindaki B2 durumu = anomali kutusu; (ii) degilse ve tetik Box1 dwell ya da Box1->X; (iii) diger)")
    emit("")
    emit("BIRINCIL tanim = alarm mantiginin gordugu debounce'lu B2 durumu (`cat`); duyarlilik = ham y_pred (`cat_raw`). Fark: seq145'te 2 alarm (1->4 gecisi, ham tahmin alarm karesinde 0, debounce'lu 4).")
    emit("")
    s3 = []
    for cc, nmc in (("cat", "debounce'lu (BIRINCIL)"), ("cat_raw", "ham y_pred (duyarlilik)")):
        for at in ("seq135", "seq145", "ikisi"):
            p = P[P.atype == at] if at != "ikisi" else P
            for cname in sorted(p[cc].unique()):
                q = p[p[cc] == cname]
                s3.append(dict(durum_tanimi=nmc, tur=at, kategori=cname, n=len(q), oran=len(q) / len(p),
                               n_kesin_once=int(q.strict_pre.sum()), n_esit=int((~q.strict_pre).sum()),
                               n_tekil_dongu=q[["session", "cycle_idx"]].drop_duplicates().shape[0]))
            s3.append(dict(durum_tanimi=nmc, tur=at, kategori="TOPLAM", n=len(p), oran=1.0, n_kesin_once=int(p.strict_pre.sum()),
                           n_esit=int((~p.strict_pre).sum()), n_tekil_dongu=p[["session", "cycle_idx"]].drop_duplicates().shape[0]))
    S3 = pd.DataFrame(s3)
    S3.to_csv(OUT / "01_session_confound_a_summary.csv", index=False)
    emit(md_table(S3))
    emit("")
    emit(f"Duyarlilik sayilari (ayni {len(P)} alarm): "
         f"ham tahmin = anomali kutusu: {int(P.pred_wrong_raw.sum())}; ham tahmin in {{3,4}}: {int(P.pred_wrong_any34.sum())}; "
         f"debounce'lu tahmin = anomali kutusu: {int(P.pred_wrong_deb.sum())}; "
         f"ToF (oracle) durumu alarm karesinde = anomali kutusu: {int((P.tof_state == P.wbox).sum())}.")
    emit("")
    emit("(ii) alarmlarini veren tekil donguler (Box1 dwell'i ToF 30 Hz'den; s02 Box1-dwell dagilimindaki yuzdelik konumuyla):")
    emit("")
    PII = P[P["cat"].str.startswith("(ii)")]
    PII.to_csv(OUT / "01_session_confound_a_cat_ii_alarms.csv", index=False)
    LINK.append(PII)
    emit("Alarm karesi ile baslangic arasindaki fark (kare; 10 Hz, pozitif = alarm baslangictan once):")
    emit("")
    emit(md_table(P.groupby("atype").frames_before_onset.describe()[["count", "min", "25%", "50%", "75%", "max"]].reset_index()))
    emit("")
    emit("### 4. Baglam: tum alarm veren ciftler (baslangic oncesi + sonrasi) - kategori dagilimi")
    emit("")
    s4 = []
    for at in ("seq135", "seq145"):
        d = A[A.atype == at]
        for flag, nm in ((True, "baslangic oncesi/esit"), (False, "baslangic sonrasi")):
            q = d[d.pre == flag]
            for cname, g in q.groupby("cat"):
                s4.append(dict(tur=at, zaman=nm, kategori=cname, n=len(g)))
    emit(md_table(pd.DataFrame(s4)))
    emit("")
    emit("Referans (A3.3, ToF x tof oracle): baslangic anina esit/oncesi alarm 4/285 (seq135) ve 0/245 (seq145); oracle etiket ToF esiginden once degisemez.")
    emit("")
    return R, A


# ================================================================== (b)
def part_b():
    emit("## A4.1(b) Karsi kanal = negatif kontrol (B2 x pred, Faz A standart p; role=test)")
    emit("")
    d = pd.read_csv(ROOT / "reports" / "faz_a" / "04_cycle_scores.csv.gz",
                    usecols=["state_source", "segmentation", "session", "cycle_idx", "f", "role", "atype", "p1", "p2", "p3"])
    d = d[(d.state_source == "b2") & (d.segmentation == "pred") & (d.role == "test")].copy()
    S = {"s02": "session_02_normal_seq125", "s03": "session_03_normal_seq135", "s04": "session_04_normal_seq145",
         "s05": "session_05_anomaly_fast", "s06": "session_06_anomaly_slow"}
    GROUPS = {"s02 (normal)": ["s02"], "s03 (seq135)": ["s03"], "s04 (seq145)": ["s04"], "s05 (fast)": ["s05"], "s06 (slow)": ["s06"],
              "seq (s03+s04)": ["s03", "s04"], "tempo (s05+s06)": ["s05", "s06"]}
    CH = {"D1": "p1", "D2": "p2", "D3": "p3"}
    rng = np.random.RandomState(SEED)
    d["cl"] = d.session + "#" + d.cycle_idx.astype(str)
    rows = []
    boot_cache = {}
    cnt = {}
    for gname, ss in GROUPS.items():
        g = d[d.session.isin([S[x] for x in ss])]
        cl_ids = np.array(sorted(g.cl.unique()))
        cnt[gname] = (len(g), len(cl_ids))
        idx = rng.randint(0, len(cl_ids), size=(B_BOOT, len(cl_ids)))
        boot_cache[gname] = (g, cl_ids, idx)

    def per_cluster(g, cl_ids, col, alpha):
        a = (g[col] <= alpha + 1e-9).astype(int)
        return g.assign(a=a).groupby("cl").agg(k=("a", "sum"), n=("a", "size")).reindex(cl_ids)

    for ch, col in CH.items():
        for alpha in (0.10, 0.05):
            for gname in GROUPS:
                g, cl_ids, idx = boot_cache[gname]
                per = per_cluster(g, cl_ids, col, alpha)
                k, n = int(per.k.sum()), int(per.n.sum())
                kk, nn = per.k.to_numpy(float), per.n.to_numpy(float)
                bs = kk[idx].sum(1) / nn[idx].sum(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                wl, wh = wilson(k, n)
                rows.append(dict(kanal=ch, alpha=alpha, grup=gname, n_cift=n, n_dongu=len(cl_ids), n_alarm=k, oran=k / n,
                                 wilson_lo=wl, wilson_hi=wh, boot_lo=lo, boot_hi=hi))
    T = pd.DataFrame(rows)
    dec = []
    for ch, grs in (("D1", ["s05 (fast)", "s06 (slow)", "tempo (s05+s06)"]), ("D2", ["s03 (seq135)", "s04 (seq145)", "seq (s03+s04)"])):
        col = CH[ch]
        for alpha in (0.10, 0.05):
            nm = T[(T.kanal == ch) & (T.alpha == alpha) & (T.grup == "s02 (normal)")].iloc[0]
            gN, clN, idxN = boot_cache["s02 (normal)"]
            perN = per_cluster(gN, clN, col, alpha)
            bsN = perN.k.to_numpy(float)[idxN].sum(1) / perN.n.to_numpy(float)[idxN].sum(1)
            for gname in grs:
                r = T[(T.kanal == ch) & (T.alpha == alpha) & (T.grup == gname)].iloc[0]
                g, cl_ids, idx = boot_cache[gname]
                per = per_cluster(g, cl_ids, col, alpha)
                bs = per.k.to_numpy(float)[idx].sum(1) / per.n.to_numpy(float)[idx].sum(1)
                dlo, dhi = np.percentile(bs - bsN, [2.5, 97.5])
                inside = nm.wilson_lo - 1e-12 <= r.oran <= nm.wilson_hi + 1e-12
                overlap = not (r.boot_hi < nm.wilson_lo or r.boot_lo > nm.wilson_hi)
                dec.append(dict(kanal=ch, alpha=alpha, grup=gname, oran=r.oran, boot_lo=r.boot_lo, boot_hi=r.boot_hi,
                                normal_FAR=nm.oran, normal_wilson_lo=nm.wilson_lo, normal_wilson_hi=nm.wilson_hi,
                                fark_pp=100 * (r.oran - nm.oran), fark_boot_lo_pp=100 * dlo, fark_boot_hi_pp=100 * dhi,
                                oran_normal_GA_icinde=bool(inside), GA_ortusuyor=bool(overlap),
                                hukum=("oturum etkisi bu kanalda saptanamadi" if inside else
                                        ("oturum etkisi VAR (normal FAR GA'sinin ustunde: alarm fazlasi)" if r.oran > nm.oran else
                                         "oturum etkisi VAR (normal FAR GA'sinin ALTINDA: daha az alarm; yanlis alarm sisirmesi degil)"))))
    DEC = pd.DataFrame(dec)
    T.to_csv(OUT / "01_session_confound_b_rates.csv", index=False)
    DEC.to_csv(OUT / "01_session_confound_b_decision.csv", index=False)
    emit("Birim: (dongu, f) cifti (anomali donguleri her f'de test; normal s02 donguleri tek f'de). Alarm: kanal p <= alpha (+1e-9), p = Faz A standart conformal. "
         "Wilson GA havuzlu ciftler uzerinde (anomali donguleri 5 f'de tekrar ettigi icin dar kalir); bootstrap = dongu-kumeli (kume = oturum#dongu, tum f'leri birlikte), "
         f"B={B_BOOT}, tohum {SEED}, yuzdelik %95.")
    emit("")
    emit("Grup buyuklukleri (cift / tekil dongu): " + "; ".join(f"{k}: {v[0]}/{v[1]}" for k, v in cnt.items()))
    emit("")
    fm = lambda r: f"{r.oran:.3f} [{r.wilson_lo:.3f}, {r.wilson_hi:.3f}] / [{r.boot_lo:.3f}, {r.boot_hi:.3f}]"
    emit("### Tum kanal x grup oranlari (oran [Wilson GA] / [kume-bootstrap GA])")
    emit("")
    for ch in CH:
        emit(f"**{ch}** (karsi-kanal hucreleri: " + ("tempo oturumlari" if ch == "D1" else "sira oturumlari" if ch == "D2" else "yok; referans") + ")")
        emit("")
        tab = []
        for gname in GROUPS:
            row = {"grup": gname, "n_cift/n_dongu": f"{cnt[gname][0]}/{cnt[gname][1]}"}
            for alpha in (0.10, 0.05):
                r = T[(T.kanal == ch) & (T.alpha == alpha) & (T.grup == gname)].iloc[0]
                row[f"alpha={alpha:.2f}"] = fm(r)
            tab.append(row)
        emit(md_table(pd.DataFrame(tab)))
        emit("")
    emit("### Karsi kanal karari (on-yazili okuma: oran normal FAR'in Wilson GA'si ICINDEYSE 'saptanamadi', DISINDAYSA 'oturum etkisi var' + buyukluk)")
    emit("")
    emit("'Icinde' = karsi kanal oraninin nokta tahmini, ayni kanalin s02 normal FAR'inin Wilson %95 GA'si icinde "
         f"(n_normal = {cnt['s02 (normal)'][0]}). Ikincil: karsi kanal bootstrap GA'si ile normal Wilson GA'sinin ortusmesi; fark = oran - normal FAR, kume-bootstrap GA.")
    emit("")
    out = DEC.copy()
    out["oran [boot GA]"] = out.apply(lambda r: f"{r.oran:.3f} [{r.boot_lo:.3f}, {r.boot_hi:.3f}]", axis=1)
    out["normal FAR [Wilson]"] = out.apply(lambda r: f"{r.normal_FAR:.3f} [{r.normal_wilson_lo:.3f}, {r.normal_wilson_hi:.3f}]", axis=1)
    out["fark pp [boot GA]"] = out.apply(lambda r: f"{r.fark_pp:+.1f} [{r.fark_boot_lo_pp:+.1f}, {r.fark_boot_hi_pp:+.1f}]", axis=1)
    emit(md_table(out[["kanal", "alpha", "grup", "oran [boot GA]", "normal FAR [Wilson]", "fark pp [boot GA]",
                       "oran_normal_GA_icinde", "GA_ortusuyor", "hukum"]]))
    emit("")
    emit("Not (A4.2'ye atif): s05'teki (fast) D1 orani Box5 silinmesiyle (kisalmis dizi) kirlenmistir; temiz tempo-D1 kontrolu s06 (slow) satiridir. "
         "'tempo (s05+s06)' havuzu s05 kirliligini icerir.")
    emit("")
    return T, DEC


# ================================================================== (c)
def part_c():
    emit("## A4.1(c) Oturum basina adim sureleri (30 Hz ToF segmentleri; betimleyici)")
    emit("")
    ctx_ = ac.ctx()
    LAB = ac.source_labels(sources=("tof",), verbose=False)
    C, _ = ac.build_cycles("tof", "tof", LAB["tof"])
    exp = {s: pc.session_info(s)["expected_seq"] for s in ctx_.SESS}
    rec = []
    for s in ctx_.SESS:
        seg = pc.tof_segments(s)  # 30 Hz ToF, debounce 0.1 s / 0.15 s (probe varsayilani)
        Cs = C[C.session == s]
        for c in Cs.itertuples():
            g = seg[(seg.loop_id == c.cycle_idx)].sort_values("start")
            bx = g[g.label > 0].reset_index(drop=True)
            order = tuple(int(v) for v in bx.label)
            ok = order == tuple(exp[s])
            r = dict(session=s, cycle_idx=c.cycle_idx, atype=c.atype, cycle_s=c.dur_s, n_box_seg=len(bx),
                     order="-".join(map(str, order)), exact_seq=ok)
            if ok:
                r.update(dwell_box1=bx.duration_s[0], trans_1_to_2nd=bx.start[1] - bx.end[0], dwell_2nd=bx.duration_s[1],
                         trans_to_box5=bx.start[2] - bx.end[1], dwell_box5=bx.duration_s[2])
            rec.append(r)
    D = pd.DataFrame(rec)
    D.to_csv(OUT / "01_session_confound_c_cycles.csv", index=False)
    STEP = [("dwell_box1", "Box1 kalisi"), ("trans_1_to_2nd", "Box1->sonraki kutu gecisi"), ("dwell_2nd", "ikinci kutu kalisi"),
            ("trans_to_box5", "->Box5 gecisi"), ("dwell_box5", "Box5 kalisi"), ("cycle_s", "dongu suresi")]
    rows = []
    for s in ctx_.SESS:
        d = D[D.session == s]
        du = d[d.exact_seq]
        row = {"oturum": s[:10] + "(" + str(pc.session_info(s)["scenario"]) + ")", "n_dongu": len(d), "n_beklenen_dizi": len(du)}
        for col, nm in STEP:
            x = (d if col == "cycle_s" else du)[col].to_numpy(float)
            q = np.percentile(x, [25, 50, 75]) if len(x) else [np.nan] * 3
            row[nm] = f"{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]"
            row["_" + col] = q[1]
        rows.append(row)
    TT = pd.DataFrame(rows)
    emit("Hucre = medyan [Q25, Q75], saniye. Adim sureleri yalniz ToF dizisi tam beklenen diziyle (s02, s05, s06: 1-2-5; s03: 1-3-5; s04: 1-4-5) eslesen donguler icin; "
         "dongu suresi tum donguler icin (Faz A ToF dongu siniri, kare tablosu, 10 Hz cozunurluk). Gecis = onceki kutu segmentinin sonu ile sonraki kutu segmentinin basi arasi (ToF debounce: 0.1 s / 0.15 s).")
    emit("")
    emit(md_table(TT[[c for c in TT.columns if not c.startswith("_")]]))
    emit("")
    emit("s02'ye gore medyan orani (oturum medyani / s02 medyani):")
    emit("")
    base = TT.iloc[0]
    rr = []
    for i, r in TT.iterrows():
        row = {"oturum": r["oturum"]}
        for col, nm in STEP:
            row[nm] = f"{r['_' + col] / base['_' + col]:.2f}"
        rr.append(row)
    emit(md_table(pd.DataFrame(rr)))
    emit("")
    TT.drop(columns=[c for c in TT.columns if c.startswith("_")]).to_csv(OUT / "01_session_confound_c_steps.csv", index=False)
    emit("Dizisi beklenenle eslesmeyen donguler (sayim, oturum basina): " + "; ".join(
        f"{s[:10]}: {int((~D[D.session == s].exact_seq).sum())}" for s in ctx_.SESS))
    emit("")
    if LINK:
        PII = LINK[0]
        ref = D[(D.session == "session_02_normal_seq125") & D.exact_seq].dwell_box1.to_numpy(float)
        rr2 = []
        Cb = LINKC[0]
        Cn = Cb[Cb.atype == "normal"]
        ref_p = np.array([c_[0] for c_ in Cn.dw if len(c_)], float)       # s02 normal, B2-pred 1. ziyaret kalisi
        for (ses, cyc, at), g in PII.groupby(["session", "cycle_idx", "atype"]):
            x = D[(D.session == ses) & (D.cycle_idx == cyc)].iloc[0]
            dw = float(x.dwell_box1) if x.exact_seq else np.nan
            cr = Cb[(Cb.session == ses) & (Cb.cycle_idx == cyc)].iloc[0]
            j = int(g.trig_visit.iloc[0])
            dp = float(cr.dw[j])
            rr2.append(dict(tur=at, oturum=ses[:10], dongu=int(cyc), n_alarm_cifti=len(g), tetik="/".join(sorted(g.kind.unique())),
                            tetik_ziyaret_kutusu=int(g.trig_b.iloc[0]), B2_pred_dwell_s=dp,
                            s02_pred_yuzdelik=float((ref_p <= dp).mean() * 100),
                            s02_pred_aralik=f"[{ref_p.min():.2f}, {ref_p.max():.2f}]",
                            ToF_box1_dwell_s=dw, s02_ToF_yuzdelik=float((ref <= dw).mean() * 100) if dw == dw else np.nan,
                            s02_ToF_aralik=f"[{ref.min():.2f}, {ref.max():.2f}]",
                            kendi_oturum_ToF_medyan=float(D[(D.session == ses) & D.exact_seq].dwell_box1.median())))
        emit("(a) -> (c) baglantisi: (ii) kategorisindeki alarmlari veren tekil donguler ve Box1 kalis sureleri")
        emit("")
        emit(md_table(pd.DataFrame(rr2)))
        emit("")
    return D, TT


def main():
    emit("# A4.1 (M1) Anomali turu oturumla karisik mi?")
    emit("")
    emit("**Statu:** A4 ikincil analizdir (Faz A sayilari birincil kalir); on-kayit `faz_a4/DESIGN_A4.md` A4.1. Kaynak: `faz_a4/01_session_confound.py`. "
         "Yasak uyarisi: s02 dogal sapma dongulerinin alarm durumu ayrica incelenmedi; s02 normal Faz A FAR'i olarak butun kullanildi.")
    emit("")
    R, A = part_a()
    T, DEC = part_b()
    D, TT = part_c()
    (OUT / "01_session_confound.md").write_text("\n".join(MD), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()

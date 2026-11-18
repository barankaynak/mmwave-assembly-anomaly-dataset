"""Faz A3.3 - dongu ici (cevrimici) alarm; DESIGN_A3.md'deki on-kayitli tanimlar birebir.
Hucreler: B2 x pred (ana), ToF x tof (referans). Faz A kalibrasyon skorlari DEGISMEZ (ac.score_group ile ayni roller/parametreler).
Sert kontrol: kat basina cevrimici alarm kumesi == Faz A 04_cycle_scores p3 (role=test) alarm kumesi (alpha=0.10).
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
import a2_cycles as ac  # noqa: E402
fc = ac.fc
OUT = ROOT / "reports" / "faz_a3"
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.10
CELLS = [("b2", "pred"), ("tof", "tof")]           # (kaynak, segmentasyon); ilki ana
MD: list[str] = []


def emit(s=""):
    print(s)
    MD.append(s)


# ---------------------------------------------------------------- olaylar
def visit_list(c, ctx_, lab_deb):
    tt = ctx_.SE[c["session"]]["t"].to_numpy()
    vs = ac.visits_of(lab_deb, tt)
    sel = [v for v in vs if c["start_idx"] <= v[1] < c["end_idx"]]
    seq, dw, gp = ac.cycle_from(sel)
    assert tuple(seq) == tuple(c["seq"]) and np.allclose(dw, c["dw"], atol=1e-9) and np.allclose(gp, c["gp"], atol=1e-9)
    return vs, sel, tt


def next_box1_frame(vs, i_end_excl):
    """Sonraki Box1 girisi: start >= i_end_excl olan ilk Box1 ziyareti (debounce sonrasi)."""
    for v in vs:
        if v[0] == 1 and v[1] >= i_end_excl:
            return v
    return None


def build_events(c, par, vs, sel, tt, raw_lab, causal=False):
    """Olay listesi [(kare, sira, tur, d1_deger|None, d2_deger|None)]. causal=False: ON-KAYITLI (literal) tanim.
    causal=True : duyarlilik varyanti (onay = ham kutu etiketinin MIN_BOX'uncu karesi; ziyaret sonu e+MAX_GAP'te bilinir)."""
    n = len(tt)
    cap = lambda i: int(min(i, n - 1))
    d1c = par.d1
    ref = par.d2

    def d1v(a, b):
        return -math.log((d1c[a][b] + 1) / (sum(d1c[a].values()) + 6))

    def d2v(key, d):
        if key not in ref:
            return None
        v = math.log(d + ac.EPS)
        return abs(v - ref[key][0]) / ref[key][1]

    def conf(v):
        if not causal:
            return v[1] + ac.MIN_BOX - 1
        k = 0
        for i in range(v[1], v[2]):
            if raw_lab[i] == v[0]:
                k += 1
                if k == ac.MIN_BOX:
                    return i
        return v[1] + ac.MIN_BOX - 1                     # yedek

    endv = lambda v: v[2] + (ac.MAX_GAP if causal else 0)
    nxt = next_box1_frame(vs, c["end_idx"])
    if nxt is None:
        end_fr, end_fallback = n - 1, True
    else:
        end_fr, end_fallback = cap(conf(nxt)), False
    ev = []
    m = len(sel)
    if m == 0:
        ev.append((end_fr, 1, "trans_end", d1v("S", "E"), None))
        return ev, end_fallback, end_fr
    ev.append((cap(conf(sel[0])), 1, "trans_start", d1v("S", sel[0][0]), None))
    for j in range(m):
        b = sel[j][0]
        dwell = sel[j][4] - sel[j][3]
        fr = end_fr if j == m - 1 else cap(endv(sel[j]))
        ev.append((fr, 0, "dwell", None, d2v(f"dwell_{b}", dwell)))
        if j >= 1:
            a = sel[j - 1][0]
            fr2 = cap(conf(sel[j]))
            ev.append((fr2, 1, "trans_mid", d1v(a, b), None))
            gap = max(0.0, sel[j][3] - sel[j - 1][4])
            ev.append((fr2, 2, "gap", None, d2v(f"gap_{a}_{b}", gap)))
    ev.append((end_fr, 1, "trans_end", d1v(sel[-1][0], "E"), None))
    ev.sort(key=lambda x: (x[0], x[1]))
    return ev, end_fallback, end_fr


def run_online(ev, cal1, cal2):
    d1 = d2 = 0.0
    first = None
    for fr, _o, kind, a, b in ev:
        if a is not None:
            d1 = max(d1, a)
        if b is not None:
            d2 = max(d2, b)
        if first is None:
            p1 = float(ac.conf_p(cal1, [d1])[0])
            p2 = float(ac.conf_p(cal2, [d2])[0])
            if min(1.0, 2.0 * min(p1, p2)) <= ALPHA + 1e-9:
                first = (fr, kind)
    if first is None:
        return False, None, None, d1, d2
    return True, first[0], first[1], d1, d2


def onset_frame(c, active):
    if c["atype"] not in ("seq135", "seq145"):
        return None
    box = 3 if c["atype"] == "seq135" else 4
    a = active[c["session"]][box][c["start_idx"]:c["end_idx"]]
    k = np.flatnonzero(a)
    return int(c["start_idx"] + k[0]) if len(k) else None


def main():
    ctx_ = ac.ctx()
    LAB = ac.source_labels(sources=("tof", "b2"))
    # ToF aktif bayraklari (merged_df box3_active / box4_active) kare duzeyine
    active = {}
    for s in ctx_.SESS:
        m = fc.common.load_merged(s)
        pf = m.groupby("frame_id", sort=True)[["box3_active", "box4_active"]].max()
        fid = ctx_.SE[s]["frame_id"].to_numpy()
        active[s] = {3: pf["box3_active"].reindex(fid).fillna(0).to_numpy().astype(bool),
                     4: pf["box4_active"].reindex(fid).fillna(0).to_numpy().astype(bool)}
    OLD = pd.read_csv(ROOT / "reports" / "faz_a" / "04_cycle_scores.csv.gz",
                      usecols=["state_source", "segmentation", "session", "cycle_idx", "f", "role", "atype", "p3"])
    OLD = OLD[OLD["role"] == "test"]

    rows = []                      # (cycle, f) basina
    checks = []
    fail = False
    for src, seg in CELLS:
        C, _ = ac.build_cycles(src, seg, LAB[src])
        cl = C.to_dict("records")
        oldc = OLD[(OLD["state_source"] == src) & (OLD["segmentation"] == seg)].set_index(["session", "cycle_idx", "f"])
        for f in range(ac.N_FOLDS):
            ro = fc.anom_roles(f)
            tr, ca, te = ac._split_roles(cl, ro)
            par = ac.fit_params(tr)
            calS = ac.score_cycles(ca, par)
            cal1, cal2 = calS["D1"].to_numpy(float), calS["D2"].to_numpy(float)
            for c in te:
                lab_deb = ac.debounce_labels(LAB[src][c["session"]])
                vs, sel, tt = visit_list(c, ctx_, lab_deb)
                off = ac.score_cycle(c, par)
                rec = dict(state_source=src, segmentation=seg, session=c["session"], cycle_idx=c["cycle_idx"], f=f,
                           atype=c["atype"], n_visits=len(sel), dur_s=c["dur_s"], start_idx=c["start_idx"], end_idx=c["end_idx"])
                for var in ("lit", "causal"):
                    ev, fb, end_fr = build_events(c, par, vs, sel, tt, LAB[src][c["session"]], causal=(var == "causal"))
                    al, fr, kind, d1, d2 = run_online(ev, cal1, cal2)
                    # son maksimumlar == cevrimdisi skorlar (tanilama)
                    assert abs(d1 - off["D1"]) < 1e-12 and abs(d2 - off["D2"]) < 1e-12, (src, seg, c["session"], c["cycle_idx"], f, var, d1, off["D1"], d2, off["D2"])
                    rec.update({f"alarm_{var}": al, f"fr_{var}": fr, f"kind_{var}": kind, f"end_fr_{var}": end_fr})
                    t0 = tt[c["start_idx"]]
                    if var == "lit":
                        rec["end_fallback"] = fb
                        rec["t_alarm_lit"] = float(tt[fr]) if al else np.nan
                        rec["frac_lit"] = (tt[fr] - t0) / c["dur_s"] if al else np.nan
                        lv = sel[-1] if sel else None
                        rec["last_visit_start_fr"] = lv[1] if lv else -1
                        rec["last_visit_conf_fr"] = lv[1] + ac.MIN_BOX - 1 if lv else -1
                        oi = onset_frame(c, active)
                        rec["onset_fr"] = oi if oi is not None else -1
                        rec["onset_frac"] = (tt[oi] - t0) / c["dur_s"] if oi is not None else np.nan
                        rec["sec_from_onset_lit"] = float(tt[fr] - tt[oi]) if (al and oi is not None) else np.nan
                    else:
                        rec["frac_causal"] = (tt[fr] - t0) / c["dur_s"] if al else np.nan
                        oi = rec["onset_fr"]
                        rec["sec_from_onset_causal"] = float(tt[fr] - tt[oi]) if (al and oi >= 0) else np.nan
                # sert kontrol: Faz A p3
                p3 = float(oldc.loc[(c["session"], c["cycle_idx"], f), "p3"])
                rec["alarm_faza"] = bool(p3 <= ALPHA + 1e-9)
                rec["p3_faza"] = p3
                rows.append(rec)
    R = pd.DataFrame(rows)

    # ---- HARD CHECK
    emit("# A3.3 Cevrim ici (online) alarm - dongu icinde")
    emit("")
    emit("**Statu:** Faz A sonuclari BIRINCILdir; A3.3 sonradan yapilan IKINCIL bir analizdir (ana model B2 + D3 degismez).")
    emit("")
    emit("Kaynak: `faz_a3/03_online.py`; on-kayit: `faz_a3/DESIGN_A3.md` A3.3. Hucreler: B2 x pred (ana), ToF x tof (referans). "
         "Birim: (dongu, f) cifti; anomali donguleri her f'de test.")
    emit("")
    emit("## 0. SERT KONTROL - cevrimici alarm kumesi == Faz A cevrimdisi D3 alarm kumesi (alpha = 0.10, kat basina)")
    emit("")
    emit("| hucre | f | n_test | online alarm | Faz A alarm | fark (simetrik) | anahtar kumesi esit |")
    emit("|---|---|---|---|---|---|---|")
    for src, seg in CELLS:
        g = R[(R.state_source == src) & (R.segmentation == seg)]
        old = OLD[(OLD.state_source == src) & (OLD.segmentation == seg)]
        for f in range(ac.N_FOLDS):
            gf = g[g.f == f]
            of = old[old.f == f]
            keys_eq = set(zip(gf.session, gf.cycle_idx)) == set(zip(of.session, of.cycle_idx)) and len(gf) == len(of)
            a_on = set(zip(gf.session[gf.alarm_lit], gf.cycle_idx[gf.alarm_lit]))
            ofa = of[of.p3 <= ALPHA + 1e-9]
            a_of = set(zip(ofa.session, ofa.cycle_idx))
            diff = a_on ^ a_of
            checks.append(dict(cell=f"{src}x{seg}", f=f, n_test=len(gf), n_online=len(a_on), n_faza=len(a_of), n_diff=len(diff), keys_equal=keys_eq))
            emit(f"| {src} x {seg} | {f} | {len(gf)} | {len(a_on)} | {len(a_of)} | {len(diff)} | {keys_eq} |")
            if diff or not keys_eq:
                fail = True
                emit(f"MISMATCH {src}x{seg} f={f}: online-only={sorted(a_on - a_of)} faza-only={sorted(a_of - a_on)}")
        emit(f"- {src} x {seg}: sonraki Box1 girisi bulunamayan (son olaylari oturum sonunda skorlanan) cift sayisi: {int(g.end_fallback.sum())}")
        d = int((g.alarm_causal != g.alarm_lit).sum())
        emit(f"- {src} x {seg}: nedensel (causal) varyantin alarm kumesi literal ile farkli olan cift sayisi: {d}")
    pd.DataFrame(checks).to_csv(OUT / "03_online_check.csv", index=False)
    if fail:
        emit("")
        emit("**SERT KONTROL BASARISIZ - analiz durduruldu; yama yapilmadi.**")
        (OUT / "03_online.md").write_text("\n".join(MD), encoding="utf-8")
        print("HARD_CHECK=FAIL")
        sys.exit(1)
    emit("")
    emit("**SERT KONTROL GECTI** (tum hucre x kat; ayrica her cift icin son kosan maksimumlar D1/D2 cevrimdisi skorlarla 1e-12 icinde ayni).")
    R.to_csv(OUT / "03_online_alarms.csv", index=False)
    print("HARD_CHECK=PASS")

    # ---- METRIKLER
    def q(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return (np.nan, np.nan, np.nan)
        return (float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75)))

    summ = []
    TYPES = ["normal", "seq135", "seq145", "fast", "slow"]
    for src, seg in CELLS:
        g = R[(R.state_source == src) & (R.segmentation == seg)]
        for at in TYPES:
            d = g[g.atype == at]
            a = d[d.alarm_lit]
            row = dict(cell=f"{src}x{seg}", atype=at, n_pairs=len(d), n_cycles=d[["session", "cycle_idx"]].drop_duplicates().shape[0],
                       n_alarm=len(a), alarm_rate=len(a) / max(1, len(d)))
            m, lo, hi = q(a.frac_lit)
            row.update(frac_med=m, frac_q25=lo, frac_q75=hi)
            row["frac_gt1_share"] = float((a.frac_lit > 1.0).mean()) if len(a) else np.nan
            m, lo, hi = q(a.frac_causal)
            row.update(frac_causal_med=m, frac_causal_q25=lo, frac_causal_q75=hi)
            row["share_before_last_visit"] = float((a.fr_lit < a.last_visit_start_fr).mean()) if len(a) else np.nan
            row["share_before_last_visit_conf"] = float((a.fr_lit < a.last_visit_conf_fr).mean()) if len(a) else np.nan
            row["share_before_last_visit_causal"] = float((a.fr_causal < a.last_visit_start_fr).mean()) if len(a) else np.nan
            row["share_at_end_event"] = float((a.fr_lit >= a.end_fr_lit).mean()) if len(a) else np.nan
            if at in ("seq135", "seq145"):
                aon = a[a.onset_fr >= 0]
                row["n_alarm_with_onset"] = len(aon)
                m, lo, hi = q(aon.sec_from_onset_lit)
                row.update(sec_med=m, sec_q25=lo, sec_q75=hi)
                m, lo, hi = q(aon.sec_from_onset_causal)
                row.update(sec_causal_med=m, sec_causal_q25=lo, sec_causal_q75=hi)
                row["share_alarm_before_or_at_onset"] = float((aon.sec_from_onset_lit <= 1e-9).mean()) if len(aon) else np.nan
                m, lo, hi = q(d.onset_frac)
                row.update(onset_frac_med=m, onset_frac_q25=lo, onset_frac_q75=hi)
                row["n_pairs_no_onset"] = int((d.onset_fr < 0).sum())
            for k in a.kind_lit.value_counts().index:
                row["trig_" + str(k)] = int((a.kind_lit == k).sum())
            summ.append(row)
    SM = pd.DataFrame(summ)
    SM.to_csv(OUT / "03_online_summary.csv", index=False)

    f2 = lambda x: "nan" if pd.isna(x) else f"{x:.3f}"
    for cell in [f"{s}x{g}" for s, g in CELLS]:
        title = "ANA (dagitilabilir): B2 x pred" if cell == "b2xpred" else "REFERANS: ToF x tof"
        emit("")
        emit(f"## {title}")
        emit("")
        emit("### Alarm zamani - gecen dongu orani (literal tanim; (dongu,f) ciftleri havuzlu, yalniz alarm alan ciftler)")
        emit("")
        emit("| tur | n_cift | n_alarm | alarm orani | medyan oran | Q25 | Q75 | oran>1 payi | son ziyaretten ONCE verilen pay (alarm < son ziyaret baslangici) | ... (alarm < son ziyaret onayi) | son olayda (->END/son dwell) verilen pay |")
        emit("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in SM[SM.cell == cell].itertuples():
            emit(f"| {r.atype} | {r.n_pairs} | {r.n_alarm} | {f2(r.alarm_rate)} | {f2(r.frac_med)} | {f2(r.frac_q25)} | {f2(r.frac_q75)} | {f2(r.frac_gt1_share)} | {f2(r.share_before_last_visit)} | {f2(r.share_before_last_visit_conf)} | {f2(r.share_at_end_event)} |")
        emit("")
        emit("### Gercek baslangictan saniye (seq135/seq145; baslangic = box3/box4'e ilk ToF girisi, merged_df)")
        emit("")
        emit("| tur | n_alarm (baslangic var) | medyan s | Q25 | Q75 | alarm <= baslangic payi | baslangicin dongudeki oransal konumu (medyan [Q25,Q75]) | baslangici bulunamayan cift |")
        emit("|---|---|---|---|---|---|---|---|")
        for r in SM[(SM.cell == cell) & SM.atype.isin(["seq135", "seq145"])].itertuples():
            emit(f"| {r.atype} | {int(r.n_alarm_with_onset)} | {f2(r.sec_med)} | {f2(r.sec_q25)} | {f2(r.sec_q75)} | {f2(r.share_alarm_before_or_at_onset)} | {f2(r.onset_frac_med)} [{f2(r.onset_frac_q25)}, {f2(r.onset_frac_q75)}] | {int(r.n_pairs_no_onset)} |")
        emit("")
        emit("### Alarmi tetikleyen olay turu (sayi)")
        emit("")
        tc = [c for c in SM.columns if c.startswith("trig_")]
        emit("| tur | " + " | ".join(tc) + " |")
        emit("|---|" + "---|" * len(tc))
        for r in SM[SM.cell == cell].itertuples():
            emit(f"| {r.atype} | " + " | ".join(str(int(getattr(r, c))) if not pd.isna(getattr(r, c)) else "0" for c in tc) + " |")
        emit("")
        emit("### Duyarlilik: nedensel (causal) tanim (onay = ham etiketin MIN_BOX'uncu karesi; ziyaret sonu e+MAX_GAP'te bilinir) - alarm kumesi ayni")
        emit("")
        emit("| tur | medyan oran | Q25 | Q75 | son ziyaretten once pay | saniye medyan (seq) [Q25,Q75] |")
        emit("|---|---|---|---|---|---|")
        for r in SM[SM.cell == cell].itertuples():
            sec = f"{f2(r.sec_causal_med)} [{f2(r.sec_causal_q25)}, {f2(r.sec_causal_q75)}]" if r.atype in ("seq135", "seq145") else "-"
            emit(f"| {r.atype} | {f2(r.frac_causal_med)} | {f2(r.frac_causal_q25)} | {f2(r.frac_causal_q75)} | {f2(r.share_before_last_visit_causal)} | {sec} |")

    # ---- ON-KAYITLI TAHMINLER
    emit("")
    emit("## On-kayitli tahminler")
    emit("")
    emit("| tahmin | hucre | tur | gozlenen medyan | sonuc |")
    emit("|---|---|---|---|---|")
    for cell in ("b2xpred", "tofxtof"):
        for at, rule in (("seq135", "<0.5"), ("seq145", "<0.5"), ("fast", ">0.8"), ("slow", ">0.8")):
            r = SM[(SM.cell == cell) & (SM.atype == at)].iloc[0]
            ok = (r.frac_med < 0.5) if rule == "<0.5" else (r.frac_med > 0.8)
            emit(f"| medyan oran {rule} | {cell}{' (ana)' if cell == 'b2xpred' else ' (ref)'} | {at} | {f2(r.frac_med)} | {'TUTTU' if ok else 'TUTMADI'} |")
    # ---- SEKIL
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"seq135": "#0072B2", "seq145": "#009E73", "fast": "#D55E00", "slow": "#CC79A7", "normal": "#7f7f7f"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a_, (src, seg), ttl in zip(ax, CELLS, ["B2 x pred (ana)", "ToF x tof (referans)"]):
        g = R[(R.state_source == src) & (R.segmentation == seg) & R.alarm_lit]
        for at in TYPES:
            x = np.sort(g[g.atype == at].frac_lit.to_numpy())
            if len(x):
                a_.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=cols[at], label=f"{at} (n={len(x)})",
                        lw=2 if at != "normal" else 1.2, ls="-" if at != "normal" else "--")
        a_.axvline(0.5, color="k", lw=0.6, ls=":")
        a_.axvline(0.8, color="k", lw=0.6, ls=":")
        a_.set_xlabel("alarm aninda gecen dongu orani")
        a_.set_ylabel("alarm alan ciftlerde kumulatif pay")
        a_.set_title(ttl)
        a_.set_xlim(0, 1.3)
        a_.grid(alpha=0.25)
        a_.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "03_online.png", dpi=140)
    plt.close(fig)
    emit("")
    emit("Yorum (sonuclarin okunusu): slow tempo B2 x pred'de alarmlarin cogunda ilk ziyaretler arasi bosluk (gap, Box2 onayinda skorlanir) ile ~dongunun 0.3'unde yakalaniyor; "
         "yani 'slow gec yakalanir' tahmini TUTMADI. fast ise yalniz dongu tamamlaninca (son dwell / ->END, sonraki Box1 onayinda skorlanir) yakalaniyor (medyan oran ~1.0). "
         "DESIGN cumlesindeki 'tempo sapmalari ancak dongu sonuna dogru' yalniz fast icin dogru; slow icin dogru degil.")
    emit("fast'te oran > 1 (~1.02-1.03): son olay sonraki Box1 ziyaretinin onay karesinde (dongu sinirindan MIN_BOX-1 = 1 kare = 0.1 s sonra) skorlandigi icin; 1 + 0.1/dur_s ile uyumlu, literal ->END kuraliyla bekleniyor.")
    for cell in ("b2xpred", "tofxtof"):
        src, seg = ("b2", "pred") if cell == "b2xpred" else ("tof", "tof")
        for at in ("seq135", "seq145"):
            d = R[(R.state_source == src) & (R.segmentation == seg) & (R.atype == at) & R.alarm_lit & (R.onset_fr >= 0)]
            e = d[d.sec_from_onset_lit <= 1e-9]
            vc = e.kind_lit.value_counts().to_dict()
            emit(f"- {cell} {at}: baslangic anina esit/oncesi verilen alarm {len(e)}/{len(d)}; tetikleyen olay dagilimi: {vc}")
    emit("- Baslangictan saniye medyani ~0.1 s, 10 Hz'de MIN_BOX onay gecikmesinin (onay = ziyaret baslangici + 1 kare) kendisidir; ToF x tof'ta tum ceyreklikler tam 0.100 s cikmasi bundandir. "
         "Dolayisiyla saniye degeri 'tespit gecikmesi' degil, onay kurali tabanidir.")
    emit("")
    emit("Sekil: `reports/faz_a3/03_online.png` (alarm anindaki dongu orani, kumulatif dagilim; noktali cizgiler 0.5 ve 0.8).")

    # ---- TANIMLAR / UYARILAR
    emit("")
    emit("## Uygulanan tanimlar (kare duzeyinde) ve belirsizlik giderimleri")
    emit("")
    emit("- Zaman = OOF/kare tablosu `t` (oturumun ilk karesinden saniye; OOF ile kare tablosu birebir ayni, maks fark 0). Oran = (t_alarm - t[start_idx]) / dur_s.")
    emit("- Ziyaret = Faz A debounce'lu etiket kosusu (MIN_BOX=2, MAX_GAP=1); D1/D2 parametreleri ve kalibrasyon skorlari Faz A rolleriyle (egit f+3,f+4; kal f+1,f+2) ayni.")
    emit("- Literal (on-kayitli) onay: b ziyareti = debounce'lu kosunun MIN_BOX'uncu karesi (s+1); ziyaret sonu: ilk ziyaret-sonrasi kare e. Gecis a->b ve bosluk onay karesinde; dwell_a, e karesinde.")
    emit("- ->END gecisi ve son dwell: sonraki Box1 ziyaretinin onay karesinde (pred segmentasyonunda bu sonraki dongunun ilk ziyareti = dongu siniri). Dongu orani bu yuzden 1'i biraz asabilir (`oran>1 payi`).")
    emit("- ToF x tof: sonraki Box1 girisi = dongu sonu (end_idx) sonrasi ilk Box1 ziyareti; oturumun son dongusunde (sonraki giris yok) son olaylar oturumun son karesinde skorlanir (`end_fallback`).")
    emit("- Alarm = ilk olayda 2*min(p1,p2) <= 0.10 (+1e-9), p = Faz A standart conformal, kosan maksimum D1 (Laplace, START baglami) ve D2 (log(d+0.05) z, min-ref 3). Asim kurali yok.")
    emit("- Anomali (cycle,f) ciftleri 5 f'de tekrar eder (her f'nin kalibrasyonu farkli); metrikler ciftler uzerinden havuzlanmistir, dongu basina degil.")
    emit("- 'Normal' satiri: yanlis alarm veren normal dongulerin zamanlamasi; dogal-sapma donguleri ayristirilmamistir (on-kayit yasagi).")
    (OUT / "03_online.md").write_text("\n".join(MD), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()

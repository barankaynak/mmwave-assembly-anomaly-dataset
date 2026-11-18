"""Faz D / D.2 - Sizinti tablosu, butun CNN-GRU kosulari RTX 3090'da. Plan: faz_d/DESIGN_D.md.

A2.2 (faz_a2/02_leakage.py) ile ayni metrikler (fa_common.state_metrics, kat basina; havuzlama yok) ve ayni Welch araligi.
- B1/B2 satirlari: A2.2'nin 02_leakage_per_run.csv'sinden aynen (CPU, deterministik).
- CNN-GRU S1: Faz A OOF (3090). CNN-GRU S0: tohum 42 Faz A OOF (3090) + tohum 1-4 --source klasorunden.
Kapilar (gecmezse cikis kodu != 0, rapor yazilmaz):
  K1 her S0 OOF'unun test pencereleri (oturum, frame_id) ve y_true ilgili tohum bolmesiyle birebir ayni (A2.2'deki assert);
  K2 tohum 1-4'te test pencereleri ve y_true A2.2'nin GTX OOF dosyalariyla birebir ayni;
  K3 yontem kapisi: --source a2 ile ayni kod A2.2'nin 02_leakage_summary.csv'sini birebir uretir (|fark| < 1e-12);
  K4 (--source d) 01_env.json'da GPU adi '3090' icerir ve 01_cnn_s0_runs.csv'de 4 tohumun device'i 'cuda'.
Kullanim: python faz_d/02_leakage_3090.py --source d     (gercek analiz -> reports/faz_d/02_*)
          python faz_d/02_leakage_3090.py --source a2    (yalniz K3 yontem kapisi; dosya yazmaz)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
import numpy as np
import pandas as pd
from scipy import stats
import a2_s0 as s0

fc = s0.fc
OUTD = ROOT / "reports" / "faz_d"
A2 = s0.OUT2
A4 = ROOT / "reports" / "faz_a4"
SEEDS_NEW = [1, 2, 3, 4]
LINES: list[str] = []


def out(s=""):
    LINES.append(s)
    print(s, flush=True)


def metr(df):
    return fc.state_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())


def welch(x0, x1, level=0.95):          # faz_a2/02_leakage.py ile birebir
    n0, n1 = len(x0), len(x1)
    m0, m1 = np.mean(x0), np.mean(x1)
    v0, v1 = np.var(x0, ddof=1), np.var(x1, ddof=1)
    se = np.sqrt(v0 / n0 + v1 / n1)
    df = (v0 / n0 + v1 / n1) ** 2 / ((v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1))
    t = stats.t.ppf(0.5 + level / 2, df)
    d = m0 - m1
    tstat = d / se
    p = 2 * stats.t.sf(abs(tstat), df)
    return dict(m0=m0, s0=np.sqrt(v0), n0=n0, m1=m1, s1=np.sqrt(v1), n1=n1, d=d, se=se, df=df,
                lo=d - t * se, hi=d + t * se, t=tstat, p=p)


def cnn_rows(W, src: str):
    rows = []
    for f in range(fc.N_FOLDS):
        m = metr(pd.read_csv(fc.oof_path("cnngru", "S1", f)))
        rows.append({"model": "cnngru", "split": "S1", "run": f, "n": m["n"], "acc": m["acc"], "macro_f1": m["macro_f1"]})
    for seed in [42] + SEEDS_NEW:
        if seed == 42:
            p = fc.oof_path("cnngru", "S0", 0)
        elif src == "a2":
            p = s0.oof2_path("cnngru", seed)
        else:
            p = OUTD / "oof" / f"cnngru__S0s{seed}__f0.csv.gz"
        if not p.exists():
            raise SystemExit(f"EKSIK: {p}")
        df = pd.read_csv(p)
        te = s0.s0_test_idx(W, seed)
        exp = W.iloc[te][["session", "frame_id"]].reset_index(drop=True)
        got = df[["session", "frame_id"]].reset_index(drop=True)
        assert exp.equals(got), f"K1: {p.name} test pencereleri tohum {seed} bolmesiyle ayni degil"
        assert (df["y_true"].to_numpy() == W.iloc[te]["y_true"].to_numpy()).all(), f"K1: {p.name} y_true"
        if src == "d" and seed != 42:
            g = pd.read_csv(s0.oof2_path("cnngru", seed))
            assert g[["session", "frame_id"]].reset_index(drop=True).equals(got), f"K2: {p.name} pencereler A2.2 GTX ile ayni degil"
            assert (g["y_true"].to_numpy() == df["y_true"].to_numpy()).all(), f"K2: {p.name} y_true"
        m = metr(df)
        rows.append({"model": "cnngru", "split": "S0", "run": seed, "n": m["n"], "acc": m["acc"], "macro_f1": m["macro_f1"]})
    return rows


def summarize(R):
    S = []
    for model in ("b1", "b2", "cnngru"):
        for mk in ("acc", "macro_f1"):
            x0 = R[(R.model == model) & (R.split == "S0")][mk].to_numpy()
            x1 = R[(R.model == model) & (R.split == "S1")][mk].to_numpy()
            w = welch(x0, x1)
            w.update(model=model, metric=mk)
            S.append(w)
    return pd.DataFrame(S)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["a2", "d"], required=True)
    a = ap.parse_args()
    W = fc.windows()
    base = pd.read_csv(A2 / "02_leakage_per_run.csv")
    R = pd.concat([base[base.model.isin(["b1", "b2"])], pd.DataFrame(cnn_rows(W, a.source))], ignore_index=True)
    SM = summarize(R)

    # K3 yontem kapisi
    if a.source == "a2":
        ref = pd.read_csv(A2 / "02_leakage_summary.csv")
        cols = ["m0", "s0", "m1", "s1", "d", "se", "df", "lo", "hi", "t", "p"]
        mx = 0.0
        for _, r in ref.iterrows():
            g = SM[(SM.model == r.model) & (SM.metric == r.metric)].iloc[0]
            mx = max(mx, max(abs(float(g[c]) - float(r[c])) for c in cols))
        ok = mx < 1e-12
        print(f"K3 yontem kapisi: A2.2 ozeti yeniden uretildi, max |fark| = {mx:.3e} -> {'GECTI' if ok else 'KALDI'}")
        return 0 if ok else 5

    # K4 ortam kapisi
    env = json.loads((OUTD / "01_env.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(OUTD / "01_cnn_s0_runs.csv")
    runs = runs[runs.seed.isin(SEEDS_NEW)].drop_duplicates("seed", keep="last")
    k4 = ("3090" in env["gpu"]) and len(runs) == 4 and (runs.device == "cuda").all()
    if not k4:
        print(f"K4 KALDI: env={env} runs={runs[['seed', 'device']].to_dict('records')}")
        return 6

    OUTD.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUTD / "02_leakage_per_run.csv", index=False)
    SM.to_csv(OUTD / "02_leakage_summary.csv", index=False)

    # Keras tek kosu (A4.3b, 3090, degismedi)
    kp = pd.read_csv(A4 / "03_leakage_matched_keras_per_run.csv")
    k0 = kp[(kp.model == "keras") & (kp.split == "S0")]
    k1 = kp[(kp.model == "keras") & (kp.split == "S1")]
    kd = {mk: float(k0[mk].iloc[0] - k1[mk].mean()) for mk in ("acc", "macro_f1")}
    a2s = pd.read_csv(A2 / "02_leakage_summary.csv")

    out("# Faz D — Sızıntı tablosu, bütün CNN-GRU koşuları RTX 3090'da")
    out("")
    out(f"Plan: `faz_d/DESIGN_D.md`. Üreten: `faz_d/02_leakage_3090.py --source d`. Koşu: `faz_d/01_cnn_s0_seeds_3090.py`.")
    out(f"Ortam: {env['gpu']}, torch {env['torch']}, CUDA {env['cuda']}, cuDNN {env['cudnn']}.")
    out("")
    out("## Kapılar")
    out("- K1: her S0 OOF dosyasının test pencereleri ve y_true ilgili tohumun bölmesiyle birebir aynı (assert) — GEÇTİ.")
    out("- K2: tohum 1–4'te test pencereleri ve y_true, A2.2'nin GTX OOF dosyalarıyla birebir aynı (assert) — GEÇTİ.")
    out("- K3: yöntem kapısı ayrıca `--source a2` ile koşulur (A2.2 özetini birebir üretmeli).")
    out(f"- K4: GPU '{env['gpu']}', 4 tohumun device'ı cuda — GEÇTİ.")
    out("")
    out("## Koşu başına (CNN-GRU)")
    out("| bölme | koşu | n | doğruluk | macro-F1 |")
    out("|---|---|---|---|---|")
    for _, r in R[R.model == "cnngru"].iterrows():
        out(f"| {r.split} | {int(r.run)} | {int(r.n)} | {100 * r.acc:.2f} | {r.macro_f1:.4f} |")
    out("")
    out("## Özet (S0 beş tohum ortalaması − S1 beş kat ortalaması; Welch %95 GA)")
    out("| model | metrik | S0 ort ± sd | S1 ort ± sd | etki (puan) | %95 GA | A2.2 karışık-GPU etkisi |")
    out("|---|---|---|---|---|---|---|")
    for _, r in SM.iterrows():
        old = a2s[(a2s.model == r.model) & (a2s.metric == r.metric)].iloc[0]
        out(f"| {r.model} | {r.metric} | {100 * r.m0:.2f} ± {100 * r.s0:.2f} | {100 * r.m1:.2f} ± {100 * r.s1:.2f} | "
            f"**{100 * r.d:+.2f}** | [{100 * r.lo:+.2f}, {100 * r.hi:+.2f}] | {100 * old.d:+.2f} |")
    out("")
    out("## Okuma kuralı (DESIGN_D, koşudan önce)")
    c = SM[(SM.model == "cnngru")].set_index("metric")
    out(f"- Başlık değeri (PyTorch portu, tamamen 3090, beş tohum): doğruluk **{100 * c.loc['acc', 'd']:+.2f}** puan "
        f"[{100 * c.loc['acc', 'lo']:+.2f}, {100 * c.loc['acc', 'hi']:+.2f}], macro-F1 **{100 * c.loc['macro_f1', 'd']:+.2f}** "
        f"[{100 * c.loc['macro_f1', 'lo']:+.2f}, {100 * c.loc['macro_f1', 'hi']:+.2f}].")
    out(f"- Keras (tek koşu, 3090, A4.3b'den değişmeden): doğruluk {100 * kd['acc']:+.2f}, macro-F1 {100 * kd['macro_f1']:+.2f} puan.")
    s0n = R[(R.model == "cnngru") & (R.split == "S0") & (R.run.isin(SEEDS_NEW))]["acc"]
    out("")
    out("## Önceden yazılan beklentiler (betimsel)")
    ina = bool(((s0n >= 0.895) & (s0n <= 0.930)).all())
    inb = bool(0.02 <= c.loc['acc', 'd'] <= 0.06)
    out(f"- (a) 3090 S0 tohum 1–4 doğrulukları %89,5–93,0 içinde: {'TUTTU' if ina else 'TUTMADI'} "
        f"({', '.join(f'{100 * x:.2f}' for x in s0n)}).")
    out(f"- (b) beş tohumlu doğruluk etkisi +2 ile +6 puan arasında: {'TUTTU' if inb else 'TUTMADI'} ({100 * c.loc['acc', 'd']:+.2f}).")
    (OUTD / "02_leakage_3090.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

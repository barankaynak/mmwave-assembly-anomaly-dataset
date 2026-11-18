"""Faz D / D.1 - CNN-GRU (PyTorch), S0 tohum 1-4, tek GPU (RTX 3090). Plan: faz_d/DESIGN_D.md.

faz_a2/12_cnn_s0_seeds.py modulu DEGISTIRILMEDEN ice aktarilir (mimari, egitim, erken durdurma, ic dogrulama, tohum semasi aynen):
bolme random_state = tohum; ic dogrulama sinif-tabakali %10, random_state = tohum; model tohumu = 20261104 + tohum.
Yalniz cikti yollari degisir (A2 dosyalarina yazilmaz):
  OOF   -> reports/faz_d/oof/cnngru__S0s{tohum}__f0.csv.gz
  ozet  -> reports/faz_d/01_cnn_s0_runs.csv
  ortam -> reports/faz_d/01_env.json (torch, CUDA, cuDNN surumu, GPU adi)
CPU'ya dusme YOK: CUDA yoksa ya da GPU adi --require-gpu icermiyorsa cikis kodu 3/4 (plan: CPU koşusu gecersiz).
Smoke: --smoke -> CACHE_DIR/smoke_d1 altina yazar, varsayilan 1 epoch, --require-gpu kontrolu atlanir.
Kullanim: python faz_d/01_cnn_s0_seeds_3090.py [--seeds 1,2,3,4] [--max-epochs N] [--require-gpu 3090] [--smoke]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
spec = importlib.util.spec_from_file_location("a2_12_cnn_s0_seeds", ROOT / "faz_a2" / "12_cnn_s0_seeds.py")
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m12)

import torch  # noqa: E402

log = m12.log
OUTD = ROOT / "reports" / "faz_d"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--require-gpu", default="3090")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    assert 42 not in seeds, "tohum 42 Faz A'dan (3090); yeniden egitilmez"
    if a.smoke:
        out_dir = m12.fa.CACHE_DIR / "smoke_d1"
        runs_csv = out_dir / "runs.csv"
        env_json = out_dir / "env.json"
        max_ep = a.max_epochs or 1
    else:
        out_dir = OUTD / "oof"
        runs_csv = OUTD / "01_cnn_s0_runs.csv"
        env_json = OUTD / "01_env.json"
        max_ep = a.max_epochs or m12.m01.MAX_EPOCHS
    out_dir.mkdir(parents=True, exist_ok=True)
    m12.RUNS_CSV = runs_csv
    m12.OOF_OVERRIDE_DIR = out_dir

    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        log("CUDA yok -> plan geregi CPU'da kosulmaz")
        return 3
    gpu = torch.cuda.get_device_name(0)
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "gpu": gpu, "max_epochs": max_ep, "seeds": seeds, "smoke": a.smoke}
    log(f"ortam {env} free={torch.cuda.mem_get_info()[0] / 2**30:.2f} GB")
    if not a.smoke and a.require_gpu and a.require_gpu not in gpu:
        log(f"GPU '{gpu}' istenen '{a.require_gpu}' degil -> cikis")
        return 4
    env_json.write_text(json.dumps(env, indent=1), encoding="utf-8")

    data = m12.m01.Data()
    log(f"windows={len(data.win)} hm={tuple(data.hm.shape)}")
    m12.s0.assert_seed42_matches_faz_a(data.win)
    log("seed 42 S0 split == Faz A (assert OK)")
    dev = torch.device("cuda")
    for seed in seeds:
        target = out_dir / f"cnngru__S0s{seed}__f0.csv.gz"
        if target.exists() and not a.smoke:
            log(f"[cnngru S0s{seed}] OOF var -> atla")
            continue
        m12.train_seed(seed, data, dev, max_ep)   # hata olursa durur; CPU'ya dusme yok
        torch.cuda.empty_cache()
    log("all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

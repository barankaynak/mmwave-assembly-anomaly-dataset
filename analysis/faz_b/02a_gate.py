"""B.2 kapi: 'merged/tam' seviyesi fa_common.frame_table() istatistiklerini birebir yeniden uretmeli."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import b_frames as bf, fa_common as fc
ft = fc.frame_table()
st = bf.frame_stats(dict(tau_db=15.0, vel="full", pts="merged", group=False))
print("frames", len(ft), len(st), bin_info if False else bf.bin_info())
tot = 0.0
for c in bf.STAT_COLS:
    a = ft[c].to_numpy(float); b = st[c].to_numpy(float)
    nanmatch = (np.isnan(a) == np.isnan(b)).all()
    d = np.nanmax(np.abs(a - b)); tot = max(tot, d)
    print(f"{c:11s} max|diff|={d:.3e} nan_match={nanmatch} nan_n={int(np.isnan(a).sum())} exact={(np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))}")
print("MAX|diff| over 7 stats:", tot)

# Phase E - 03: intervals for the one-shot E1 rates (descriptive) [S]

Source `reports/faz_e/02_oneshot_cycles.csv.gz`; conventions as `paper/scripts/gap2_wilson.py` (Wilson for the normal cycles; cluster bootstrap over cycles, 10,000 resamples, stratified by type, RandomState(20261104), for anomaly rates over (cycle, fold) pairs; Clopper-Pearson over cycles where every cycle alarmed in every fold).

| rule | group | k | pairs | cycles | rate | 95% lo | 95% hi | interval | cycles alarmed in all folds | CP lo (cycles) |
|---|---|---|---|---|---|---|---|---|---|---|
| Phase A D3 | normal (FAR) | 4 | 51 | 51 | 0.0784 | 0.0309 | 0.1850 | Wilson over test cycles |  |  |
| Phase A D3 | seq135 | 71 | 280 | 56 | 0.2536 | 0.2107 | 0.3107 | cluster bootstrap over cycles | 3 |  |
| Phase A D3 | seq145 | 114 | 240 | 48 | 0.4750 | 0.4250 | 0.5375 | cluster bootstrap over cycles | 6 |  |
| Phase A D3 | fast | 221 | 240 | 48 | 0.9208 | 0.8625 | 0.9667 | cluster bootstrap over cycles | 36 |  |
| Phase A D3 | slow | 237 | 255 | 51 | 0.9294 | 0.8745 | 0.9765 | cluster bootstrap over cycles | 43 |  |
| Phase A D3 | seq(135+145) | 185 | 520 | 104 | 0.3558 | 0.3212 | 0.3962 | cluster bootstrap over cycles | 9 |  |
| Phase A D3 | tempo(fast+slow) | 458 | 495 | 99 | 0.9253 | 0.8848 | 0.9596 | cluster bootstrap over cycles | 79 |  |
| Phase A D3 | all anomalies | 643 | 1015 | 203 | 0.6335 | 0.6079 | 0.6601 | cluster bootstrap over cycles | 88 |  |
| E1 D3' | normal (FAR) | 4 | 51 | 51 | 0.0784 | 0.0309 | 0.1850 | Wilson over test cycles |  |  |
| E1 D3' | seq135 | 280 | 280 | 56 | 1.0000 | 1.0000 | 1.0000 | cluster bootstrap over cycles | 56 | 0.9362 |
| E1 D3' | seq145 | 240 | 240 | 48 | 1.0000 | 1.0000 | 1.0000 | cluster bootstrap over cycles | 48 | 0.9260 |
| E1 D3' | fast | 231 | 240 | 48 | 0.9625 | 0.9042 | 1.0000 | cluster bootstrap over cycles | 46 |  |
| E1 D3' | slow | 237 | 255 | 51 | 0.9294 | 0.8745 | 0.9765 | cluster bootstrap over cycles | 43 |  |
| E1 D3' | seq(135+145) | 520 | 520 | 104 | 1.0000 | 1.0000 | 1.0000 | cluster bootstrap over cycles | 104 | 0.9652 |
| E1 D3' | tempo(fast+slow) | 468 | 495 | 99 | 0.9455 | 0.9051 | 0.9778 | cluster bootstrap over cycles | 89 |  |
| E1 D3' | all anomalies | 988 | 1015 | 203 | 0.9734 | 0.9547 | 0.9892 | cluster bootstrap over cycles | 193 |  |

## Table metrics (definitions as the Phase A anomaly table; Phase A D3 row re-derived and asserted)

| rule | FAR % (0.10) | seq % | tempo % | all % | all % (0.05) | AUROC seq | AUROC tempo | AUROC all (sd) |
|---|---|---|---|---|---|---|---|---|
| Phase A D3 | 7.8 | 35.6 | 92.5 | 63.3 | 0.0 | 0.897 | 0.952 | 0.924 (0.035) |
| E1 S_E alone | 2.0 | 100.0 | 13.1 | 57.6 | 57.6 | 1.000 | 0.556 | 0.783 (0.011) |
| E1 D3' | 7.8 | 100.0 | 94.5 | 97.3 | 0.0 | 0.960 | 0.954 | 0.957 (0.023) |

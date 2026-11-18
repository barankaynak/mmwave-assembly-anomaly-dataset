# 05 D0 and D2 coverage on the sequence sessions (descriptive; 2026-11-10)

Source: reports/faz_a/04_cycle_scores.csv.gz, B2 x pred, test rows, alpha = 0.10.

## Steps scored by D2 (d2_nref), per (cycle, fold) pair
- normal: 51 pairs; d2_nref counts {4: 1, 5: 47, 6: 3}
- seq135: 280 pairs; d2_nref counts {2: 275, 3: 5}
- seq145: 240 pairs; d2_nref counts {2: 235, 3: 5}
- fast: 240 pairs; d2_nref counts {3: 60, 4: 5, 5: 175}
- slow: 255 pairs; d2_nref counts {5: 250, 6: 5}

## D0 alone at alpha = 0.10
- normal: 5/51 = 0.098 (Wilson 0.043-0.210)
- seq135: 68/280 = 0.243
- seq145: 74/240 = 0.308
- sequence pooled: 142/520 = 0.273 (cycle-clustered bootstrap over 104 cycles, 4000 draws: 0.204-0.346)
- D0-alarmed sequence pairs shorter than the fold's mean training cycle: 126/142 = 0.887

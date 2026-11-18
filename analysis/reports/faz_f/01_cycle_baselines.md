# Faz F / 01 Cycle-level baselines (descriptive; 2026-11-10)

Plan sha256: d9c8d7e7f4f82d6892584a1b088766b12d500b9a24bd2d9e134fb166142de1e9

Reproduction: D0, D1, D2 and p3 equal reports/faz_a/04_cycle_scores.csv.gz (B2 x pred, all test rows).

## Results at alpha = 0.10 (FAR pooled over the 51 normal test cycles; detection over (cycle, fold) pairs)

```
                       detector  FAR_0.10  det_seq135_0.10  det_seq145_0.10  det_fast_0.10  det_slow_0.10  det_seq_0.10  det_tempo_0.10  det_all_0.10  auroc_seq  auroc_tempo  auroc_all
                       D0 alone     0.098            0.243            0.308          0.787          1.000         0.273           0.897         0.577      0.610        0.954      0.778
                       D1 alone     0.059            0.604            0.800          0.217          0.008         0.694           0.109         0.409      0.974        0.535      0.760
                       D2 alone     0.078            0.068            0.125          0.800          0.992         0.094           0.899         0.487      0.508        0.961      0.729
                   D3 (primary)     0.078            0.254            0.475          0.921          0.929         0.356           0.925         0.633      0.897        0.952      0.924
                      LEV alone     0.059            0.421            0.412          0.108          0.008         0.417           0.057         0.241      0.962        0.531      0.752
                       LEV + D2     0.059            0.086            0.125          0.771          0.929         0.104           0.853         0.469      0.875        0.955      0.914
                      IF seed 0     0.118            0.300            0.138          0.329          0.788         0.225           0.566         0.391      0.636        0.897      0.763
                      IF seed 1     0.098            0.343            0.154          0.388          0.792         0.256           0.596         0.422      0.653        0.901      0.774
                      IF seed 2     0.098            0.229            0.042          0.092          0.788         0.142           0.451         0.293      0.629        0.848      0.736
                      IF seed 3     0.078            0.207            0.071          0.113          0.792         0.144           0.463         0.300      0.618        0.826      0.719
                      IF seed 4     0.118            0.343            0.154          0.388          0.788         0.256           0.594         0.421      0.642        0.888      0.762
R0 exact route (no calibration)     0.078            1.000            1.000          0.271          0.020         1.000           0.141         0.581        NaN          NaN        NaN
```

## Isolation forest over seeds 0-4 (mean, min-max)

- FAR_0.10: mean 0.102, range 0.078-0.118
- det_seq135_0.10: mean 0.284, range 0.207-0.343
- det_seq145_0.10: mean 0.112, range 0.042-0.154
- det_fast_0.10: mean 0.262, range 0.092-0.388
- det_slow_0.10: mean 0.790, range 0.788-0.792
- det_seq_0.10: mean 0.205, range 0.142-0.256
- det_tempo_0.10: mean 0.534, range 0.451-0.596
- det_all_0.10: mean 0.365, range 0.293-0.422
- auroc_seq: mean 0.636, range 0.618-0.653
- auroc_tempo: mean 0.872, range 0.826-0.901
- auroc_all: mean 0.751, range 0.719-0.774

## Per fold (LEV)

```
   n_cal  cal_lev_ge1  det_seq_LEValone  det_seq_LEVD2
f                                                     
0     19            2             0.029          0.106
1     20            1             1.000          0.096
2     21            2             0.029          0.106
3     22            2             0.029          0.106
4     20            1             1.000          0.106
```

## LEV distance of anomaly test pairs

```
lev       0    1   2
atype               
fast    175   65   0
seq135    0  270  10
seq145    0  235   5
slow    250    5   0
```

## alpha = 0.05

```
                       detector  FAR_0.05  det_seq135_0.05  det_seq145_0.05  det_fast_0.05  det_slow_0.05  det_seq_0.05  det_tempo_0.05  det_all_0.05
                       D0 alone     0.059            0.168            0.183          0.662          1.000         0.175           0.836         0.498
                       D1 alone     0.020            0.211            0.400          0.217          0.000         0.298           0.105         0.204
                       D2 alone     0.059            0.054            0.125          0.771          0.929         0.087           0.853         0.460
                   D3 (primary)     0.000            0.000            0.000          0.000          0.000         0.000           0.000         0.000
                      LEV alone     0.000            0.036            0.021          0.000          0.000         0.029           0.000         0.015
                       LEV + D2     0.000            0.000            0.000          0.000          0.000         0.000           0.000         0.000
                      IF seed 0     0.039            0.057            0.042          0.117          0.196         0.050           0.158         0.102
                      IF seed 1     0.039            0.071            0.050          0.133          0.200         0.062           0.168         0.113
                      IF seed 2     0.039            0.021            0.004          0.004          0.176         0.013           0.093         0.052
                      IF seed 3     0.039            0.032            0.017          0.079          0.180         0.025           0.131         0.077
                      IF seed 4     0.039            0.064            0.029          0.104          0.471         0.048           0.293         0.167
R0 exact route (no calibration)       NaN              NaN              NaN            NaN            NaN           NaN             NaN           NaN
```

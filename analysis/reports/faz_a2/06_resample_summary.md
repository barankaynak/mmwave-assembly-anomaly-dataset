# 06 Resample summary (descriptive; 2026-11-10)

Source: 04_comm_anomaly_far.csv / 04_comm_anomaly_groups.csv, B2 x pred, n = 20 training cycles, 50 draws.

Predicted calibration cycles per draw: min 16, median 20, max 23; draws with n_cal <= 18: 13 of 50.

D3 can alarm at alpha = 0.10 only if n_cal >= 19 (smallest D3 = 2/(n_cal+1)).

```
                   group   subset  draws  mean    sd   q10   q50   q90  zero   min   max
                    fast      all     50 0.577 0.413 0.000 0.760 0.960    13 0.000 0.979
                    fast ncal>=19     37 0.780 0.265 0.333 0.958 0.979     0 0.208 0.979
                    fast ncal<=18     13 0.000 0.000 0.000 0.000 0.000    13 0.000 0.000
            seq(135+145)      all     50 0.278 0.378 0.000 0.082 1.000    13 0.000 1.000
            seq(135+145) ncal>=19     37 0.375 0.396 0.019 0.096 1.000     0 0.010 1.000
            seq(135+145) ncal<=18     13 0.000 0.000 0.000 0.000 0.000    13 0.000 0.000
                  seq135      all     50 0.209 0.376 0.000 0.036 1.000    14 0.000 1.000
                  seq135 ncal>=19     37 0.283 0.413 0.018 0.071 1.000     1 0.000 1.000
                  seq135 ncal<=18     13 0.000 0.000 0.000 0.000 0.000    13 0.000 0.000
                  seq145      all     50 0.357 0.448 0.000 0.125 1.000    13 0.000 1.000
                  seq145 ncal>=19     37 0.483 0.459 0.021 0.125 1.000     0 0.021 1.000
                  seq145 ncal<=18     13 0.000 0.000 0.000 0.000 0.000    13 0.000 0.000
                    slow      all     50 0.633 0.409 0.000 0.804 0.980    13 0.000 1.000
                    slow ncal>=19     37 0.855 0.180 0.600 0.961 0.988     0 0.314 1.000
                    slow ncal<=18     13 0.000 0.000 0.000 0.000 0.000    13 0.000 0.000
        tempo(fast+slow)      all     50 0.606 0.396 0.000 0.773 0.971    13 0.000 0.990
        tempo(fast+slow) ncal>=19     37 0.819 0.186 0.596 0.869 0.980     0 0.263 0.990
        tempo(fast+slow) ncal<=18     13 0.000 0.000 0.000 0.000 0.000    13 0.000 0.000
             tum anomali      all     50 0.438 0.345 0.000 0.441 0.976    13 0.000 0.995
             tum anomali ncal>=19     37 0.591 0.262 0.303 0.517 0.985     0 0.133 0.995
             tum anomali ncal<=18     13 0.000 0.000 0.000 0.000 0.000    13 0.000 0.000
FAR (normal test cycles)      all     50 0.049 0.078 0.000 0.000 0.155    32 0.000 0.300
FAR (normal test cycles) ncal>=19     37 0.066 0.085 0.000 0.000 0.180    19 0.000 0.300
FAR (normal test cycles) ncal<=18     13 0.000 0.000 0.000 0.000 0.000    13 0.000 0.000
```

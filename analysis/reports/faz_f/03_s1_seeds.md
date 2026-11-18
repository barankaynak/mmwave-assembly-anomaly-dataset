# Faz F / 03 CNN-GRU under S1, five seeds per fold (secondary; 2026-11-10)

Cells (fold x seed): 25; replicas per fold: {0: 5, 1: 5, 2: 5, 3: 5, 4: 5}

Same-seed rerun of fold 0 (replica 0): macro-F1 0.8088 vs primary 0.8275

CNN-GRU S1 macro-F1, mean over all cells 0.8326 (min 0.7969, max 0.8583); primary fold mean 0.8408; B2 fold mean 0.8564

B2 leads in 25 of 25 fold x seed cells

B2 minus CNN-GRU (five-seed fold means), paired by fold: mean +0.0239, 95% t interval [+0.0122, +0.0355] (5 folds)

Within-fold seed s.d. of CNN-GRU macro-F1: mean 0.0090 (per fold {0: 0.0129, 1: 0.0051, 2: 0.0061, 3: 0.013, 4: 0.008})

```
      n  cnn_mean  cnn_min  cnn_max     b2  gap_mean
fold                                                
0     5    0.8113   0.7969   0.8275 0.8455    0.0342
1     5    0.8513   0.8449   0.8583 0.8701    0.0189
2     5    0.8477   0.8391   0.8533 0.8633    0.0157
3     5    0.8276   0.8066   0.8396 0.8615    0.0338
4     5    0.8251   0.8143   0.8355 0.8418    0.0166
```

Leakage (secondary): S0 five seeds 90.97 % vs S1 five-seed fold means 86.10 % accuracy; difference +4.87 points, Welch 95% interval [+3.04, +6.70] (df 5.5, p 0.00078)


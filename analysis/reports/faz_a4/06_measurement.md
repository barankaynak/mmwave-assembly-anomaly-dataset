# 06 Measurement characterisation (descriptive; 2026-11-10)

## (a) Radar configuration
Source: iwr6843isk_radar_profile.cfg

- start frequency 60.0 GHz; slope 70.0 MHz/us; ramp end 57.14 us (ramp bandwidth 4.00 GHz)
- ADC: 112 samples at 2279.0 ksps = 49.14 us; bandwidth used 3.44 GHz; range resolution c/2B = 4.36 cm
- header: range resolution 0.044 m; max range 3.95 m; max radial velocity 1 m/s; velocity resolution 0.13 m/s; azimuth resolution 15 deg; frame 100 ms
- antennas: RX mask 15 (4 RX), TX mask 5 (2 TX) -> 8 virtual channels, azimuth only (no elevation TX)
- chirps per frame: 16 loops x 2 chirps; frame period 100.0 ms
- CFAR range: -1 0 2 8 4 3 0 15 0; CFAR Doppler: -1 1 0 4 2 3 1 15 0 (procDir mode win guard shift cyclic thr_dB peakgroup)
- CFAR field of view: range 0 0 0.5 m; Doppler 1 -1 1.00 m/s
- clutterRemoval -1 1 (static zero-Doppler removal on); multiObjBeamForming -1 1 0.5

## (b) Step-duration agreement, B2 minus ToF (s), ToF cycle boundaries, equal visit sequences only

Cycles with equal debounced sequences: 87 of 254 (0.343); natural-deviation cycles left out.

Per session (equal / all): session_02_normal_seq125: 19/50; session_03_normal_seq135: 21/57; session_04_normal_seq145: 27/48; session_05_anomaly_fast: 12/47; session_06_anomaly_slow: 8/52

All steps: n 421, bias +0.002, sd 0.128, LoA [-0.249, +0.254], median |error| 0.099, share within 0.1 s 0.644

```
       step      n  tof_median   bias    sd  loa_lo  loa_hi  mae_median  within_0p1
    dwell_1 87.000       0.700  0.000 0.111  -0.217   0.217       0.001       0.667
    dwell_2 39.000       0.601  0.043 0.133  -0.217   0.303       0.098       0.667
    dwell_3 21.000       0.600  0.052 0.107  -0.159   0.262       0.099       0.762
    dwell_4 27.000       0.600  0.051 0.116  -0.175   0.278       0.098       0.704
    dwell_5 80.000       0.500  0.051 0.115  -0.175   0.278       0.100       0.525
transit_1_2 39.000       0.502 -0.000 0.168  -0.330   0.330       0.098       0.667
transit_1_3 21.000       0.500 -0.019 0.103  -0.221   0.183       0.100       0.619
transit_1_4 27.000       0.499 -0.019 0.127  -0.268   0.231       0.099       0.704
transit_2_5 32.000       1.950 -0.109 0.142  -0.386   0.169       0.100       0.500
transit_3_5 21.000       1.600 -0.076 0.070  -0.212   0.061       0.099       0.762
transit_4_5 27.000       1.601 -0.048 0.075  -0.195   0.100       0.001       0.741
```

```
                 session       n  tof_median   bias    sd  loa_lo  loa_hi  mae_median  within_0p1
session_02_normal_seq125  95.000       0.601  0.011 0.119  -0.222   0.243       0.098       0.705
session_03_normal_seq135 105.000       0.600 -0.001 0.097  -0.192   0.190       0.099       0.752
session_04_normal_seq145 135.000       0.600 -0.000 0.108  -0.211   0.211       0.098       0.659
 session_05_anomaly_fast  46.000       0.500 -0.015 0.178  -0.365   0.335       0.100       0.565
 session_06_anomaly_slow  40.000       1.002  0.020 0.202  -0.375   0.415       0.198       0.250
```

Visit onset error (B2 - ToF): n 254, median +0.000 s, IQR [+0.000, +0.000], share |error| <= 0.1 s 0.827
Visit offset error (B2 - ToF): median +0.000 s, IQR [+0.000, +0.100], share |error| <= 0.1 s 0.744

## (b1) Visit-level agreement, B2 minus ToF (s), matched by overlap (main measure)

ToF visits: 715; matched 1:1 711 (0.994), missed 0 (0.000), split 4 (0.006)

```
status                    matched  split
session                                 
session_02_normal_seq125      151      2
session_03_normal_seq135      161      1
session_04_normal_seq145      143      0
session_05_anomaly_fast       101      0
session_06_anomaly_slow       155      1
```

```
   kind       n  tof_median   bias    sd  loa_lo  loa_hi  mae_median  within_0p1
  dwell 711.000       0.601  0.067 0.153  -0.234   0.368       0.100       0.550
transit 653.000       0.699 -0.074 0.149  -0.366   0.219       0.100       0.579
```

```
       step       n  tof_median   bias    sd  loa_lo  loa_hi  mae_median  within_0p1
    dwell_1 256.000       0.701  0.049 0.151  -0.246   0.344       0.100       0.559
    dwell_2 145.000       0.700  0.079 0.166  -0.246   0.404       0.100       0.559
    dwell_3  56.000       0.601  0.074 0.099  -0.120   0.269       0.099       0.661
    dwell_4  49.000       0.601  0.050 0.115  -0.176   0.277       0.099       0.673
    dwell_5 205.000       0.500  0.083 0.166  -0.243   0.409       0.100       0.473
transit_1_2 142.000       0.600 -0.040 0.182  -0.397   0.317       0.100       0.563
transit_1_3  56.000       0.500 -0.045 0.109  -0.259   0.169       0.099       0.661
transit_1_4  49.000       0.499 -0.006 0.144  -0.288   0.275       0.099       0.714
transit_2_1  10.000       2.449  0.001 0.066  -0.128   0.129       0.000       0.900
transit_2_5 106.000       2.300 -0.156 0.160  -0.469   0.157       0.107       0.443
transit_3_5  47.000       1.600 -0.091 0.088  -0.263   0.081       0.099       0.660
transit_4_5  44.000       1.700 -0.054 0.087  -0.225   0.117       0.098       0.727
transit_5_1 199.000       0.500 -0.082 0.134  -0.344   0.180       0.100       0.538
```

```
                 session    kind       n  tof_median   bias    sd  loa_lo  loa_hi  mae_median  within_0p1
session_02_normal_seq125   dwell 151.000       0.600  0.062 0.146  -0.225   0.348       0.100       0.636
session_02_normal_seq125 transit 146.000       0.600 -0.073 0.093  -0.256   0.110       0.100       0.603
session_03_normal_seq135   dwell 161.000       0.600  0.062 0.103  -0.141   0.265       0.100       0.671
session_03_normal_seq135 transit 149.000       0.501 -0.064 0.102  -0.265   0.137       0.099       0.664
session_04_normal_seq145   dwell 143.000       0.600  0.026 0.126  -0.220   0.273       0.099       0.580
session_04_normal_seq145 transit 137.000       0.500 -0.025 0.117  -0.254   0.204       0.098       0.686
 session_05_anomaly_fast   dwell 101.000       0.501 -0.059 0.133  -0.321   0.202       0.099       0.584
 session_05_anomaly_fast transit  68.000       0.400  0.084 0.163  -0.236   0.404       0.099       0.662
 session_06_anomaly_slow   dwell 155.000       0.801  0.198 0.145  -0.087   0.483       0.199       0.290
 session_06_anomaly_slow transit 153.000       1.500 -0.197 0.156  -0.503   0.108       0.198       0.340
```

Onset error of matched visits: median +0.000 s, IQR [-0.100, +0.000], share |e| <= 0.1 s 0.757
Offset error of matched visits: median +0.000 s, IQR [+0.000, +0.100], share |e| <= 0.1 s 0.722

## (c) Alarm latency (B2 x pred, alpha = 0.10, alarmed anomaly pairs)

- fast: n 221; seconds from cycle start median 3.70 [IQR 3.40-4.10]; share after cycle end 0.801; seconds after end (those) median 0.10
- seq135: n 71; seconds from cycle start median 1.30 [IQR 1.20-1.65]; share after cycle end 0.085; seconds after end (those) median 0.10; seconds after wrong-box entry median 0.10
- seq145: n 114; seconds from cycle start median 1.30 [IQR 1.10-1.60]; share after cycle end 0.105; seconds after end (those) median 0.10; seconds after wrong-box entry median 0.10
- slow: n 237; seconds from cycle start median 2.50 [IQR 2.20-2.80]; share after cycle end 0.055; seconds after end (those) median 0.10

## (d) Timing resolution: D3 on uniformly time-scaled normal cycles (from reports/faz_e/01_tempo.csv)

```
variant  n  det_D3_phaseA  det_D2_p2_05  det_D1_05
  x0.60 50           0.92          0.92       0.00
  x0.70 50           0.58          0.58       0.00
  x0.80 50           0.12          0.12       0.00
  x1.25 50           0.24          0.22       0.02
  x1.40 50           0.50          0.50       0.00
  x1.70 50           0.96          0.96       0.02
```


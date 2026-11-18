# Faz F — Cycle-level baselines (critique M6), CPU part

**Written:** 2026-11-10, before any F score was computed. Later changes go to "Sapmalar" (dated).
**Status:** descriptive. Planned after the primary, all secondary results and E1 were known. Does not change the primary detector (B2 + D3) or any primary number. Label in the paper: descriptive.

## Purpose
Give the reader standard reference detectors on exactly the same cycles, folds and calibration, so the two-channel detector is compared with something other than the authors' own models.

## Common protocol (unchanged from the primary analysis)
- Cycles: B2 states (S1 out-of-fold), predicted cycle boundaries, `faz_a2/a2_cycles.build_cycles('b2','pred', ...)`; debounced visit sequence `seq` (the sequence D1 reads).
- Roles: `fa_common.anom_roles(f)`, f = 0..4 (train = folds f+3, f+4; calibration = f+1, f+2; test = f; all s03-s06 cycles tested in every fold).
- Conformal p = (1 + #{cal >= s}) / (n_cal + 1); alarm p <= alpha; alpha = 0.10 primary, 0.05 shown.
- Reporting: pooled false-alarm rate over the 51 normal test cycles (no per-cycle breakdown of s02), detection per type over (cycle, fold) pairs, fold-mean AUROC (ties 0.5). No intervals for rates that follow from published readings.

## Baselines (all fixed here)
1. **Exact-route rule (R0).** Alarm if the debounced visit sequence is not exactly 1-2-5. No calibration, no parameter.
2. **Levenshtein distance to the route (LEV).** s_L = edit distance (unit costs) between `seq` and (1,2,5). (a) LEV alone with its conformal p; (b) LEV + D2 with the Bonferroni rule of D3: p = min(1, 2 min(p_L, p_2)).
3. **Isolation forest (IF).** scikit-learn `IsolationForest` with default settings (100 trees, max_samples 'auto', contamination ignored), fitted on the fold's training normals; score s = -score_samples(x) (higher = more anomalous); conformal p from the calibration normals; reported alone. Seeds 0, 1, 2, 3, 4 (random_state); mean and range reported.
   Features (no imputation needed for off-route cycles): number of visits to each box 1..5 (5), number of visits, cycle duration, summed dwell, summed transit (9 features).
4. HSMM likelihood: not run (would need choices fixed after the fact); named in the text as not tested.

Comparison rows: single-score baselines (LEV alone, IF) are compared with D0, D1, D2 alone (alarm floor 1/(n+1)); combined baselines (LEV + D2) with D3 and E1 + D2 (floor 2/(n+1)). R0 has no calibration.

## Predictions written before the run (descriptive; do not enter any decision)
- P1. R0 equals the visit-sequence table: false alarms = normal test cycles not read 1-2-5 (4 of 51 in the published table, 47/51 read 1-2-5); sequence detection close to 100 %; fast detection about 27 % (cycles read 1-2); slow close to 0 %.
- P2. LEV ties: every fold's calibration set holds at least one cycle at distance 1 (1-2-2-5 or 1-3-2-5, Supplementary table e1diag), and 1-3-5 / 1-4-5 are at distance 1. So LEV + D2 cannot alarm through LEV in any fold (needs p_L <= 0.05); its sequence detection equals the D2 floor (about 5-13 %). LEV alone alarms on distance-1 cycles only in folds with exactly one distance-1 calibration cycle and n_cal >= 19 (folds 1 and 4 by the table); elsewhere only cycles at distance >= 2.
- P3. IF: sequence deviations change only one visit identity, so the count features move little; IF should detect tempo far better than sequence.

## Not part of this plan (GPU; separate plan if the author agrees)
CNN-GRU PyTorch port under S1 with more seeds; a published point-cloud network under S1.

## Outputs
`faz_f/01_cycle_baselines.py` -> `reports/faz_f/01_cycle_baselines.md`, `01_cycle_baselines.csv`.

## Part F.2 - CNN-GRU port under S1 with more seeds (written 2026-11-10, before any F.2 run; the author freed the RTX 3090)
- Script `faz_f/02_cnn_s1_seeds.py`: imports `faz_a4/13_cnn_s1_local.py` / `faz_a/01_state_cnn.py` unchanged; only the seed changes. Replica r: seed = SEED_S1 + 1000 + fold + 100 r, used for the group-wise inner validation split and for torch/numpy (the S0 seeds in faz_d varied the same two things; the S1 test folds stay fixed). RTX 3090 only, no CPU fallback.
- Gate: replica 0, fold 0 runs first and must reproduce the primary fold-0 macro-F1 0.82747 within 0.01 (cuDNN is not bit-exact). If it fails, nothing further runs and the failure is reported.
- Then replicas 1-4 on all five folds (20 runs).
- Reading rule (fixed now): the primary S1 numbers (B2 0.856, CNN-GRU 0.841) are not replaced. Reported in addition (secondary): CNN-GRU S1 macro-F1 for five seeds per fold (primary + replicas 1-4), the mean over seeds and folds, and the B2-minus-CNN-GRU difference paired by fold (B2 primary per fold), with the number of fold x seed cells in which B2 leads. The leakage headline stays the faz_d estimate; an S0-minus-S1 estimate with the five-seed S1 mean is reported beside it as secondary.
- Prediction (descriptive): B2 leads in most fold x seed cells; the mean gap stays between 0.005 and 0.03 macro-F1.
- A published point-cloud network is not part of F.2; it needs its own plan.

## Sapmalar
(none yet)
- 2026-11-10 (after the run, results only, plan unchanged): `reports/faz_f/01_cycle_baselines.md`. Reproduction check passed. P1 held (R0: FAR 4/51, sequence 100 %, fast 27.1 %, slow 2.0 %). P2 held (LEV + D2 sequence 10.4 %; LEV alone 100 % in folds 1 and 4 only). P3 held (IF AUROC sequence 0.64, tempo 0.87; detection all 36.5 % mean over seeds, against 63.3 % for D3).
- 2026-11-10 (F.2 gate): replica 0, fold 0 on the RTX 3090 gave macro-F1 0.8088 (acc 0.8427; 76 epochs, best 51, val loss 0.3739; same train/val/test sizes 11749/1199/3300) against the primary 0.82747 (74 epochs, best 49, val loss 0.3612). Difference 0.0187 > 0.01: **gate failed**. Per plan, replicas 1-4 were not started. Log: reports/_logs/job-20261110-163208-1462-21120.log. Decision on how to continue is the author's.
- 2026-11-10 AMENDMENT F.2-gate (author's go-ahead "Evet", written before any replica 1-4 run): the gate was meant to catch a code or environment error. Environment, splits and sizes were identical, so the 0.019 gap is attributed to GPU nondeterminism (no deterministic flags in the training code). The gate is dropped; the reproduction run (replica 0, fold 0, 0.8088) is kept and reported as evidence of same-seed run-to-run variation. Replicas 1-4 run on all five folds as planned. Reading rule and prediction unchanged. In the paper, all F.2 numbers are secondary.
- 2026-11-10 (F.2 results, plan unchanged): 20 runs on the RTX 3090; `reports/faz_f/03_s1_seeds.md`. B2 leads in 25 of 25 fold x seed cells; five-seed gap +0.0239 macro-F1 (95 % t interval +0.0122 to +0.0355, 5 folds); CNN-GRU mean 0.8326 (0.797-0.858), within-fold seed s.d. 0.009. Prediction held (gap within 0.005-0.03). Leakage with five-seed S1: +4.87 points (3.04-6.70); headline stays +4.04.

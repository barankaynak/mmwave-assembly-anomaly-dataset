# Addendum to `02_oneshot.md` (2026-11-06, written after the run; no result changed)

`02_oneshot.md` and its CSV files are the once-only output and are left unedited. Two independent checks reproduced every row, table and interval; their scripts, logs and the audit report are in `faz_e/verify/oneshot/`. They add the following points.

1. **Criterion (c).** It holds through the dated amendment A3' (DESIGN_E Sapmalar). A3 failed as written: one of 46 parents scored 1.793 against a limit of 1.2. The script's "success" line should be read with this in mind.
2. **Fast gain.**
   - The rise in fast detection from 0.921 to 0.963 is ten (cycle, fold) pairs in fold 2.
   - They come from 13 cycles that B2 read as `1-2` (12) or `1-2-4` (1). In each of them the ToF record shows a Box-5 touch of 1–3 frames, so these are not real skips. B2's longest Box-5 run in these cycles is at most one frame, and the debouncer removed it.
   - These alarms are a sequence-channel response, not better tempo detection.
   - The duration channel alone flags 185/240 fast and 237/255 slow pairs under both rules.
   - No Phase A alarm is lost in any session.
3. **Exposure.** The B2 state estimator (pooled S1) was trained on ToF-labelled frames of other s03–s06 cycles. "All 104 sequence-deviation cycles" holds for this estimator; the reading-level result was not computed with an estimator that never saw these sessions.
4. **Box-2 count.**
   - Boxes 3 and 4 are never on the route, so their visits are always charged in full. Every s03/s04 cycle therefore scores at least its foreign evidence (≥ 3.36), and a miss would need every Box-3/4 visit to vanish.
   - The one `1-3-5-2` cycle has a trailing 2-frame Box-2 reading during the return to Box 1. It scored 14.0–14.7.
5. **Fold 0.** n_cal = 19, so the smallest attainable p3' = 2/20 = 0.10 = alpha exactly. Alarms in that fold rest on "≤", as for Phase A D3.
6. **Bootstrap.** The bootstrap lower bounds depend on the order of clusters by up to 0.004 (Monte Carlo variation; `faz_e/verify/oneshot/my_intervals_numeric_order.csv`). Quote them to two decimals.

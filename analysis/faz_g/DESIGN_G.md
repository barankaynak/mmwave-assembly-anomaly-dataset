# Phase G — E1 with a state estimator that never saw the deviation sessions

**Requested:** 2026-11-18. The corresponding author chose to run E1 with estimator (ii) once and to report the result whatever it shows.
**Written:** 2026-11-18, before any Phase G score was computed. Later changes go, dated, into "Sapmalar" at the end.
**Status:** secondary analysis, planned after all primary and secondary results were known. Tag [S]. Do not use the word "pre-registered".

## 1. Question
E1 + D2 flagged all 104 sequence-deviation cycles in its single run (Phase E). That run used the pooled B2 estimator, which had been trained on other cycles of the deviation sessions s03–s06. Does the result hold when the state estimator has never seen s03–s06?

## 2. What is known before the run
1. Everything in Phase E (plan `faz_e/DESIGN_E.md`, result `reports/faz_e/02_oneshot.md`).
2. Estimator (ii) (B2 trained on s01 + s02 only; `faz_a4/04_unseen_sessions.py`): its D3 results (Table 6), its frame metrics, its predicted-cycle counts and its per-fold calibration counts (`reports/faz_a4/07_ncal_counts.md`).
   - D3, alpha = 0.10: FAR 3/50; seq135 0.264, seq145 0.480, fast 0.929, slow 0.973; AUROC 0.938.
   - Scored cycles: s02 50, s03 56, s04 45, s05 48, s06 51. n_cal per fold: 20, 21, 20, 19, 20.
3. **Not known:** any E1 score under estimator (ii), and the (ii) visit readings of s03–s06. They are not opened before the run.

## 3. Inputs and frozen parts
- State labels and class probabilities: `reports/faz_a4/oof/b2_s0102__all.csv.gz` (existing; no retraining).
- E1 rule: `faz_e/e1_core.py`, unchanged; its sha256 is checked against the Phase E battery record. No parameter is set (E1 has none).
- Folds, roles, debouncing, cycle cutting, D2 and conformal p-values: the Phase A protocol (`faz_a2/a2_cycles.py`), unchanged.
- Route and mu are fitted per fold on the s02 training normals, as in Phase E, now from estimator (ii)'s probabilities.

## 4. Procedure (script `faz_g/01_e1_estimator_ii.py`)
1. Refuses to run without `E1_II_GO=1` and writes a start sentinel before loading data; a second run is refused.
2. **Gate:** rebuild the estimator (ii) cycles and D3 and reproduce the Table 6 row (FAR and the four detection rates, 3 decimals, from `reports/faz_a4/04_unseen_anomaly_table.csv` and `04_unseen_far.csv`). If the gate fails, stop before any E1 score is computed.
3. Compute S_E, p_E and D3' = min(1, 2 min(p_E, p_2)) per (cycle, fold), as in Phase E.
4. s02 cycles marked natural_dev stay in the pool and are dropped before tabulation; only their count is reported, never their status.
5. Write all rates, the per-fold table, AUROC of S_E and the collapsed readings to `reports/faz_g/`.

## 5. Quantities reported (all of them, in every case)
At alpha = 0.10: FAR on s02 (Wilson 95% interval); detection for seq135, seq145, fast, slow and all (cycle-clustered bootstrap intervals as in the paper); p_E alone; alarm path per channel; AUROC of S_E (sequence vs normal); D3 of estimator (ii) beside it.

## 6. Reading rule (written before the run)
- **(A)** seq135 and seq145 both >= 0.95 under D3', and FAR no more than one alarm above D3 of estimator (ii) (<= 4/50):
  "With an estimator that never saw the deviation sessions, E1 + D2 detected X% and Y% of sequence deviations at a false-alarm rate of Z%. The result does not depend on the estimator having seen these sessions."
- **(B)** otherwise: report the numbers and write: "With an estimator that never saw the deviation sessions, E1 + D2 detected X% and Y% of sequence deviations at a false-alarm rate of Z%. The full detection of the pooled run therefore depends partly on the estimator having seen these sessions."
- In both cases: one new row in Table 6, one paragraph in section 6.4, and the abstract and contribution caveats are updated to match. The 104/104 result stays the Phase E result and is not replaced.
- Tempo rates are reported as they come; a change there is described, not explained away.

## 7. What is not allowed
- No second run, no change to E1, D2, debouncing, alpha or the roles after the run.
- If the code fails for a technical reason (crash, gate), it may be fixed without looking at any s03–s06 output, and the fix is logged in "Sapmalar".
- Estimator (i) is not run: it saw the other deviation sessions.

## Sapmalar

- 2026-11-18, attempt 1: the script stopped at the gate, before any E1 score was computed and before any output was written (`print(NCAL.to_string())` on a list; traceback in `reports/faz_g/01_stdout.log`). Technical fix only: the line now wraps NCAL in a DataFrame. No other change. The first start sentinel is kept as `reports/faz_g/01_STARTED_attempt1_crashed.sentinel`; the run was then started again.

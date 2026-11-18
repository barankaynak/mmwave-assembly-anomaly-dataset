# Phase E — An evidence-weighted sequence channel (E1)

**Requested:** 2026-11-05, about 22:30 TR. The corresponding author said: "we must also catch sequence errors; ask Fable".
**Written:** 2026-11-06, about 00:35 TR. At that time no Phase E script existed in this folder, and **no Phase E score had been computed on s03–s06**. Later changes go, dated, into "Sapmalar" at the end.
**Status:** secondary analysis, planned after the Phase A–D results were known. The Phase A rule stays the planned primary result: D3, B2 × pred, alpha = 0.10, with FAR 0.078, seq135 0.254, seq145 0.475, fast 0.921 and slow 0.929; this is the "36 %" sequence figure. Do not use the word "pre-registered". Tag: [S].
**Origin:** brief `faz_e/00_BRIEF_seq.md`. Method search by the workflow `seq-channel-fable`, 2026-11-05/06:
- four proposers (Fable), each with an adversarial critic, and a Fable synthesis;
- the synthesis draft is kept verbatim in `faz_e/prelim/DESIGN_E_draft_workflow.md`;
- the cited preliminary logs are in `faz_e/prelim/`.

This file is the draft as edited by the analyst. The edits are:
- the date was corrected;
- the pooled layer P2 was removed from the one-shot (see "Not run");
- scratch paths were replaced by `faz_e/prelim/`;
- the implementation and guard section was added.

## 1. Status and disclosure (to be carried into the report and the manuscript)
When this plan was written, the following were already known:
1. The full Phase A and A2 result tables for all sessions (brief §3).
2. The per-fold diagnosis in brief §4. It used the D1 values that the s03/s04 cycles receive in each fold. It also identified the artefact normal cycles `1-2-2-5` ×3 and `1-3-2-5` ×1, which tie with every sequence anomaly.
3. Two quantities, computed on s03/s04 during that diagnosis and before any plan existed (brief §4):
   - the longest raw B2 run at the substituted box: median 7 frames, minimum 4, 10th percentile 5;
   - the summed p3 + p4 posterior mass: seq135 median 6.83 (minimum 4.05), seq145 median 6.68 (minimum 4.55), normals median 0.91 (90th percentile 1.58, maximum 2.30).

   **E1 is built on posterior mass, i.e. on this family of quantities, so it is not blind.** No E1 parameter is set from these numbers, because E1 has no free parameter. The term that decides s03/s04 is the deletion cost mu_2, a constant estimated from normal cycles.
4. The Phase A B2 × pred visit-reading table for every session (`reports/faz_a/04_anomaly.md`):
   - s03: `1-3-5` ×54, plus one each of `1-3-5-2` and `1-3-3-5`;
   - s04: `1-4-5` ×47, plus one `1-4-5-5`;
   - s05: `1-2-5` ×35, `1-2` ×11, `1-2-4` ×1, and one reading not in the table;
   - s06: `1-2-5` ×50, plus one `1-2-2-5`.

   E1's alarms on s03/s04 follow arithmetically from these readings (see §6). **The one-shot run on s03/s04 therefore checks that the decision layer is consistent; it does not test the signal.**
5. The probe geometry table `reports/probe/q1.md` (s03 Box-3 dwell at x ≈ −0.006 ± 0.021 m, among others) was read by the geometry proposer. It plays no role in E1.

**Forking paths.** Before E1 was chosen, nine channels and calibration layers were explored in the workflow on s01, s02-normal and synthetic data only:
- E: Viterbi log-likelihood ratio, route versus free, over a visit grammar;
- F: forward profile-HMM Bayes factor;
- P1: pooled union rule;
- P2: pooled (2, 3) slot rule;
- D4: dwell-position geometry, and its posterior analogue;
- DTW over the full trajectory;
- DTW over dwells only;
- E2: frame-level route-HMM forced alignment.

None was run on s03–s06. An audit of every workflow script (2026-11-06) found loaders filtered to `session_01_training` / `session_02_normal_seq125`, with assertions in most of them. No workflow agent wrote into the project folders.

**One change to E1 after its proposal.** The critic found that the original collapse rule charges a 2-frame flicker inside a dwell with the flicker mass plus the smaller half of the split dwell. On 46 synthetic parents the score was then 2.29–8.70 (`prelim/rev_S1.log`, CHECK D). The multi-match alignment of §2.4 fixes this:
- the mid-dwell flicker scores its own mass, 1.09;
- every real normal score, the FAR and the omission/swap/repeat minima are unchanged.

The multi-match rule is frozen here. The plain variant is never run on s03–s06.

## 2. Method

### 2.1 Inputs (all existing; no retraining; CPU)
- Per-frame out-of-fold outputs of B2 (S1): `reports/faz_a/oof/b2__S1__f{0..4}.csv.gz`, columns `y_pred` and `p0..p5`.
  - The S1 folds are cycle-grouped over all six sessions.
  - Each cycle's posteriors are out-of-fold for that cycle.
  - The B2 models have seen ToF-labelled frames of s03–s06 from other folds. This exposure is inherited from Phase A and already declared there.
- Cycles, fold roles and the D2 reference come from `faz_a2/a2_cycles.py`, which reproduces Phase A exactly:
  - `labels_from_oof`, `build_cycles('b2', 'pred', ...)`, `fit_params`, `score_cycles`, `conf_p`, `d3`;
  - roles: test f; calibration f+1, f+2; training f+3, f+4 (mod 5);
  - `reports/faz_a/folds_anom.csv`.
- Phase A debouncer, unchanged: MAX_GAP = 1, MIN_BOX = 2. Cycle = predicted Box-1 entry to the next one.

### 2.2 Visits and evidence
- A visit is a maximal run of debounced label b > 0. Its evidence (mass) is E_v = Σ_{t in run} p_b(t), in frames.
- **Collapse:** consecutive visits of the same box that are separated only by none frames merge into one visit, and their masses add.

### 2.3 Route and reference masses (training normal cycles of the fold only)
- Route R = the modal collapsed visit sequence of the training normal cycles. Expected: `1-2-5` in every fold.
- mu_b for b in R = the mean, over training normal cycles, of the whole-cycle posterior mass of box b, transit frames included: mu_b = mean_c Σ_{t in c} p_b(t).
- Preliminary values (`prelim/e1_run1.log`), folds f0..f4:
  - mu_1: 8.37 / 8.30 / 8.00 / 8.03 / 8.56;
  - mu_2: 8.05 / 8.52 / 8.25 / 7.86 / 8.15;
  - mu_5: 5.64 / 5.55 / 5.48 / 5.74 / 5.57.

### 2.4 Score S_E: multi-match, evidence-weighted edit distance (frozen)
The observed visits v_1..v_n (box b_j, mass E_j) are aligned left to right to the route r_1..r_m.

States of the dynamic programme:
- A[i, j]: cheapest cost with route step i matched (open) after consuming visits 1..j.
- B[i, j]: cheapest cost with step i deleted (or i = 0). B[0, 0] = 0; every other cell starts at +inf.

Recurrences, for j ≥ 1 with E = E_j:
- **Insertion** (step i open or deleted): A[i, j] ← A[i, j−1] + E; B[i, j] ← B[i, j−1] + E.
- **Match or continuation**, if b_j = r_i and i ≥ 1: A[i, j] ← min(A[i−1, j−1], B[i−1, j−1], A[i, j−1]). The last term is the multi-match: an open step absorbs a later visit of its own box at zero cost.
- **Deletion** of step i (i ≥ 1): B[i, j] ← min(A[i−1, j], B[i−1, j]) + mu_{r_i}.

The score is S_E = min(A[m, n], B[m, n]) ≥ 0, in frames of posterior evidence.

Unit cases (the implementation must assert them; route 1-2-5; E_k is the mass of the visit in question):
| reading | S_E |
|---|---|
| `1-2-5` | 0 |
| `1-2-2-5` | 0 |
| `1-5` | mu_2 |
| `1-5-2` | mu_2 + E_2 |
| `1-2-3-2-5` | E_3 |
| `1-2-5-2` | E_2 (trailing) |
| `1-2` | mu_5 |
| `2-5` | mu_1 |
| `1-3-5` | mu_2 + E_3 |

How the score treats each kind of deviation:
- substitution: mu of the skipped box plus the foreign mass;
- omission: mu of the skipped box;
- swap: one deletion plus one insertion;
- foreign or extra station: its mass;
- far repeat: the mass of the repeated visit;
- truncation: mu_5;
- an immediate same-box repeat with no other box in between scores 0. **This channel declares it undetectable**: in the label domain it cannot be told apart from the split visits that occur in normal data.

### 2.5 Parameters (all fixed now; none free)
| item | value | origin |
|---|---|---|
| debouncer MIN_BOX / MAX_GAP | 2 / 1 frames | Phase A |
| route R | modal collapsed training sequence | fixed by the data |
| mu_b | mean whole-cycle posterior mass of box b, training normals | fixed by the data |
| alignment | multi-match DP of §2.4 | structural, frozen |
| alpha, Bonferroni factor, ties | 0.10, 2, conservative (cal ≥ s − 1e-12) | Phase A |
| fold roles | test f; cal f+1, f+2; train f+3, f+4 | Phase A |
| posterior floor / minimum dwell / scale | none | E1 has none |

### 2.6 Explanation of an alarm
The DP traceback lists:
- the deleted route steps, with their mu;
- the charged visits, with box, first and last frame, and mass.

Example: "S_E = 8.05: route step Box 2 missing (mu_2 = 8.05); reading 1-5".

## 3. Calibration and combination with D2 (one rule)
- p_E = (1 + #{calibration cycles with S_E ≥ s}) / (n_cal + 1), with the fold's calibration cycles (n_cal = 19–22) and ties counted conservatively.
- **D3' = min{1, 2 · min(p_E, p_2)}**, where p_2 is the unchanged Phase A D2 p-value. Alarm if D3' ≤ 0.10. E1 replaces p_1 and nothing else.
- Secondary column: p_E ≤ 0.10 on its own.
- **FAR argument.** Within s02 the normal cycles were dealt to folds by a random permutation. For a fixed fold, the training folds fix the score map, and the calibration and test normal cycles are exchangeable under it. The B2 posteriors are out-of-fold for every cycle. Hence P(p_E ≤ a) ≤ a, and by the union bound P(D3' ≤ 0.10) ≤ P(p_E ≤ 0.05) + P(p_2 ≤ 0.05) ≤ 0.10, whatever the dependence between S_E and D2.
- **What is approximate:**
  - the guarantee is marginal over calibration draws and holds within one session;
  - at n_cal ≈ 20 the FAR conditional on the calibration set can reach about 0.18 at the 90th percentile;
  - drift between interleaved folds is a second-order approximation;
  - the B2 models carry exposure to s03–s06.
- **The decision layer is degenerate, and this is stated openly.** 50 of the 51 normal S_E are exactly 0, and the one positive score is the 0.2 s flicker, 1.09. Preliminary per-fold calibration maxima are 1.09 / 0 / 0 / 0 / 1.09. In folds 1–3 the rule therefore reduces to "any collapsed off-route visit alarms".
- Preliminary normal-side result (`prelim/e1_check3.log`):
  - p_E ≤ 0.10 gives FAR 1/51;
  - D3' gives FAR 4/51 = 0.078 on **the same four cycles as Phase A D3** (three through p_2, one through the flicker).

## 4. Synthetic validation (normal cycles of s02 + s01 material; run BEFORE the one-shot, with the frozen rule)
**Material:**
- Parents: the 50 s02 normal cycles in their Phase A test role whose collapsed reading is `1-2-5`.
- Foreign runs: ToF-true Box-3 and Box-4 runs of s01, the commissioning session, with their own B2 OOF posteriors.

**Scoring rules:**
- Every synthetic cycle is scored against its parent fold's training and calibration sets, unmodified.
- Debouncing is re-run on the composite labels.
- **D2 and p_2 are recomputed for every synthetic cycle from its own debounced labels and times, never copied from the parent.**
- The plain E1 is tabulated next to the frozen multi-match score, to show the change.

Kinds and preliminary values (workflow logs):
1. **omission_2** (50): delete the frame block of the collapsed Box-2 visit. Expected S_E = mu_2 of the fold; detection 1.000.
2. **swap_1-5-2** (50): exchange the Box-2 and Box-5 blocks. Preliminary minimum 9.30.
3. **repeat_2, far** (50): the parent's Box-2 block inserted after Box 5. Preliminary minimum 3.06.
4. **repeat_2, immediate** (50; negative by declaration): `1-2-[transit]-2-5`. Expected S_E = 0.
5. **subst_3 / subst_4** (50 parents × every s01 ToF-true Box-3 run (43) / Box-4 run from complete loops (30)): the s01 run block replaces the Box-2 block. Report overall, and split by whether the composite contains a visit of the pasted box ("readable").
   - Preliminary RNG version: subst_3 0.656 overall, 327/327 readable; subst_4 500/500.
6. **foreign_3 / foreign_4** (50 × 43 / 50 × 30): the s01 run block inserted at the midpoint of the 2→5 transit. Preliminary foreign_4 minimum 1.95.
7. **truncation** (50): delete the Box-5 block, plus variants that keep 1, 2 or 3 raw Box-5 frames. Expected S_E = mu_5 when the debounced reading lacks Box 5.
8. **tempo scaling** (50 × {0.6, 0.7, 0.8, 1.25, 1.4, 1.7}): labels and posteriors resampled uniformly in time (nearest frame), with times recomputed at 0.1 s. Report:
   - S_E;
   - D3' next to Phase A D3 on the same scaled cycles, both with recomputed D2.
9. **Negative control, reach flicker** (50): the real 2-frame Box-3 block of the `1-3-2-5` cycle inserted at the midpoint of the 1→2 transit. Expected S_E = 1.09; alarm only in the all-zero folds.
10. **Negative control, mid-dwell flicker** (parents whose Box-2 dwell is long enough, ≥ 46): the same block inserted at the middle of the Box-2 dwell. Expected multi-match S_E = 1.09 in every parent, and plain 2.29–8.70.
11. **Negative control, split visit** (50 × gaps of 2, 3 and 5 none frames inserted mid Box-2 dwell): S_E = 0. Report the recomputed D2/p_2 alarm rate.
12. **D1 exhibit:** Phase A D1 (p_1 ≤ 0.05) on kinds 1–3, 5 and 6. Preliminary values (`prelim/e1_checks.log`):
    - subst_4 0.354 (folds 0/2/4 at 0.00), subst_3 0.126, omission 0.360, swap/repeat/foreign 0.800;
    - E1 1.000 on all of these except subst_3.

    This exhibit is the evidence that the fix addresses the mechanism; the s03/s04 number is not.

**Acceptance, stated before the battery is run** (if an item fails, the plan is amended on normal data only and re-dated before the one-shot):
- detection ≥ 0.95 at D3' for omission, swap, far repeat, subst_4 and foreign_4;
- subst_3 readable-conditional ≥ 0.95;
- mid-dwell flicker multi-match S_E ≤ 1.2 in every parent;
- the 51 real normal S_E are 50 × 0 and 1 × 1.09 (cycle 19), and the D3' normal alarm set equals the Phase A D3 set;
- the unit cases of §2.4 pass;
- the D1 exhibit reproduces the tie mechanism (D1 detection < 0.5 on subst_4 or omission).

## 5. One-shot run on s03–s06 (computed once, only after the battery report is written and dated, and only with the corresponding author's go-ahead)
Script `faz_e/02_e1_oneshot.py`. It runs only with `E1_ONESHOT_GO=1` set. It first asserts that the Phase A D0–D3 and p0–p3 columns reproduce `reports/faz_a/04_cycle_scores.csv.gz` (B2 × pred). It then computes, once per cycle:
1. S_E and its traceback; p_E under the Phase A roles (anomaly cycles in every fold); D3'.
2. Summary at alpha = 0.10 for:
   - Phase A D3 (reproduced);
   - E1 D3' (the Phase E row);
   - p_E alone;
   - the alarm path per channel (p_E only, p_2 only, both).
3. For each session:
   - FAR (s02);
   - detection for seq135, seq145, fast and slow;
   - AUROC of S_E (sequence and tempo against normal);
   - the number of s03/s04 cycles whose collapsed reading contains Box 2, the only way a sequence deviation can be missed.
4. Nothing is re-run, re-tuned or re-thresholded afterwards. No other score variant is run on s03–s06.
5. The s02 natural_dev cycles stay in the pool, and their alarm status is never looked up or reported. None of them is in the B2 × pred pool.

## 6. Predictions (written before the run)
| quantity | prediction | why |
|---|---|---|
| seq135, E1 D3' and p_E alone | 0.98–1.00 | no reading in the table has Box 2 after collapse → S_E ≥ mu_2 ≥ 7.86 > calibration max 1.09 → p_E = 1/(n_cal+1) ≤ 0.05; one odd cycle allowed |
| seq145, same | 0.98–1.00 | `1-4-5` ×47, `1-4-5-5` |
| S_E on sequence cycles | minimum ≥ 7.86; typical 12–16 | mu_2 + foreign mass |
| AUROC of S_E, sequence vs normal | 0.98–1.00 | normals ≤ 1.09 |
| fast, E1 D3' | 0.90–1.00 | D2 half 0.771 at p_2 ≤ 0.05; the 12/48 `1-2` / `1-2-4` readings score ≥ mu_5 ≥ 5.48 in every fold |
| slow, E1 D3' | 0.91–0.94 | the D2 half alone, 0.929; s06 readings collapse to the route |
| sequence-channel alarms on tempo sessions | fast ≈ 0.25, slow ≈ 0 | truncation alarms (`1-2`), reported per channel and not called substitutions |
| FAR, E1 D3' | 0.078 (4/51), same cycles as Phase A | normals are tested once |

## 7. Reading rule
- **s03/s04.** Given the reading table, ≥ 0.95 on both sessions is reported as "consistent with the mechanism, foreseeable from the Phase A readings; the decision layer no longer loses the signal". It is not reported as a predicted-and-confirmed test. A result < 0.95 is treated as a bug-class finding: the production path differs from the reading table. It is investigated on normal data only, and no changed rule is re-run on s03/s04.
- **Criteria that can genuinely fail:**
  - (a) tempo retention: fast ≥ 0.90 and slow ≥ 0.90 under E1 D3';
  - (b) FAR unchanged: D3' alarms the same 4/51 normal cycles;
  - (c) the battery's acceptance items.
- **Success** means (a)–(c) hold and seq135/seq145 ≥ 0.95. Then:
  - the Phase A 36 % stays the primary result;
  - the paper reports E1 as a secondary sequence channel, designed after the primary results were known on the posterior-mass quantity that had been looked at on the deviation sessions, validated on synthetic deviations built from normal cycles, and then run once;
  - the paper includes the disclosure list of §1, the D1-vs-E1 synthetic exhibit and the per-channel attribution on s05;
  - the abstract may give the E1 number only with the secondary label.
- **If (a) fails:** E1 is reported as a sequence-only improvement, with the tempo loss stated in numbers. There is no switch to another rule afterwards.
- **If (b) fails:** the FAR is reported with its SE, without re-tuning.
- **In every case** the full one-shot table is reported, including negative rows.

## 8. Risks
- **R1. Not blind** (§1). The D1-vs-E1 synthetic exhibit carries the evidential weight.
- **R2. Degeneracy:** 50/51 normal scores are 0, so in three folds any off-route visit alarms. The flicker is E1's one normal alarm and is not patched with a floor.
- **R3. The 1.09-vs-≥5.5 margin is a property of s02.** In s01, 80/321 debounced visits lack ToF support, with mass up to 7.63 (`prelim/rev_s01_spur.log`, an upper bound). A less tidy operator would weaken detection of weak insertions and far repeats. Full wrong-box dwells and omissions would not be affected.
- **R4. State-estimator bound:** Box-3 substitution reaches 0.656 on s01 material (readable: 1.000). The boxes are about one azimuth resolution cell apart. The s03 order recovery came from a B2 that had seen other s03 cycles. A leave-one-session-out check is future work, not part of the one-shot.
- **R5. Tempo-to-sequence mixing:** fast `1-2` readings alarm as truncations. This is reported per channel.
- **R6. Immediate same-box repeats are undetectable** by E1 and by D2.
- **R7.** The validity is marginal and within one session (§3).
- **R8. Dependence** on the Phase A debouncer and on pred segmentation.
- **R9.** S_E in frames overstates independent evidence, because consecutive window posteriors overlap. This does not matter for the rank-based p-value.

## 9. Not run (and why)
- **P2, the pooled (2, 3) slot rule (removed by the analyst from the workflow draft).**
  - Its slot allocation was shaped by the diagnosis.
  - Its jackknife+ bound is 0.20.
  - Its purpose is a tempo gain the authors did not ask for.
  - One rule means fewer forking paths.
- **E** (Viterbi LLR):
  - three structural parameters with one-frame margins;
  - weak on truncated Box-5 touches.
- **F** (forward Bayes factor):
  - confounded with cycle length;
  - dominated by E.
- **P1** (pooled union): decision-identical to P2 here, and fragile to one extra artefact.
- **D4** (dwell geometry):
  - blind to repeats;
  - split visits 0.340;
  - a flicker-immunity cliff;
  - FAR 0.098 with no headroom.
- **DTW, full trajectory:** warping absorbs omissions (0.043).
- **DTW, dwells only:** sensitive to flickers (0.298).
- **E2** (frame-level forced alignment):
  - an undeclared posterior floor dominates the score;
  - omission power falls under LOSO (0.74);
  - its choices would be fixed post hoc.
- **Cross-conformal** (n_cal ≈ 40): already exists in Phase A2 and still holds 0–1 flickers.

## 10. Implementation and guards
- **`faz_e/e1_core.py`:**
  - visits with mass, collapse, route and mu fit, multi-match DP with traceback;
  - unit tests of §2.4 run on import (`assert`);
  - written fresh (not copied from the workflow scratch code).
- **`faz_e/01_e1_synthetic.py`:**
  - loads the OOF and `folds_anom.csv`, keeps only `session_01_training` and `session_02_normal_seq125`, and asserts that no other session is in memory;
  - prints every table number to stdout;
  - writes `reports/faz_e/01_synthetic.md` and CSVs.
- **`faz_e/02_e1_oneshot.py`:** the only script allowed to load s03–s06. Guarded by `E1_ONESHOT_GO=1`.
- **Freezing:** the sha256 of this file is logged in `paper/review/CHANGELOG_A4.md` before the battery is run. Any later change is dated in Sapmalar.

## Sapmalar (dated deviations and implementation notes)
- **2026-11-06 ~00:45–01:10 TR: implementation and battery** (workflow `phase-e-implement`; s01/s02 only).
  - `faz_e/e1_core.py`, `01_e1_synthetic.py` and `02_e1_oneshot.py` were written fresh.
  - The battery script was run several times while it was being developed; the report describes the last run. The code was edited between runs.
  - Two independent verifiers re-derived every reported number exactly:
    - a re-implementation of S_E from §2.4 that does not import `e1_core` (`faz_e/verify/verify_e1.py`, `verify_stdout.log`);
    - a code review with a brute-force check of the dynamic programme on 12,000 random cases (`faz_e/verify/review_rev_e1.py`).
  - No executed script loaded s03–s06. `02_e1_oneshot.py` was never executed with `E1_ONESHOT_GO=1`.
  - A `e1_core` bytecode file written 0.3 s after the last battery run shows one extra import. Per the verifier this was a sanity import or a refused `02` start; no session table was opened.
  - Choices the plan left open are listed in `reports/faz_e/01_synthetic.md` §6:
    - spliced real frame intervals;
    - floor midpoints;
    - mid-dwell parents are those with a Box-2 block of at least 6 frames;
    - nearest-frame tempo resampling.
- **2026-11-06: note on the §2.4 unit table.** Two rows are not identities under the frozen recurrences. In general:
  - `1-5-2` = min(mu_2 + E_2, mu_5 + E_5);
  - `1-2-5-2` = min(E_2, E_5 + mu_5).

  The code asserts this general form. "Far repeat = mass of the repeated visit" therefore holds only when that path is the cheaper one; for example, repeat_2_far has a multi-match maximum of 10.15 against 17.30 for the plain rule. The paper must not quote those rows as exact.
- **2026-11-06: tempo resampling artefact.** Nearest-frame upsampling at ×1.25–×1.70 duplicates single raw off-route frames into 2-frame runs that survive MIN_BOX = 2. This happens in parents 22, 42 and 47, with S_E between 0.80 and 1.41. The small sequence-channel rates of the slow scalings (0.02–0.06) are therefore partly an artefact of the construction.
- **2026-11-06 ~01:40 TR: AMENDMENT A3** (written on normal data only, before the one-shot). The decision on it rests with the corresponding author.
  - **What failed.** A3 as written ("mid-dwell flicker multi-match S_E ≤ 1.2 in every parent") **failed in 1 of 46 parents**. Parent 47 (fold 2) scored 1.793.
  - **Cause.** That parent's Box-2 dwell reads `223222` in raw labels. It contains one raw Box-3 frame (p3 0.70), which the Phase A debouncer had absorbed. That frame sits just before the midpoint, so the inserted 2-frame block becomes a 3-frame Box-3 visit.
  - **Where the 1.2 bound came from.** The preliminary check (`prelim/rev_checks.py`, CHECK D) *overwrote* two Box-2 frames instead of inserting them. Under that construction all 46 parents score 1.09, as the code verifier recomputed.
  - **What A3 was meant to test** is that the multi-match rule charges only the foreign visit and not the split dwell. That property holds in **46/46** composites:
    - S_E equals the mass of the off-route visit (column `offroute_mass`);
    - the plain rule charges more in 46/46 (2.67–9.64).
  - **Amended item A3'.** "In every eligible parent, the multi-match S_E of the mid-dwell flicker composite equals the mass of its off-route visit(s), and is below the plain score." It is satisfied: 46/46 and 46/46.
  - **Reporting.**
    - The method is unchanged.
    - A3 is reported as **failed as written**, together with this amendment and its reason.
    - The 3-frame case is reported as a limitation. A 3-frame stray visit with mass 1.79 exceeds the calibration maximum of 1.09 and would alarm in every fold; this sharpens R2/R3.
    - In the one-shot report, reading-rule criterion (c) counts a failed item only together with this dated amendment, and says so.
- **2026-11-06: code fixes before any one-shot**, from the verifiers' findings. None of these changes affects the battery: rerun outputs are byte-identical to the first run, archived in `reports/faz_e/archive_run1/`; the pre-fix code is in `faz_e/archive/pre_fix_2026-11-06/`.
  - (B1) `e1_core.oneshot_tables` now drops only **normal (s02)** natural_dev rows and keeps anomaly-session natural_dev rows in the rates, as Phase A did. The plan's §5.5 refers to s02 only. The smoke test in 01 now checks both cases.
  - `02` asserts that the reproduced Phase A D3 rates equal the published ones (0.078 / 0.254 / 0.475 / 0.921 / 0.929) before anything is written.
  - (M2) The `E1_ONESHOT_ACK_BATTERY` bypass was removed. Instead, every failed battery item needs a dated "AMENDMENT <item>" entry here.
  - (M3) A start sentinel `reports/faz_e/02_STARTED.sentinel` is written before any data is loaded, so a second start is refused. The suggested command uses `tee -a`.
  - (M4) `02` pins `e1_core.py` to the sha256 recorded by the battery (`ea5b9276…`).
  - Code hashes after the fixes: `e1_core.py` ea5b927640107b3ca7a9d2d32f9adfb38aa0a764a6fee9d83757185a1d87f2c6; `01_e1_synthetic.py` 50ff64e5c73a77a54d2af92fcdab0790c2863b82629c73384fd858358b4555af; `02_e1_oneshot.py` edcb4820f81b84a045584f9c64acb6898a2c7a892d697e236b298c13f14bda02.
- **2026-11-06 ~01:50 TR: crash protocol for `02`, declared before any run.**
  - **Scope.** The part of `02` that only exists there (tabulation and output, its sections 2–4) has never been executed.
  - **What happens on a crash.** If `02` fails after the Phase A gate for a reason unrelated to the method (a code error in tabulation or output):
    1. the failed console output is kept in `reports/faz_e/02_oneshot_stdout.log` (`tee -a`);
    2. the sentinel is deleted and the deletion is logged here;
    3. only the failing glue code is fixed;
    4. `02` is run again.
  - **Why a rerun is safe.** S_E, p_E and D3' are deterministic functions of frozen inputs (OOF files, folds, frozen code `e1_core.py`), so a rerun cannot change a result.
  - **What does not count.** A failure of the Phase A reproduction gate, or of the battery-consistency checks, is not a glue error. It is reported, and no E1 result is taken.
  - **Assert change.** The published-rate assert now uses a tolerance, |reproduced − published| < 5e-4, instead of `round(·, 3) ==`, so that a rounding artefact cannot spend the run.
  - **Hash.** `02_e1_oneshot.py` sha256 is now `4b070e77453c732407eb91bbb960101a9fd419cb6bfd779a398bc0913cedb959`.
- **2026-11-06 07:45 TR: go-ahead.** The corresponding author answered "Yes, run it" when asked "Shall I start the one-shot run on the anomaly sessions?". The A3 amendment stands as written above.
  - Plan frozen part: sha256 d8c2cafd…, unchanged.
  - Code: `e1_core.py` ea5b9276…; `02_e1_oneshot.py` 4b070e77….
  - Run command: `E1_ONESHOT_GO=1 bash tools/local_run.sh faz_e/02_e1_oneshot.py 2>&1 | tee -a reports/faz_e/02_oneshot_stdout.log`.
- **2026-11-06 07:46 TR: one-shot run done, once** (exit 0, 8.3 s; console output in `reports/faz_e/02_oneshot_stdout.log`; report `reports/faz_e/02_oneshot.md`).
  - Phase A gate: 1270/1270 rows.
  - E1 D3': seq135 1.000 (56 cycles), seq145 1.000 (48), fast 0.963, slow 0.929.
  - FAR 0.078, on the same four cycles as Phase A.
  - AUROC of S_E, sequence vs normal: 1.000.
  - Every §6 prediction fell inside its range. Every §7 criterion held, with (c) through AMENDMENT A3. Success as defined in §7.
  - The reading rule applies: s03/s04 are reported as consistent with the mechanism and foreseeable from the Phase A readings, not as a confirmed prediction.
  - No crash, so the crash protocol was not used. Nothing is to be re-run.
- **2026-11-06 ~08:20 TR: verification of the one-shot run and record corrections** (written after the run; changes no result).
  - Two independent checks reproduced every output row by row and recomputed the intervals (`faz_e/verify/oneshot/`):
    - an independent re-implementation of S_E;
    - an audit with brute-force alignment.
  - The run happened once: the sentinel was written at 07:45:04, the outputs at 07:45:12, and the log holds a single run.
  - **Times.** The outputs were written at 07:45:12, so "07:46" in the entry above is wrong. The go-ahead entry was appended just before the run and the result entry right after it. The file's modification time reflects only the last save, so it cannot show this order.
  - **What the audit adds:**
    - **Fast gain.** The rise in fast detection, 0.921 → 0.963, is ten pairs in fold 2. They come from 13 cycles read without Box 5, although ToF shows a Box-5 touch of 1–3 frames in each. This is a sequence-channel response, not better tempo detection. The paper says so, and the abstract no longer quotes a tempo rate for E1.
    - **Exposure.** The B2 models were trained on ToF-labelled frames of other s03–s06 cycles. The "all 104" result holds for that pooled estimator only. The abstract and the cover letter now say so.
    - **Criterion (c).** It holds through the dated AMENDMENT A3; A3 failed as written.
    - **Box-2 count.** Boxes 3 and 4 are never on the route, so a miss needs every Box-3/4 visit to vanish, not merely a Box 2 in the reading.
    - **Fold 0.** n_cal = 19, so the smallest p3' is 0.10 exactly, and alarms there rest on "≤". This is as in Phase A.
    - **Bootstrap.** The lower bounds move by up to 0.004 with the order of clusters in the bootstrap (Monte Carlo variation).
  - These notes are also in `reports/faz_e/02_oneshot_addendum.md`. The once-only outputs are not edited.

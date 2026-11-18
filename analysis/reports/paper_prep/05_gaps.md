# 05 — Closing the numeric gaps (OUTLINE §9), 2026-11-04

Scripts: `paper/scripts/gap1_minutes.py`, `gap2_wilson.py`, `gap4_minimisation.py` (run from ROOT with the local venv; they read only `reports/*`, `faz_a*/` helper modules and the raw data, read-only). Nothing was re-trained and A2.4 was not re-run. Output CSVs sit next to this file: `05_gaps_minutes_*.csv`, `05_gaps_wilson.csv`, `05_gaps_online_timing_medians.csv`, `05_gaps_minimisation_*.csv`, and the BibTeX in `05_gaps_refs.bib`. Tags: [P] primary (Faz A), [S] secondary (A2/A3), [D] descriptive.

---

## 1. Recording minutes for commissioning (A2.4) — one definition

**Why the old numbers were inconsistent (state part (b) only).** `merged_df`/the frame table appear to contain only radar frames that had at least one point (consistent with the data, but not confirmed in the logging code; correction 2026-11-05, see `faz_c/02_time_gaps.md` §5 and §7). Over the six sessions 16,274 frames were recorded, but the timestamps span about 19,075 radar periods; about 2,800 frames (14.7 %) have no row. The old state-estimation minutes in `04_commissioning.md` used `n_frames × 0.1 s` and therefore under-counted wall time (s02: 4.37 s per loop group instead of 5.30 s). The anomaly minutes (a) already used timestamp-based `dur_s` (5.00 s per predicted cycle, 5.39 s per ToF loop); for (a) the change below is the counting unit (predicted cycle -> recorded ToF loop), not the duration definition.

**Definition used everywhere below (timestamp-based).** For a loop (one `loop_id` group, 320 in total), `dur = t(first frame of the next loop_id) − t(first frame of this loop)`; for the last loop of a session, `t(last frame) + 0.1 − t(first frame)`. Loops are contiguous, so the sum over a session equals its wall time (checked: 507.8, 275.4, 272.0, 228.4, 199.9, 423.8 s). The alternative `t_last + 0.1 − t_first` differs by < 0.001 s in the means. It equals the stored `dur_s` of the ToF segmentation to 1e-15 (cross-check passed over all 51 s02 normal loops), and the stored B2×pred normal-cycle mean `dur_s` is 4.996 s.

### (a) Anomaly detector: n training + 20 calibration cycles of s02 normal [S]

s02 normal ToF loops (n = 51): mean 5.39 s, median 5.00 s, sd 2.85 s (a few long loops; IQR 4.55–5.25 s). Old frame-count mean was 4.45 s.

Closed form: minutes = (n + 20) × 5.39 s / 60. Simulation replicates the A2.4 random assignment (`RandomState([20261104, r])`, R = 50) and sums actual loop durations (the loops that had to be recorded; in the B2×pred cell calibration = all 20 cal loops, training = the distinct loops behind the first n pooled cycles).

| n training cycles | training min (ToF×tof) | calibration min (20 loops) | **total, simulated mean ± sd (ToF×tof)** | total, B2×pred (19.8 cal cycles) | closed form (n+20)×5.39 s | old doc, reconstructed as (n+20) x 5.00 s (the report states only n = 5, 10) |
|---|---|---|---|---|---|---|
| 5 | 0.45 | 1.79 | 2.24 ± 0.18 | 2.15 ± 0.19 | 2.25 | 2.1 (B2) |
| **10** | 0.89 | 1.79 | **2.68 ± 0.17** | 2.48 ± 0.19 | **2.70** | ≈ 2.5 (0.8 + 1.7) |
| 15 | 1.33 | 1.79 | 3.12 ± 0.17 | 2.82 ± 0.19 | 3.14 | 2.9 |
| 20 | 1.80 | 1.79 | 3.59 ± 0.15 | 3.11 ± 0.20 | 3.59 | 3.3 |

(± = sd over the 50 random assignments; 5th–95th percentile of the total at n = 10 is 2.46–2.88 min for ToF×tof. B2×pred is lower because pooled predicted cycles split some loops in two, so n predicted cycles come from fewer recorded loops.) File: `05_gaps_minutes_anomaly.csv`.

**Which unit the 2.7 min rests on.** 2.7 min counts 30 recorded ToF loops (what a practitioner would actually record) at the mean 5.39 s. The main cell B2×pred needs fewer loops because some loops split into two predicted cycles, giving 2.48 ± 0.19 min, essentially the old 2.5 min. Mean (5.39 s) exceeds median (5.00 s) because of a few long loops (sd 2.85 s); a reader computing 30 × 5.0 s gets 2.5 min. State both: "about 30 normal cycles (10 for the parameters, 20 for calibration), 2.5–2.7 min of recording".

**Sentence for the paper:** "about 30 recorded normal cycles (10 for the parameters, 20 for calibration), roughly 2.7 min at a mean of 5.4 s per cycle (2.5 min at the 5.0 s median)". The sd of 0.17 min is assignment randomness only, from 51 cycles of one operator; it is not a generalisation interval.

### (b) State estimator: k labelled cycles per session [S]

Mean duration of a `loop_id` group per session (timestamp-based, all groups, since the sampler draws any group; complete cycles only in brackets) and minutes for k cycles:

| session | mean loop (s) | complete-only (s) | old frame-count (s) | k = 5 min | **k = 10 min** | k = 20 min | old k = 10 min |
|---|---|---|---|---|---|---|---|
| s01 training (random picking) | 8.91 | (8.91) | 8.22 | 0.74 | 1.48 | 2.97 | 1.37 |
| s02 normal | 5.30 | (5.39) | 4.37 | 0.44 | 0.88 | 1.77 | 0.73 |
| s03 seq135 | 4.69 | (4.75) | 4.15 | 0.39 | 0.78 | 1.56 | 0.69 |
| s04 seq145 | 4.57 | (4.65) | 3.86 | 0.38 | 0.76 | 1.52 | 0.64 |
| s05 fast | 4.00 | (4.06) | 3.56 | 0.33 | 0.67 | 1.33 | 0.59 |
| s06 slow | 8.00 | (8.14) | 6.04 | 0.67 | 1.33 | 2.67 | 1.01 |

Total over all six sessions (the experiment pools them; sampling replicated exactly and validated: the replicated frame-count minutes equal the stored `rec_min_total` in 255/255 rows to 1e-15, same loop counts):

| k per session | loops | old min (frames×0.1) | **new min (timestamps), mean ± sd over 5 folds×10 reps** | closed form Σ means × k |
|---|---|---|---|---|
| 1 | 6 | 0.51 | 0.59 ± 0.17 | 0.59 |
| 2 | 12 | 1.02 | 1.21 ± 0.22 | 1.18 |
| 5 | 30 | 2.55 | 3.00 ± 0.35 | 2.95 |
| **10** | 60 | 5.01 | **5.88 ± 0.43** | 5.91 |
| 20 | 120 | 10.04 | 11.78 ± 0.56 | 11.82 |
| all training pool | 256 | 21.70 | 25.43 ± 0.21 | — |

(Per-session-mean minutes for one station, k = 10: 0.98 min averaged over the six sessions; for the normal session s02: 0.88 min.) Files: `05_gaps_minutes_state_*.csv`.

**Sentence for the paper:** "10 labelled cycles per session (about 0.7–1.5 min per session, 5.9 min across the six sessions) reach 95 % of the full-data macro-F1 (0.829 vs 0.856)". The macro-F1 values are untouched (A2_RESULTS). The claim "5.1 min" in the old report becomes 5.9 min. Caveat unchanged: the pool is six sessions of one operator, so "per station" is not demonstrated.

---

## 2. Binomial 95 % CIs (Wilson)

Counting units. **Normal cycles:** each of the 51 s02 normal cycles is a test cycle in exactly one fold, so pairs = cycles (n = 51) and the Wilson interval is exact in its unit. **Anomaly cycles:** each cycle is scored in all five folds (a different calibration set each time), so rates count (cycle, fold) pairs, n_pairs = 5 × n_cycles (e.g. 280 pairs = 56 cycles). The pair-based Wilson interval treats the five repeats as independent. It is therefore **too narrow** (anti-conservative), not conservative, for generalising to new cycles. The CSV carries three intervals: (i) Wilson on pairs (what the outline asked for), (ii) **cluster bootstrap over cycles** (10,000 resamples, stratified by anomaly type; recommended for detection rates), (iii) Wilson with n = number of distinct cycles and the same rate (conservative bound, design effect 5). All of them condition on the one operator, one calibration draw per fold and the 51-cycle normal pool. They quantify finite-sample noise only.

Write FAR as "k/51" with the interval; do not call it "nominal".

### A. Faz A [P]: D3, B2×pred, standard conformal, α = 0.10 (`05_gaps_wilson.csv`, block A)

| quantity | k / n (unit) | rate | Wilson (pairs) | cluster bootstrap (cycles) | Wilson, cycle unit (conservative) |
|---|---|---|---|---|---|
| FAR, normal cycles | 4 / 51 cycles | 0.078 | 0.031–0.185 | — | 0.031–0.185 |
| detection seq135 | 71 / 280 (56 cycles) | 0.254 | 0.206–0.308 | 0.211–0.311 | 0.158–0.381 |
| detection seq145 | 114 / 240 (48) | 0.475 | 0.413–0.538 | 0.425–0.538 | 0.341–0.613 |
| detection fast | 221 / 240 (48) | 0.921 | 0.880–0.949 | 0.862–0.967 | 0.810–0.969 |
| detection slow | 237 / 255 (51) | 0.929 | 0.891–0.955 | 0.875–0.976 | 0.825–0.973 |
| **sequence, pooled** | 185 / 520 (104) | 0.356 | 0.316–0.398 | 0.321–0.396 | 0.270–0.451 |
| **tempo, pooled** | 458 / 495 (99) | 0.925 | 0.899–0.945 | 0.885–0.960 | 0.856–0.963 |
| all anomalies | 643 / 1015 (203) | 0.633 | 0.603–0.663 | 0.608–0.660 | 0.565–0.697 |

α = 0.05: FAR 0/51 (Wilson 0–0.070); detection 0 in every group (structural: smallest attainable D3 p ≈ 0.087–0.10 with ~20 calibration cycles; see `01_conformal.md` §1). Wilson upper bounds for the 0 detection counts: 0.004–0.016 (pairs).

Wording suggestion: "4 of 51 normal cycles raised an alarm (7.8 %, 95 % Wilson CI 3.1–18.5 %), so the data cannot distinguish the nominal 10 % from roughly 3–19 %."

### B. A2.1 [S]: D3, B2×pred, α = 0.10, cross-conformal + smoothed p (headline) and neighbours

Rates are means over 200 uniform draws, so k = rate × n is an **expected count** (not an integer alarm count); the Wilson formula is applied to it. The draw band in `01_conformal.md` reflects only the randomisation, not sampling error.

| variant | FAR (n = 51) | seq pooled (n = 520 pairs / 104 cycles) | tempo pooled (495 / 99) | all (1015 / 203) |
|---|---|---|---|---|
| **cross + smoothed** | 0.091 (4.62 expected alarms), Wilson 0.038–0.200 | 0.601, Wilson 0.558–0.642 (cycle unit 0.505–0.690) | 0.946, 0.922–0.963 (0.883–0.976) | 0.769, 0.742–0.794 (0.707–0.822) |
| cross (not smoothed) | 0.059 (3/51), 0.020–0.159 | 0.438, 0.396–0.481 | 0.931, 0.906–0.950 | 0.679, 0.649–0.707 |
| smoothed (no cross) | 0.087, 0.036–0.196 | 0.571, 0.528–0.613 | 0.934, 0.908–0.953 | 0.748, 0.720–0.774 |

Per type for cross + smoothed: seq135 0.377 (0.322–0.435), seq145 0.862 (0.813–0.900), fast 0.951 (0.916–0.972), slow 0.942 (0.906–0.964); all in the CSV together with cycle-unit intervals. No bootstrap was run for block B (the p-values come from random draws that this step did not recompute); use the cycle-unit Wilson as the conservative bound. This block is a secondary analysis specified after the primary results were known.

### C. A3.3 [S]: within-cycle alarms (`03_online_summary.csv`, same (cycle, fold) pairs as A)

Alarm counts are identical to A by construction (online set = offline set in 10/10 fold×cell comparisons). CIs as in A, e.g. FAR 4/51 (0.031–0.185); alarm rate seq135 71/280, seq145 114/240, fast 221/240, slow 237/255 (same intervals as table A).

- **Equality check:** 0 discordant pairs out of 1,066 (B2×pred) and 1,086 (ToF×tof) test pairs; Wilson upper bound 0.4 % for each. Write "no discordant alarm in 1,066 test pairs (95 % upper bound 0.4 %)".
- **Share of alarms raised before the last visit began** (B2×pred): seq135 64/71 = 0.901 (0.810–0.951); seq145 102/114 = 0.895 (0.825–0.939); slow 213/237 = 0.899 (0.854–0.931); fast 35/221 = 0.158 (0.116–0.212) — fast tempo is caught at the end of the cycle, slow is not.
- **Medians with cluster bootstrap over cycles** (`05_gaps_online_timing_medians.csv`; alarm pairs, literal definition): alarm time / cycle duration — seq135 0.289 (0.277–0.311), seq145 0.293 (0.278–0.304), slow 0.317 (0.301–0.338), fast 1.025 (1.024–1.026); seconds after the true onset — seq135 0.100 (0.099–0.100), seq145 0.100 (0.100–0.101). The slow > 0.8 prediction failed (median 0.317, interval entirely below 0.5).
- ToF×tof reference counts (seq135 285/285, seq145 245/245, fast 228/245, slow 156/260; normal 4/51) are in the CSV.

---

## 3. The four 'uncertain' citations — checked 2026-11-04

Checked with the Crossref REST API (no `mailto`), the arXiv export API, the handle system, doi.org and publisher pages. Final entries are in `05_gaps_refs.bib` (keys `Singh2019RadHAR`, `Yan2025OGPCL`, `Vovk2022`, `Breque2021`).

| reference | result | action |
|---|---|---|
| RadHAR (Singh et al. 2019) | Crossref `10.1145/3349624.3356768`: Akash Deep Singh, Sandeep Singh Sandha, Luis Garcia, Mani Srivastava; *Proc. 3rd ACM Workshop on Millimeter-wave Networks and Sensing Systems* (mmNets '19, held with MobiCom '19, Los Cabos), **pp. 51–56**, ACM, issued 2019-10-07. | **Fixed**: pages and DOI confirmed. |
| OG-PCL (arXiv:2511.08910) | arXiv API: **three** authors, Jiuqi Yan, Chendong Xu, Dongyu Liu; posted 2025-11-12; no journal reference. The old "Yan, Wang, Xu, Liu" had a spurious second author. | **Fixed**: cite as Yan, Xu & Liu (2025), arXiv preprint. Update the author lists in `01_split_audit.md` (§9) and any in-text "Yan et al." is still fine. |
| Vovk, Gammerman & Shafer | Crossref `10.1007/978-3-031-06649-8` (Springer International Publishing, Cham, 2022, e-book ISBN 978-3-031-06649-8; print ISBN 978-3-031-06648-1). Catalogue records describe it as the second edition (476 pp., three new chapters). The DOI `10.1007/b106715` is the 2005 first edition. | **Fixed**: cite the 2022 second edition with DOI 10.1007/978-3-031-06649-8 (or the 2005 first edition with 10.1007/b106715, not both under one DOI). Edition wording rests on catalogue records; the Springer page itself could not be fetched (cookie redirect). |
| Breque, De Nul & Petridis (2021), Industry 5.0 | DOI `10.2777/308407` resolves (doi.org → op.europa.eu manifestation PUB_KIBD20021ENN, the EU Publications Office). The EC research page lists the report (Maija Breque, Lars De Nul, Athanasios Petridis; 5 Jan 2021) and is a stable landing page. ISBN 978-92-76-25308-2 as shown on the Publications Office detail page. Fetched on 2026-11-04: doi.org redirects (302) to the manifestation URL, and that URL loaded with title, authors, DOI and ISBN (04_cite_check.md had reported 404 earlier; it may be intermittent, so keep the EC landing page as the `url`). The publication-detail page and the EC research page also load. | **Fixed**: cite with DOI and the EC landing URL, no access-date needed beyond the usual "accessed" note. Optional: drop it, since Xu et al. (2021) already covers Industry 5.0. |

Other corrections already listed in `04_cite_check.md` and still to apply in the `.bib`: milliFlow year 2024 (not 2025); PmTrack year 2023 (not 2024) and volume 7(4); Xia & Xu (2022) has two authors; Zhang & Cao (2019) has two authors; Tesla-Rapture journal version (IEEE TMC 22(8), 4946–4960, 2023); RadMamba volume/pages unverified; page ranges for Yu 2022 (3036–3047), Yu 2010 HSMM (215–243), Shwartz-Ziv & Armon (84–90); Grinsztajn title capitalisation; duplicate Dehghani 2019 entry.

---

## 4. Data-minimisation numbers (OUTLINE §5.3), from the raw data and the cfg

Raw data: `mmwave_data.csv` (all points the chip emitted, including v = 0) and `merged_df.csv` (v = 0 points removed). Six sessions, 143,937 raw points, 114,979 merged points. Files: `05_gaps_minimisation_*.csv`.

### Geometry and range gate [D]

| quantity | value |
|---|---|
| points with range > 0.5 m | **0 of 143,937** (max range 0.496 m) — by construction: the cfg sets `cfarFovCfg -1 0 0 0.5`, an on-chip range gate of 0–0.5 m (about 11.5 range bins of 4.36 cm). The recordings contain no pre-gate data, so the share of points the gate removed cannot be computed. |
| points with z ≠ 0 / elevation ≠ 0 | **0 / 0** — by construction: `channelCfg 15 5 0` uses two TX antennas in azimuth only, so the cloud is 2D. |
| range, 0.25 / 50 / 99.75 percentile | 0.382 / 0.458 / 0.496 m |
| x, central 99.5 % band (0.125–99.875 percentile) | −0.359 to +0.299 m (full range −0.478 to +0.449 m) |
| y, central 99.5 % band | 0.316 to 0.496 m (full range 0.131 to 0.496 m); y is capped by the range gate |
| azimuth range | −74.7° to +64.7° |
| points inside the heatmap grid (x ∈ [−0.5, 0.5], y ∈ [0.1, 1.0]) | 100 % — but no point has y > 0.496, so the upper half of the 48×48 grid (y 0.5–1.0 m) is always empty. Worth one sentence in the methods ("the gate occupies the lower half of the grid"). |
| velocity = 0 points in the raw stream | 20.1 % (dropped from `merged_df`) |

### Points per frame

| statistic | raw chip stream | merged (v ≠ 0) |
|---|---|---|
| median, recorded frames | **9** | 7 |
| 95th percentile, recorded frames | **15** | 12 |
| mean, recorded frames | 8.84 | 7.07 |
| maximum | 27 | — |
| mean over all radar frames including empty ones | 7.55 | — |

Frames with no points have no row. From timestamp gaps (gaps are integer multiples of 100 ms; at most 0.06 % of gaps per session are off that grid) about **2,801 of 19,075 radar frames (14.7 %)** are empty; a gap can also be a lost UART packet, which the data cannot separate. So "points per frame" must say "per non-empty frame".

### Bytes per frame, and the ratio to raw ADC

Raw ADC (from the cfg): 2 chirps per loop × 16 loops = 32 chirps × 4 RX × 112 ADC samples × 4 B (16-bit I + 16-bit Q, `adcCfg 2 1`) = **57,344 B/frame** (573 kB/s, 2.06 GB/h at 10 Hz).

Point-cloud side (assumed TI SDK 3.6 out-of-box UART format, **documented constants, not measured on the wire**): frame header 40 B; each TLV 8 B header; detected points 16 B/point (x, y, z, v float32); side info 4 B/point (SNR, noise uint16); with this cfg (`guiMonitor -1 1 1 0 0 0 1`) the stream also carries the range profile (128 bins × 2 B) and a 24 B stats TLV; packets are padded to 32 B.

| variant (median 9 points) | bytes/frame | raw ADC ÷ this |
|---|---|---|
| V1 full on-chip stream as configured | 544 | **105 ×** |
| **V2 point-cloud TLVs only (TLV1 + TLV7, with TLV headers)** | **196** | **293 ×** |
| V3 payload the pipeline reads (x, y, v float32 + SNR uint16 = 14 B/point) | 126 | 455 × |

Sensitivity of V2 to the point count: p95 (15 points) 316 B → 181 ×; mean over all radar frames incl. empty (7.55 points) 167 B → 344 ×; maximum (27 points) 556 B → 103 ×. V2 at the median is 1.96 kB/s versus 573 kB/s.

**The outline's "320×" is not reproduced.** The defensible statement is "about 290× less data per frame (196 B versus 57 kB at the median of nine points per frame), roughly 100× if the unused range-profile and statistics TLVs of the default stream are counted". Use "two orders of magnitude" if a single round number is wanted. Correct the OUTLINE §5.3 accordingly. Assumptions to state in the paper: 16-bit complex samples; SDK 3.6 TLV layout; 10 Hz; the raw-data capture board streams all chirps of the frame.

---

## 5. Items to forward to the drafting step

1. Anomaly commissioning: "≈ 2.5–2.7 min" for 30 cycles (2.7 = recorded ToF loops at 5.39 s mean; 2.5 = B2×pred cell / 5.0 s median); the old frame-count error affected only the state part and "5.1 min" by "5.9 min" (k = 10 per session, six sessions); macro-F1 and AUROC values do not change.
2. Pair-based Wilson CIs are anti-conservative (5 repeats per anomaly cycle), not conservative as the task text assumed. Report FAR as k/51 with Wilson CI; do not use "nominal". Detection rates: cluster-bootstrap or cycle-unit Wilson, never the pair-based Wilson alone.
3. Fix the four references (`05_gaps_refs.bib`).
4. Replace "320×" with "about 290×" (V2) and state the assumptions; add the sentence on the on-chip 0.5 m range gate and the 2D geometry (both constructed by configuration, not measured reductions).
5. The upper half of the 48×48 heatmap grid is never populated (y ≤ 0.496 m); say so rather than leaving "Y ∈ [0.1, 1.0] m" unqualified.

# Analysis code, folds, out-of-fold predictions and analysis plans (release 2.0.0)

This folder holds the analysis behind the article

> Baran Kaynak, Sümeyye Kaynak, Andrew Kusiak. **Detecting sequence and timing errors in manual assembly with a low-cost mmWave radar.** Submitted.

The radar is a TI IWR6843ISK. Only its on-chip point cloud is used. Time-of-flight (ToF) sensors (VL53L0X) over the parts boxes give frame labels. The labels are used for training, calibration and evaluation only.

The raw data are in the dataset record (Zenodo concept DOI **10.5281/zenodo.20323005**). The article uses dataset version 1.0.0 (DOI 10.5281/zenodo.20523433). In the dataset repository this folder sits in `analysis/`, next to the `data/` folder. This folder contains no raw radar or ToF data. Corrections to the v1.0.0 description are in the repository's main `README.md` (section "Corrections in version 2.0.0").

## 1. Contents

| Folder or file | Content |
|---|---|
| `probe/` | First look at the data: dwell modes, tempo scaling, few-shot calibration. `probe/common.py` loads the data for all later scripts. |
| `faz_a/` | Phase A, the primary analysis. Fold construction, state estimators (B1, B2, CNN-only, CNN-GRU), cycle scores (D0, D1, D2), conformal p-values, alarm rule D3, and the initial TFT + z-score pipeline as a comparison. Plan: `faz_a/DESIGN.md`. |
| `faz_a2/` | Phase A2 (secondary). Equivalence check, conformal variants, leakage with five seeds, label sensitivity, commissioning, latency, and a descriptive summary of the 50 commissioning resamples (`06_resample_summary.py`). Plan: `faz_a2/DESIGN_A2.md`. |
| `faz_a3/` | Phase A3 (secondary). Error propagation, operating envelope, within-cycle alarms. Plan: `faz_a3/DESIGN_A3.md`. |
| `faz_a4/` | Phase A4 (secondary). Session confound, tempo channels, leakage with matched hardware and the Keras architecture, estimators that never saw the deviation sessions. Three descriptive scripts added later: `05_d0_sequence.py` (duration score D0 on the sequence sessions), `06_measurement.py` (radar configuration, B2-vs-ToF timing agreement, alarm latency in seconds), `07_ncal_counts.py` (number of calibration cycles per fold). Plan: `faz_a4/DESIGN_A4.md`. |
| `faz_b/` | Phase B (exploratory). Further checks (`00_referee_leftovers.py`: D0 + D1 combination, windows with missing rows, time-ordered calibration), reduced random forests, sensing budget of the point cloud, frame rate. Plan: `faz_b/DESIGN_B.md`. |
| `faz_c/` | Phase C (exploratory). Export of the reduced forest to C with measured compiled sizes (generated C in `faz_c/c/`, out-of-fold predictions of the C code in `faz_c/oof/`), and the analysis of gaps in the recording. Plan: `faz_c/DESIGN_C.md`. |
| `faz_d/` | Phase D (secondary). PyTorch CNN-GRU, random window split, seeds 1 to 4, re-run on the RTX 3090 so that all network runs in the article use one GPU; new leakage table. Plan: `faz_d/DESIGN_D.md`. |
| `faz_e/` | Phase E (secondary). An evidence-weighted sequence score (E1): core code `e1_core.py`, validation on synthetic deviations built from normal cycles (`01_e1_synthetic.py`), one run on all sessions (`02_e1_oneshot.py`), intervals (`03_e1_intervals.py`). `00_BRIEF_seq.md` records the diagnosis and the quantities seen before the plan was written. Plan: `faz_e/DESIGN_E.md`. |
| `faz_f/` | Phase F. Reference detectors on the same cycles (exact-route rule, edit distance, isolation forest; descriptive), and CNN-GRU under the cycle-grouped split with more seeds (secondary). Plan: `faz_f/DESIGN_F.md`. |
| `faz_g/` | Phase G (secondary). One run of E1 with a state estimator that never saw the deviation sessions. Plan: `faz_g/DESIGN_G.md`. |
| `reports/<phase>/` | Outputs of each phase: Markdown reports, CSV tables, and `oof/` folders. `*_RESULTS.md` files summarise Phases A to C. |
| `reports/*/oof/*.csv.gz` | Out-of-fold predictions, one file per (model, split, fold): class probabilities and predicted state for every window. |
| `reports/faz_a/folds_state.csv`, `folds_anom.csv`, `split_s0.csv` | Fold files: state-estimation splits S0, S1, S2; anomaly folds, roles and calibration assignment. |
| `reports/faz_a2/03_labels_*` | Alternative label variants (label sensitivity). |
| `reports/faz_a/05_old_*` | Initial TFT + z-score pipeline on the cycle tables, in three state-sequence variants (default, `_deb` debounced, `_none` unsmoothed). |
| `reports/paper_prep/01_split_audit.md`, `04_cite_check.md` | Literature audit of split practice and a citation check. |
| `reports/paper_prep/05_gaps*` | Data-minimisation tables, recording minutes, Wilson intervals, online timing medians. |
| `reports/_logs/` | Run logs of the GPU runs: PyTorch Phase A run and Keras run (both cited in the article for software versions), Phase D run, Phase F runs. |
| `paper/scripts/` | Scripts that make the article's tables (`make_tables.py`, `make_tables_a4.py`, `make_table_e1.py`) and figures (`fig0_overview.py` to `fig8_commissioning.py`), the scripts behind `reports/paper_prep/05_gaps*` (`gap1_minutes.py`, `gap2_wilson.py`, `gap4_minimisation.py`), and the shared `style.py`. |
| `paper/tables/` | The LaTeX tables as written by these scripts for the submitted manuscript. |
| `faz_a2/fazA_md5_before.txt` | MD5 snapshot of the Phase A out-of-fold files taken before Phase A2 ran (read by `faz_a2/02_leakage.py`). |
| `requirements*.txt` | Package versions (section 4). |
| `MANIFEST.sha256` | SHA-256 of every file in this folder except itself. |

Plans and many reports are in Turkish; newer ones are in English.

## 2. Analysis plans

Each phase has a dated plan: `faz_a/DESIGN.md`, `faz_a2/DESIGN_A2.md`, `faz_a3/DESIGN_A3.md`, `faz_a4/DESIGN_A4.md`, `faz_b/DESIGN_B.md`, `faz_c/DESIGN_C.md`, `faz_d/DESIGN_D.md`, `faz_e/DESIGN_E.md`, `faz_f/DESIGN_F.md`, `faz_g/DESIGN_G.md`. Each plan was written before the results of its phase were seen. Later changes are added, dated, in the section "Sapmalar" (deviations) at the end of each plan. The status line at the top of each plan says whether the phase is primary, secondary, exploratory or descriptive, and what was known when it was written. Phase A is the only primary analysis. All later phases were planned after the Phase A results were known.

Four plans were edited for this release: a local path, an IP address, or one sentence or clause of a deviation-log entry. The SHA-256 of the original files are:

- `faz_a2/DESIGN_A2.md` 43e13335237182826b8af2094426534da05fd9c502d378eef76b6f149f401ad2
- `faz_a4/DESIGN_A4.md` 1b0d6b816fc235e8568c9d7856a99138db3705b650cca3cf30a40a2ef8a2a2dc
- `faz_b/DESIGN_B.md` 2beecb75099ba9f4366247f66d0e8db54544544c81d2f05c4dc63416ea872e18
- `faz_c/DESIGN_C.md` 312636d315d6198cb559a769cf232793d493afb3a53f4c5bd267b13a9c2112bc

The other six plans are shipped as written. The Phase E code (`faz_e/e1_core.py`) checks the hash of the part of `faz_e/DESIGN_E.md` above "Sapmalar".

## 3. Mapping to the article

Numbers of tables and figures follow the submitted manuscript. Supplementary tables are named by their caption.

### Figures

| Figure | Script | Inputs |
|---|---|---|
| 1 Data flow of the method | `paper/scripts/fig0_overview.py` | `reports/faz_e/02_oneshot_cycles.csv.gz`, `reports/faz_a/04_cycle_scores.csv.gz`, `reports/faz_a/oof/b2__S1__f*.csv.gz`, dataset |
| 2 Set-up and pipeline | `fig1_setup.py` | set-up photographs (not in this folder; see `PHOTO_DIR`) |
| 3 Evaluation protocols | `fig2_protocol.py` | `reports/faz_a/04_anomaly.md`, `reports/faz_a3/03_online.md`, `reports/faz_a/folds_anom.csv` |
| 4 Split protocol and ranking | `fig3_ranking.py` | `reports/faz_d/02_leakage_per_run.csv`, `02_leakage_summary.csv` |
| 5 Cycle-level scores of the detector | `fig4_scores.py` | `reports/faz_a/04_cycle_scores.csv.gz`, `04_anomaly_table.csv` |
| 6 Frame errors and detection | `fig5_error_propagation.py` | `reports/faz_a3/01_error_propagation*`, `reports/faz_a/04_anomaly_table.csv` |
| 7 Within-cycle alarms | `fig6_online.py` | `reports/faz_a3/03_online_*.csv` |
| 8 Operating envelope | `fig7_envelope.py` | `reports/faz_a3/02_envelope_*.csv` |
| 9 Commissioning cost | `fig8_commissioning.py` | `reports/faz_a2/04_comm_*_summary.csv`, `reports/faz_a/04_anomaly_table.csv` |

### Tables

| Table | Script and output | Inputs |
|---|---|---|
| 1 Radar configuration | written by hand from `reports/faz_a4/06_measurement.md` (made by `faz_a4/06_measurement.py`) | radar profile of the dataset |
| 2 Recording sessions | `make_tables.py` → `tab1_sessions.tex` | `reports/probe/00_schema.md` |
| 3 State estimation | `make_tables.py` → `tab2_state.tex`, `tab2b_state_sessions.tex` | `reports/faz_a/03_state_table.csv` |
| 4 Leakage, two CNN-GRU implementations and B2 | `make_tables_a4.py` → `tab_leak_matched.tex` | `reports/faz_a4/03_leakage_matched_keras_per_run.csv`, `reports/faz_d/02_leakage_per_run.csv` |
| 5 Cycle-level anomaly detection | `make_tables.py` → `tab3_anomaly.tex`, then `make_table_e1.py` adds the E1 rows | `reports/faz_a/04_anomaly_table.csv`, `reports/faz_a2/01_conformal_table.csv`, `reports/faz_e/03_table_metrics.csv` |
| 6 Estimators that had not seen the deviation sessions | `make_tables_a4.py` → `tab_unseen.tex` | `reports/faz_a4/04_unseen_*.csv`, `reports/faz_g/01_e1_ii_summary.csv` |
| 7 CPU inference time | `make_tables.py` → `tab4_latency.tex` | `reports/faz_a2/05_latency.md` |
| S: State estimation per S1 fold | `make_tables_a4.py` → `supp_keras_folds.tex` | `reports/faz_a4/03_leakage_matched_keras_per_run.csv` |
| S: Opposite-channel negative controls | `make_tables_a4.py` → `supp_opposite.tex` | `reports/faz_a4/01_session_confound_b_rates.csv` |
| S: Step durations per session | `make_tables_a4.py` → `supp_steps.tex` | `reports/faz_a4/01_session_confound_c_steps.csv` |
| S: Channel composition of tempo alarms | `make_tables_a4.py` → `supp_tempo.tex` | `reports/faz_a4/02_tempo_*.csv` |
| S: Single channels and D3 with three estimators | `make_tables_a4.py` → `supp_unseen_detail.tex` | `reports/faz_a4/04_unseen_anomaly_table.csv` |
| S: Per-fold diagnosis of the sequence channel D1 | written by hand | `faz_e/00_BRIEF_seq.md` (section 4), `reports/faz_a/04_cycle_scores.csv.gz` |
| S: Synthetic deviations; one-shot run (E1) | written by hand | `reports/faz_e/01_synthetic*`, `02_oneshot*`, `03_intervals*` |
| S: Alarms at or before the wrong-box entry | written by hand | `reports/faz_a4/01_session_confound*` |
| S: Constant data of forests exported to C | written by hand | `reports/faz_c/01_export_c*` |
| S: Reference detectors | written by hand | `reports/faz_f/01_cycle_baselines.md` |

### Results by section

| Article section | Main code | Main outputs |
|---|---|---|
| System and data; what the radar stream contains | `probe/00_load.py`, `faz_a4/06_measurement.py`, `faz_c/02_time_gaps.py`, `paper/scripts/gap4_minimisation.py` | `reports/probe/00_schema.md`, `reports/faz_a4/06_measurement.md`, `reports/faz_c/02_time_gaps.md`, `reports/paper_prep/05_gaps_minimisation_*.csv` |
| Methods; evaluation design | `faz_a/00_folds.py`, `fa_common.py`, `01_state_cnn.py`, `02_state_baselines.py`, `04_cycle_anomaly.py`, `faz_a2/a2_cycles.py` | `reports/faz_a/00_folds.md`, fold files |
| Which state estimator, and does the split protocol change the answer? | `faz_a/03_state_report.py`, `faz_a2/02_leakage.py`, `11_baselines_s0_seeds.py`, `12_cnn_s0_seeds.py`, `faz_a4/03_leakage_matched_keras.py`, `14_keras_*.py`, `faz_d/*.py`, `faz_f/02_cnn_s1_seeds.py`, `03_s1_seeds_summary.py` | `reports/faz_a/03_state*`, `reports/faz_a2/02_leakage*`, `reports/faz_a4/03_*`, `reports/faz_d/`, `reports/faz_f/03_s1_seeds.md` |
| Does the detector catch order and pace errors? | `faz_a/04_cycle_anomaly.py`, `05_old_pipeline.py`, `faz_a2/01_conformal.py`, `06_resample_summary.py`, `faz_a3/01_error_propagation.py`, `faz_a4/02_tempo_channels.py`, `paper/scripts/gap2_wilson.py` | `reports/faz_a/04_*`, `05_old*`, `reports/faz_a2/01_conformal*`, `06_resample_summary*`, `reports/faz_a3/01_*`, `reports/faz_a4/02_*` |
| Why were sequence errors missed, and can the decision rule be changed? | `faz_e/*.py`, `faz_f/01_cycle_baselines.py` | `reports/faz_e/`, `reports/faz_f/01_cycle_baselines*` |
| Can we trust the result? | `faz_a4/01_session_confound.py`, `04_unseen_sessions.py`, `05_d0_sequence.py`, `faz_g/01_e1_estimator_ii.py` | `reports/faz_a4/01_*`, `04_unseen_*`, `05_d0_sequence.md`, `reports/faz_g/` |
| Is it deployable? | `faz_a3/02_envelope.py`, `03_online.py`, `faz_a2/04_commissioning.py`, `05_latency.py`, `faz_a4/07_ncal_counts.py`, `faz_b/*.py`, `faz_c/01_export_c.py`, `paper/scripts/gap1_minutes.py` | `reports/faz_a3/02_*`, `03_*`, `reports/faz_a2/04_*`, `05_latency*`, `reports/faz_a4/07_ncal_counts.md`, `reports/faz_b/`, `reports/faz_c/`, `reports/paper_prep/05_gaps_minutes_*` |
| Supplementary: literature audit of split practice | (manual, with a Crossref check) | `reports/paper_prep/01_split_audit.md`, `04_cite_check.md` |

## 4. How to run

### Python and packages

- Python 3.12. The main environment was Python 3.12.13.
- `requirements-main-venv.txt`: exact versions of the main environment (numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, statsmodels 0.15.0, pyarrow 26.0.1, torch 2.6.0+cu124). They were read from the installed package metadata.
- `requirements.txt`: the project's first list. It also names `pytorch-forecasting==1.8.0` and `lightning==2.6.6`. These belong to the initial TFT pipeline (`faz_a/05_old_pipeline.py`) and were not in the local main environment.
- Keras: `requirements-keras-linux.txt` (tensorflow 2.18.0, keras 3.9.2, numpy 2.0.2, pandas 3.0.6) for Linux. `requirements-keras-venv.txt` is the local Windows CPU copy; it pins `tensorflow_intel` and does not install on Linux.
- `faz_c/01_export_c.py` needs a C compiler (`gcc`) and the `size` tool.
- The figure scripts need matplotlib and Pillow, and the Arial font (else matplotlib uses a fallback font).

### GPU runs

| Runs | GPU | Framework | cuDNN |
|---|---|---|---|
| PyTorch CNN-GRU, Phase A (S1 folds, S0 seed 42) | RTX 3090 | PyTorch 2.6.0+cu124 | not recorded |
| PyTorch CNN-GRU, Phase D (S0 seeds 1 to 4) | RTX 3090 | PyTorch 2.6.0+cu124, CUDA 12.4 | 9.1.0 (`reports/faz_d/01_env.json`) |
| PyTorch CNN-GRU, Phase F (S1, more seeds) | RTX 3090 | PyTorch 2.6.0+cu124 (`reports/_logs/job-20261110-*.log`) | not recorded |
| Keras CNN-GRU (Phase A4) | RTX 3090 | TensorFlow 2.18.0, Keras 3.9.2, Python 3.12.3 (`reports/_logs/job-20261105-003141-626-10306.log`) | 9.3.0 |
| Earlier PyTorch runs: S0 seeds 1 to 4 (`faz_a2/12_cnn_s0_seeds.py`) and an S1 re-run (`faz_a4/13_cnn_s1_local.py`) | GTX 1650 SUPER | PyTorch 2.6.0+cu124 | 9.1.0 |

The article's results use the RTX 3090 runs. The GTX 1650 SUPER runs were used in an earlier version of the leakage comparison; the Supplementary Material lists that result as superseded by the single-GPU re-run (Phase D). Their outputs are kept because later scripts read them. Networks trained on a GPU are not bit-reproducible across machines.

### Data

1. Get dataset v1.0.0 (DOI 10.5281/zenodo.20523433) or the dataset repository. The data folder must hold six folders `session_*`, each with `mmwave_data.csv`, `distance_sensors.csv` and `merged_df.csv`.
2. Set `DATA_PATH` to that `data/` folder. `probe/common.py` reads `DATA_PATH` and finds the sessions as the folders `session_*` that hold `merged_df.csv`. If `DATA_PATH` is not set, the scripts look for `data/` inside this folder. That is wrong in the dataset repository, where this folder is `analysis/`. There, set `DATA_PATH` to the repository's `data/` folder, for example `DATA_PATH=../data` when you run from `analysis/`.

### Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `DATA_PATH` | all scripts that read data | folder with the six `session_*` folders (see above) |
| `CACHE_DIR` | `faz_a/fa_common.py` and the network scripts | where the heatmap cache (`heatmaps_*.npy`, several hundred MB) is written; default `.cache/` inside this folder |
| `RADAR_CFG` | `paper/scripts/gap4_minimisation.py`, `faz_a4/06_measurement.py` | radar profile `iwr6843isk_radar_profile.cfg`; default `$DATA_PATH/../sensor_config/iwr6843isk_radar_profile.cfg`, which is where the dataset keeps it |
| `PHOTO_DIR` | `paper/scripts/fig1_setup.py` | folder with the four set-up photographs `img_side_view.jpg`, `img_step1.jpg`, `img_mmwave_sensor.jpg`, `img_tof_sensor.jpg` (not in this folder); default `photos/` inside this folder |
| `PYTHON_TF` | `faz_a4/14_keras_launch.py` | interpreter of the TensorFlow 2.18 / Keras 3.9.2 environment |
| `GCC`, `SIZE_TOOL` | `faz_c/01_export_c.py` | C compiler and `size` tool; default `gcc` and `size` on `PATH`. The article run used x86-64 MinGW-w64 gcc with `-Os`. |
| `LEAK_DIR` | `paper/scripts/fig3_ranking.py` | report folder with the leakage table; default `faz_d` |
| `E1_ONESHOT_GO=1` | `faz_e/02_e1_oneshot.py` | the script refuses to run without it; see below |
| `E1_II_GO=1` | `faz_g/01_e1_estimator_ii.py` | the script refuses to run without it; see below |
| `OMP_NUM_THREADS` | CPU scripts | the CPU scripts were run with at most 6 threads; the latency runs with 1 |

### Paths

The scripts find the root of this folder from their own location (`Path(__file__).parents[...]`). They read and write `reports/<phase>/` and `paper/tables/` below it. So the folder layout must stay as shipped. The layout is the same as in the original project folder, so no path needs editing apart from the variables above. Run the scripts from this folder. Work in a copy: every script overwrites its shipped outputs. The style module `paper/scripts/style.py` creates `paper/figures/` when imported.

Some docstrings and plans name `tools/local_run.sh`, `tools/local_job.py` or `tools/remote_*.sh`. These were local and remote wrappers that set the variables above. They are not shipped. `faz_a4/14_keras_remote.py` was the launcher on the remote Linux machine and assumes that machine's folders. `faz_c/02_time_gaps.py` also reads the manuscript sources in its last part. The manuscript is not part of this folder, so that part cannot be regenerated here.

### Order of the scripts

Each script has a docstring with its arguments.

1. `probe/00_load.py` (schema check), then `faz_a/00_folds.py` (writes the fold files).
2. `faz_a/01_state_cnn.py`, `02_state_baselines.py`, `03_state_report.py`.
3. `faz_a/04_cycle_anomaly.py`, `05_old_pipeline.py`.
4. Phase A2: `faz_a2/00_equivalence.py`, `00b_cycles_selftest.py`, `01_conformal.py`, `02_leakage.py`, `03_labels.py`, `04_commissioning.py`, `05_latency.py`, `11_baselines_s0_seeds.py`, `12_cnn_s0_seeds.py`, `06_resample_summary.py`.
5. Phase A3: `faz_a3/01_error_propagation.py`, `02_envelope.py`, `03_online.py`.
6. Phase A4: `faz_a4/01_session_confound.py`, `02_tempo_channels.py`, `13_cnn_s1_local.py`, `14_keras_launch.py` or `14_keras_train.py`, `03_leakage_matched_keras.py`, `04_unseen_sessions.py`, then the descriptive `05_d0_sequence.py`, `06_measurement.py`, `07_ncal_counts.py`.
7. Phase B: `faz_b/00_referee_leftovers.py`; `01_tiny_forests.py`, then `01b_tiny_forests_report.py`; `02_sensing_budget.py` (helper `b_frames.py`, gate `02a_gate.py`), `02c_noise_floor.py`, `02d_level_seeds.py`, then `02b_sensing_budget_report.py`; `03_frame_rate.py`, then `03b_frame_rate_report.py`.
8. Phase C: `faz_c/01_export_c.py` (needs the Phase B forest outputs and a C compiler), then `faz_c/02_time_gaps.py`. `faz_c/02_time_gaps_search.txt` is the raw listing of the search for the recording code, with local paths replaced by placeholders.
9. Phase D: `faz_d/01_cnn_s0_seeds_3090.py` (GPU), then `faz_d/02_leakage_3090.py --source d`.
10. Phase E: `faz_e/01_e1_synthetic.py` (normal sessions s01 and s02 only), then `faz_e/02_e1_oneshot.py` (all sessions), then `faz_e/03_e1_intervals.py`.
11. Phase F: `faz_f/01_cycle_baselines.py` (CPU); `faz_f/02_cnn_s1_seeds.py` (GPU), then `03_s1_seeds_summary.py`.
12. Phase G: `faz_g/01_e1_estimator_ii.py`.
13. Tables and figures: `paper/scripts/make_tables.py`, `make_table_e1.py` (after `make_tables.py`, because it adds rows to `tab3_anomaly.tex`), `make_tables_a4.py`, `gap1_minutes.py`, `gap2_wilson.py`, `gap4_minimisation.py`, `fig0_overview.py` to `fig8_commissioning.py`.

**Single-run scripts.** The plans allow `faz_e/02_e1_oneshot.py` and `faz_g/01_e1_estimator_ii.py` to run once. Each script runs only with its variable (`E1_ONESHOT_GO=1`, `E1_II_GO=1`). It writes a start file (`*_STARTED.sentinel`) before it loads data and refuses to start if that file or any of its outputs exists. The start files are not shipped. To repeat the run, use an empty copy of the output folder.

**Removed columns.** Some per-cycle columns were removed from the shipped copies of several tables, and one row was removed from `reports/faz_a4/01_session_confound_c_cycles.csv`. The changed files include `reports/faz_a/folds_anom.csv`, `reports/probe/00_loops.csv`, `reports/faz_a/04_cycle_scores.csv.gz`, `04_anomaly_table.csv`, `04_cycle_counts.csv` and some `reports/faz_a2/` tables. Scripts that read these columns need the full files. So run `probe/00_load.py`, `faz_a/00_folds.py` and `faz_a/04_cycle_anomaly.py` first; they write the full files from the data. Section 6 lists the scripts that read them. In a check of an earlier draft of this release (5 November 2026), `make_tables.py`, `make_tables_a4.py` and `gap2_wilson.py` ran on the shipped files. Both table scripts were changed later and were not re-run for this release.

## 5. What was checked for this release

The analysis scripts were not re-run for this release. The outputs are the files written in the project folder. The tables in `paper/tables/` are the files used for the submitted manuscript. Every `.py` file passes a Python syntax check (`py_compile`). The only code changes are path constants (section 6).

## 6. Changes made for publication

- Local and remote absolute paths, user names, host names and an IP address were replaced by placeholders such as `<ROOT>`, `<LOCAL_HOME>`, `<REMOTE_HOME>`, `<REMOTE_HOST>`, `<LOCAL_HOST>`.
- Hard-coded paths in code became the environment variables of section 4 (`DATA_PATH`, `CACHE_DIR`, `RADAR_CFG`, `PHOTO_DIR`, `PYTHON_TF`, `GCC`, `SIZE_TOOL`). The program logic is unchanged.
- Because of these edits, hashes of the edited files taken in the project folder will not match the files here. The out-of-fold prediction files are unchanged.
- The normal session s02 contains a few cycles in which the operator deviated from 1-2-5 without being asked to. The analyses keep them in the pool and do not report their alarm status. Columns, rows and sentences that marked these cycles or gave their number were removed from the shipped reports and tables. Code and plans that refer to them are unchanged.
- Not shipped: the manuscript and its sources, internal working notes, execution wrappers (`tools/`), caches and virtual environments, preliminary method-search logs (`faz_e/prelim/`), superseded copies (`archive/` folders), start files of single-run scripts, and the scripts and reports of independent re-computations (`verify_*`, `vr_*`, `verify/` folders). Some plans and `*_RESULTS.md` files refer to these files by name.

**Scripts that need regenerated files.** `probe/00_load.py`, `faz_a/00_folds.py` and `faz_a/04_cycle_anomaly.py` write the full versions of the changed tables. A text search finds the removed column names in at least these other scripts (scripts that get the columns through `faz_a2/a2_cycles.py`, such as `faz_a3/03_online.py`, are not all listed): `faz_a/fa_common.py`, `faz_a2/00b_cycles_selftest.py`, `01_conformal.py`, `04_commissioning.py`, `a2_cycles.py`, `faz_a3/01_error_propagation.py`, `faz_a4/01_session_confound.py`, `02_tempo_channels.py`, `04_unseen_sessions.py`, `06_measurement.py`, `faz_b/00_referee_leftovers.py`, `faz_e/01_e1_synthetic.py`, `02_e1_oneshot.py`, `e1_core.py`, `faz_f/01_cycle_baselines.py`, `faz_g/01_e1_estimator_ii.py`, `paper/scripts/fig0_overview.py`, `probe/02_tempo_scaling.py`, `03_few_shot.py`, `common.py`. Run them after the three scripts above.

## 7. Licence

- Code (all `*.py` files and the generated C code in `faz_c/c/`): MIT License, see `LICENSE`.
- Everything else (derived tables, fold files, out-of-fold predictions, reports, analysis plans, literature notes): Creative Commons Attribution 4.0 International (CC BY 4.0), the licence of the dataset record.

## 8. Citation

Cite the article and the dataset record (concept DOI 10.5281/zenodo.20323005 for all versions).
